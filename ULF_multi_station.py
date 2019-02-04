#!/usr/bin/env python37-32
#-*- coding: utf-8 -*-

"""
Created on Wed Oct 08 18:48:01 2018
Last modified on 11/08/2018

@author (orignial): tyler_chapman

This program generates plot of multiple stations over 24 hr periods.
"""
#generic
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import pylab
import datetime as datetime
import sys
import time
from pathlib import Path

#Specific to this script
from UNH_analysis import ULF_txt2npy
from scipy.signal import savgol_filter
from itertools import cycle

#------------------------------------------------------------------------------#
#Functions

def ULF_multi_station(yyyy, mm, dd, UTC_start, UTC_end, showplot=False, saveplot=False, smoothed=False, showstations=False, window=11, offset=False):

    #DESCRIPTION: combines data for different stations.

    #------------------------------------------------------------------------------#
    #Variables and settings

    if UTC_start >= UTC_end:
        print('ERROR: Ending hour must be greater than starting hour.')
        sys.exit()

    np_LYR = ULF_txt2npy(yyyy, mm, dd, station='LYR')
    dat_LYR = np.load(np_LYR)

    np_NAL = ULF_txt2npy(yyyy, mm, dd, station='NAL')
    dat_NAL = np.load(np_NAL)

    np_HOR = ULF_txt2npy(yyyy, mm, dd, station='HOR')
    dat_HOR = np.load(np_HOR)

    idx_start = UTC_start*36000
    idx_end = UTC_end*36000

    x_mag = np.array([])
    y_mag = np.array([])

    d1= dat_LYR[idx_start:idx_end,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)

    x_NAL = dat_NAL[idx_start:idx_end,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    x_LYR = dat_LYR[idx_start:idx_end,1]
    x_HOR = dat_HOR[idx_start:idx_end,1]
    y_NAL = dat_NAL[idx_start:idx_end,2] #y-axis signal
    y_LYR = dat_LYR[idx_start:idx_end,2]
    y_HOR = dat_HOR[idx_start:idx_end,2]

    x_mag = np.column_stack((x_NAL, x_LYR, x_HOR))
    y_mag = np.column_stack((y_NAL, y_LYR, y_HOR))

    hours = UTC_end - UTC_start
    x_ticks = np.arange(0,(len(d1)),(len(d1)/10/hours))
    x_tick_labels = np.arange((UTC_start),(UTC_end+1),1)

    time_list = np.arange(0,(3600*hours),.1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**11 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/1.2 #overlap in fft segments, of length nfft
    f_max = 0.1 #max frequency plotting bound [Hz]
    b_max = 0.8 #+/- db/dt plotting range [nT/s]

    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    my_dpi = 120
    colors = cycle(["red", "green", "blue", "fuchsia", "gray", "black", "lime", "maroon", "navy", "olive", "purple", "aqua", "silver", "teal", "yellow"])
    #-------------------------------------------------------------------------------#
    #Main

    #calculations
    x_avg = (x_mag[:,0] + x_mag[:,1] + x_mag[:,2])/3
    y_avg = (y_mag[:,0] + y_mag[:,1] + y_mag[:,2])/3

    x_avg_3 = np.c_[x_avg, x_avg, x_avg]
    y_avg_3 = np.c_[y_avg, y_avg, y_avg]

    if offset == True: #adds offset in data (verticle) to allow for better plotting visualization
        x_NAL = x_NAL + 0.2
        x_LYR = x_LYR + 0.4
        x_HOR = x_HOR + 0.6
        y_NAL = y_NAL + 0.2
        y_LYR = y_LYR + 0.4
        y_HOR = y_HOR + 0.6
        x_mag = np.column_stack((x_NAL, x_LYR, x_HOR))
        y_mag = np.column_stack((y_NAL, y_LYR, y_HOR))

    x_diff = x_mag - x_avg_3
    y_diff = y_mag - y_avg_3

    #plotting
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = str(mm) + '/' + str(dd) + '/' + str(yyyy) + ' at NAL/LYR/HOR from ' + str(UTC_start) + ':00-' + str(UTC_end) + ':00 UTC'
    plt.suptitle(title)

    #First Column of plot
    ax1 = fig.add_subplot(4,1,1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('dBx/dT [nT/sec]')
    ax1.set_ylim(-0.2,b_max)

    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_avg, nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('Freq [Hz]')
    ax2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    ax3 = fig.add_subplot(4,1,3, sharex=ax1)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('dBy/dT [nT/sec]')
    ax3.set_ylim(-0.2,b_max)

    ax4 = fig.add_subplot(4,1,4, sharex=ax1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_avg, nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel('Freq [Hz]')
    ax4.set_ylim(0,f_max)

    stations= ['NAL', 'LYR', 'HOR']
    if smoothed == True:
        x_mag_smoothed = savgol_filter(x_avg, window, 0, axis=0)
        y_mag_smoothed = savgol_filter(y_avg, window, 0, axis=0)
        ax1.plot(time_list, x_mag_smoothed, label='AVG', color=plotcolor, linewidth=ln_wth)
        ax3.plot(time_list, y_mag_smoothed, label='AVG', color=plotcolor, linewidth=ln_wth)

        if showstations == True:
            for i, item in enumerate(stations):
                Hz_color = next(colors)
                x_pwr_smooth = savgol_filter(x_diff[:,i], window, 0) #smoothing out the periodogram (window size 11, polynomial order 0)
                ax1.plot(time_list, x_pwr_smooth, label=str(item)+' diff', color=Hz_color, linewidth=ln_wth)
                y_pwr_smooth = savgol_filter(y_diff[:,i], window, 0)
                ax3.plot(time_list, y_pwr_smooth, label=str(item)+' diff', color=Hz_color, linewidth=ln_wth)
    else:
        ax1.plot(time_list, x_avg, label='AVG', color=plotcolor, linewidth=ln_wth)
        ax3.plot(time_list, y_avg, label='AVG', color=plotcolor, linewidth=ln_wth)
        if showstations == True:
            for i, item in enumerate(stations):
                Hz_color = next(colors)
                ax1.plot(time_list, x_diff[:,i], label=str(item)+' diff', color=Hz_color, linewidth=ln_wth)
                ax3.plot(time_list, y_diff[:,i], label=str(item)+' diff', color=Hz_color, linewidth=ln_wth)

    ax1.legend(loc="upper right", fontsize=6)
    ax3.legend(loc="upper right", fontsize=6)
    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/SD/'
    UTC_range = 'from_' + str(UTC_start) + '_to_' + str(UTC_end)

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + UTC_range + '_SD.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + UTC_range + '_SD.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + UTC_range + '_SD.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + UTC_range + '_SD.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig.savefig(png_file, bbox_inches='tight')


#------------------------------------------------------------------------------#
#Executions
'''
for day in range(1,(31+1)):
    #for hour in range(1,25):
    for hour in range(4,24,4):
        ULF_multi_station(2009, 5, day, (hour-4), hour, showplot=False, smoothed=True, showstations=True, window=11, offset=True, saveplot=True)
        plt.close('all')
'''
ULF_multi_station(2009, 4, 18, 0, 24, showplot=True, smoothed=True, showstations=True, window=51, offset=True, saveplot=False)
