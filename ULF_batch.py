#!/usr/bin/env python37-32
#-*- coding: utf-8 -*-

"""
Created on Wed Oct 09 11:16:29 2018
Last modified on 11/27/2018

@author (orignial): tyler_chapman

This program batch plots ULF plots (a la MIRL server).
"""
#generic
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import signal
import pylab
import datetime as datetime
import sys
import time
from pathlib import Path

#specific to this section
from UNH_analysis import ULF_txt2npy
from scipy.signal import savgol_filter
from calendar import monthrange

#------------------------------------------------------------------------------#
#Functions

__all__ = ['ULF_subday','ULF_subday_multistation','ULF_subday_multistation_grad',\
'run_range','ULF_filter','TEST_CASE','TEST_CASE_2']

def ULF_TEMP_OLD_JUST_KEEPING_FOR_CODE_REFERENCE(yyyy, mm, dd, hh_start, hh_end, showplot=False, saveplot=False):

    #DESCRIPTION: Basic sub-24 hr. plots of LYR, NAL, and HOR together (for dB/dt and spectrograph)

    #Note: Should add plotting subdivisions down to 10 min, maybe interactive plot or GUI, and animation

    #------------------------------------------------------------------------------#
    #Variables and settings

    if hh_start >= hh_end:
        print('Ending hour must be greater than starting hour.')
        sys.exit()

    np_LYR = ULF_txt2npy(yyyy, mm, dd, station='LYR')
    dat_LYR = np.load(np_file)

    np_LYR = ULF_txt2npy(yyyy, mm, dd, station='NAL')
    dat_LYR = np.load(np_file)

    np_LYR = ULF_txt2npy(yyyy, mm, dd, station='HOR')
    dat_LYR = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    time_list = np.arange(0,86400,.1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft
    f_max = 0.1 #max frequency plotting bound [Hz]
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    #------------------------------------------------------------------------------#
    #Main

    #calculations
    Bx_field = [] #create empty lists
    By_field = []
    n_pts = 864 #number of integration iterations, equal to resolution of 100 sec
    idx = np.linspace(864000/n_pts, 864000, n_pts, dtype=int) #indexing vector (can't start at zero)

    for val in np.nditer(idx): #iterates through the indexing array
        #computes the instantaneous integral via Simpsons method
        B = scipy.integrate.trapz(x_mag[(val-n_pts):val], x=None, dx=dt)
        #B = scipy.integrate.sims(x_mag[:val], x=None, dx=dt, even='avg')
        Bx_field.append(B) #appends latest integral value to list
    x_int = np.asarray(Bx_field) #turns the complete list into a numpy array

    for val in np.nditer(idx): #iterates through the indexing array
        #computes the instantaneous integral via Simpsons method
        B = scipy.integrate.trapz(y_mag[(val-n_pts):val], x=None, dx=dt)
        #B = scipy.integrate.sims(y_mag[:val], x=None, dx=dt, even='avg')
        By_field.append(B) #appends latest integral value to list
    y_int = np.asarray(By_field) #turns the complete list into a numpy array

    x_2d = np.gradient(x_mag,dt) #calulate a derivative array at dt Hz
    y_2d = np.gradient(y_mag,dt)

    nT_range = 2 #setting axis range for derivative plot

    #plotting
    fig = plt.figure() #initalize the figure
    title = station + ' ULF B-data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.1 #plot linewidth

    ax1 = fig.add_subplot(611) #adds a subplot to the figure, with 6 rows, 1 column, and first subplot at index 1 (i.e. on top)
    ax1.set_xticks(np.arange(0,864001,(144000/2)/2)) #arrange spacing of x-axis ticks the same length as data
    ax1.set_xticklabels(np.arange(0,25,1)) #creates x-axis tick labels for 24 UTC hours (redundant here)
    plt.setp(ax1.get_xticklabels(), visible=False) #sets the ticks to invisible
    ax1.set_ylabel('Bx [nT]')
    ax1.set_ylim(-np.amax(x_int),np.amax(x_int))
    ax1.plot(idx, x_int, plt_clr, linewidth=ln_wth)

    ax2 = fig.add_subplot(612, sharex=ax1) #same initialization, but sharing x-axis info
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('By [nT]')
    ax2.set_ylim(-np.amax(y_int),np.amax(y_int))
    ax2.plot(idx, y_int, plt_clr, linewidth=ln_wth)

    ax3 = fig.add_subplot(613)
    ax3.set_ylabel('dBx/dt')
    ax3.set_xticks(np.arange(0,86401,(14400/2)/2))
    ax3.set_xticklabels(np.arange(0,25,1))
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylim(-b_max,b_max)
    ax3.plot(time_list, x_mag, plt_clr, linewidth=ln_wth)

    ax4 = fig.add_subplot(614, sharex=ax3)
    plt.setp(ax4.get_xticklabels(), visible=False)
    ax4.set_ylabel('dBy/dt')
    ax4.set_ylim(-b_max,b_max)
    ax4.plot(time_list, y_mag, plt_clr, linewidth=ln_wth)

    ax5 = fig.add_subplot(615)
    ax5.set_xticks(np.arange(0,86401,(14400/2)/2))
    ax5.set_xticklabels(np.arange(0,25,1))
    plt.setp(ax5.get_xticklabels(), visible=False)
    ax5.set_ylabel('(dBx/dt)^2')
    ax5.set_ylim(-nT_range,nT_range)
    #ax5.set_ylim(-np.amax(x_2d),np.amax(x_2d))
    ax5.plot(time_list, x_2d, plt_clr, linewidth=ln_wth)

    ax6 = fig.add_subplot(616, sharex=ax5)
    plt.setp(ax6.get_xticklabels(), visible=True)
    ax6.set_ylabel('(dBy/dt)^2')
    ax6.set_xlabel('Time [UTC]')
    ax6.set_ylim(-nT_range,nT_range)
    #ax6.set_ylim(-np.amax(y_2d),np.amax(y_2d))
    ax6.plot(time_list, y_2d, plt_clr, linewidth=ln_wth)

    fig2 = plt.figure() #initalize the figure
    title2 = station + ' ULF Spectrogram on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title2)

    bx1 = fig2.add_subplot(412)
    bx1.set_ylabel('dBx/dT')
    bx1.set_xticks(np.arange(0,86401,(14400/2)/2))
    bx1.set_xticklabels(np.arange(0,25,1))
    plt.setp(bx1.get_xticklabels(), visible=False)
    bx1.set_ylim(-b_max,b_max)
    bx1.plot(time_list, x_mag)

    bx2 = fig2.add_subplot(411, sharex=bx1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag, nfft, fs, noverlap = nov, cmap=cm.jet,
                                           vmin=-50, vmax=5)
    bx2.set_ylabel('Freq')
    bx2.set_ylim(0,f_max)
    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005)
    plt.setp(bx2.get_xticklabels(), visible=False)

    bx3 = fig2.add_subplot(413, sharex=bx1)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag, nfft, fs, noverlap = nov, cmap=cm.jet,
                                           vmin=-50, vmax=5)
    bx3.set_ylabel('Freq')
    bx3.set_ylim(0,f_max)
    plt.setp(bx3.get_xticklabels(), visible=False)

    bx4 = fig2.add_subplot(414, sharex=bx1)
    bx4.set_ylabel('dBy/dT')
    bx4.set_xlabel('Time [UTC]')
    bx4.set_ylim(-b_max,b_max)
    bx4.plot(time_list, y_mag)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/'

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_B_field_plots.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_B_field_plots.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_B_field_plots.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_B_field_plots.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig.savefig(png_file, bbox_inches='tight')

def ULF_subday(yyyy, mm, dd, UTC_start, UTC_end, station='LYR', f_max=0.1, showplot=False, saveplot=False, smoothed=False, save_dir='none'):

    #DESCRIPTION: Plotting basic ULF sub-24 hr. plots, as done by MIRL server, saving all to a separate directory

    #Note: Should add plotting subdivisions down to 10 min, maybe interactive plot or GUI, and animation

    #------------------------------------------------------------------------------#
    #Variables and settings

    if UTC_start >= UTC_end:
        print('ERROR: Ending hour must be greater than starting hour.')
        sys.exit()

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    idx_start = UTC_start*36000
    idx_end = UTC_end*36000

    d1 = dat_arr[idx_start:idx_end,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[idx_start:idx_end,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[idx_start:idx_end,2] #y-axis signal
    d4 = dat_arr[idx_start:idx_end,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    hours = UTC_end - UTC_start
    x_ticks = np.arange(0,(len(d1)),(len(d1)/10/hours))
    x_tick_labels = np.arange((UTC_start),(UTC_end+1),1)

    time_list = np.arange(0,(3600*hours),.1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**11 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/1.2 #overlap in fft segments, of length nfft
    #f_max = 0.1 #max frequency plotting bound [Hz] (CURRENTLY SET AS AN ARG)
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    my_dpi = 120

    #----'--------------------------------------------------------------------------#
    #Main

    '''
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag, nfft, fs, noverlap = nov)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag, nfft, fs, noverlap = nov)

    p_min = np.amin(np.minimum(pxx1, pxx2))
    p_max = np.amax(np.maximum(pxx1, pxx2))
    '''

    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = str(mm) + '/' + str(dd) + '/' + str(yyyy) + ' at ' + station + ' from ' + str(UTC_start) + ':00-' + str(UTC_end) + ':00 UTC'
    plt.suptitle(title)

    bx1 = fig.add_subplot(411)
    bx1.set_xticks(x_ticks)
    bx1.set_xticklabels(x_tick_labels)
    plt.setp(bx1.get_xticklabels(), visible=False)
    bx1.set_ylabel('dBx/dT [nT/sec]')
    bx1.set_ylim(-b_max,b_max)

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    bx2 = fig.add_subplot(412, sharex=bx1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag, nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(bx2.get_xticklabels(), visible=False)
    bx2.set_ylabel('Freq [Hz]')
    bx2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    bx3 = fig.add_subplot(413, sharex=bx1)
    plt.setp(bx3.get_xticklabels(), visible=False)
    bx3.set_ylabel('dBy/dT [nT/sec]')
    bx3.set_ylim(-b_max,b_max)

    bx4 = fig.add_subplot(414, sharex=bx1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag, nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(bx4.get_xticklabels(), visible=True)
    bx4.set_ylabel('Freq [Hz]')
    bx4.set_xlabel('Time [UTC]')
    bx4.set_ylim(0,f_max)

    #cb2 = plt.colorbar(cax2, orientation='vertical',pad=.005) #shows spectrogram colorbar

    '''
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    cb = plt.colorbar(cax1, cax = cbaxes)
    '''

    if smoothed == True:
        x_mag_smoothed = savgol_filter(x_mag, 11, 0)
        y_mag_smoothed = savgol_filter(y_mag, 11, 0)
        bx1.plot(time_list, x_mag_smoothed, color=plotcolor, linewidth=ln_wth)
        bx3.plot(time_list, y_mag_smoothed, color=plotcolor, linewidth=ln_wth)
    else:
        bx1.plot(time_list, x_mag, color=plotcolor, linewidth=ln_wth)
        bx3.plot(time_list, y_mag, color=plotcolor, linewidth=ln_wth)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    #png_dir = 'D:/' + station + '_All_Files/' + 'DayPlots/'
    if save_dir == 'none':
        png_dir = 'C:/Users/Tyler/Desktop/save_img/'
    else:
        png_dir = save_dir

    UTC_range = '_from_' + str(UTC_start) + '_to_' + str(UTC_end)

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + UTC_range + '_dBdt.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + UTC_range + '_dBdt.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + UTC_range + '_dBdt.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + UTC_range + '_dBdt.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig.savefig(png_file, bbox_inches='tight')

def ULF_subday_multistation(yyyy, mm, dd, UTC_start, UTC_end, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: Basic sub-24 hr. plots of LYR, NAL, and HOR together (for dB/dt and spectrograph)

    #Note: Should add plotting subdivisions down to 10 min, maybe interactive plot or GUI, and animation

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
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    #----'--------------------------------------------------------------------------#
    #Main

    '''
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag, nfft, fs, noverlap = nov)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag, nfft, fs, noverlap = nov)

    p_min = np.amin(np.minimum(pxx1, pxx2))
    p_max = np.amax(np.maximum(pxx1, pxx2))
    '''

    my_dpi = 120
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = str(mm) + '/' + str(dd) + '/' + str(yyyy) + ' at NAL/LYR/HOR from ' + str(UTC_start) + ':00-' + str(UTC_end) + ':00 UTC'
    plt.suptitle(title)

    #First Column of plot
    ax1 = fig.add_subplot(4,3,1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('dBx/dT [nT/sec]')
    ax1.set_title('NAL')
    ax1.set_ylim(-b_max,b_max)

    ax2 = fig.add_subplot(4,3,4, sharex=ax1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag[:,0], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('Freq [Hz]')
    ax2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    ax3 = fig.add_subplot(4,3,7, sharex=ax1)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('dBy/dT [nT/sec]')
    ax3.set_ylim(-b_max,b_max)

    ax4 = fig.add_subplot(4,3,10, sharex=ax1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag[:,0], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel('Freq [Hz]')
    ax4.set_ylim(0,f_max)

    #Second Column of plot
    bx1 = fig.add_subplot(4,3,2)
    bx1.set_xticks(x_ticks)
    bx1.set_xticklabels(x_tick_labels)
    plt.setp(bx1.get_xticklabels(), visible=False)
    plt.setp(bx1.get_yticklabels(), visible=False)
    bx1.set_title('LYR')
    bx1.set_ylim(-b_max,b_max)

    bx2 = fig.add_subplot(4,3,5, sharex=bx1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag[:,1], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(bx2.get_xticklabels(), visible=False)
    plt.setp(bx2.get_yticklabels(), visible=False)
    bx2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    bx3 = fig.add_subplot(4,3,8, sharex=bx1)
    plt.setp(bx3.get_xticklabels(), visible=False)
    plt.setp(bx3.get_yticklabels(), visible=False)
    bx3.set_ylim(-b_max,b_max)

    bx4 = fig.add_subplot(4,3,11, sharex=bx1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag[:,1], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(bx4.get_xticklabels(), visible=True)
    plt.setp(bx4.get_yticklabels(), visible=False)
    bx4.set_xlabel('Time [UTC]')
    bx4.set_ylim(0,f_max)

    #Third Column of plot
    cx1 = fig.add_subplot(4,3,3)
    cx1.set_xticks(x_ticks)
    cx1.set_xticklabels(x_tick_labels)
    plt.setp(cx1.get_xticklabels(), visible=False)
    plt.setp(cx1.get_yticklabels(), visible=False)
    cx1.set_title('HOR')
    cx1.set_ylim(-b_max,b_max)

    cx2 = fig.add_subplot(4,3,6, sharex=cx1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag[:,2], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(cx2.get_xticklabels(), visible=False)
    plt.setp(cx2.get_yticklabels(), visible=False)
    cx2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    cx3 = fig.add_subplot(4,3,9, sharex=cx1)
    plt.setp(cx3.get_xticklabels(), visible=False)
    plt.setp(cx3.get_yticklabels(), visible=False)
    cx3.set_ylim(-b_max,b_max)

    cx4 = fig.add_subplot(4,3,12, sharex=cx1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag[:,2], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(cx4.get_xticklabels(), visible=True)
    plt.setp(cx4.get_yticklabels(), visible=False)
    cx4.set_ylim(0,f_max)

    #cb2 = plt.colorbar(cax2, orientation='vertical',pad=.005) #shows spectrogram colorbar

    '''
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    cb = plt.colorbar(cax1, cax = cbaxes)
    '''

    if smoothed == True:
        x_mag_smoothed = savgol_filter(x_mag, 11, 0, axis=0)
        y_mag_smoothed = savgol_filter(y_mag, 11, 0, axis=0)
        ax1.plot(time_list, x_mag_smoothed[:,0], color=plotcolor, linewidth=ln_wth)
        ax3.plot(time_list, y_mag_smoothed[:,0], color=plotcolor, linewidth=ln_wth)
        bx1.plot(time_list, x_mag_smoothed[:,1], color=plotcolor, linewidth=ln_wth)
        bx3.plot(time_list, y_mag_smoothed[:,1], color=plotcolor, linewidth=ln_wth)
        cx1.plot(time_list, x_mag_smoothed[:,2], color=plotcolor, linewidth=ln_wth)
        cx3.plot(time_list, y_mag_smoothed[:,2], color=plotcolor, linewidth=ln_wth)
    else:
        ax1.plot(time_list, x_mag[:,0], color=plotcolor, linewidth=ln_wth)
        ax3.plot(time_list, y_mag[:,0], color=plotcolor, linewidth=ln_wth)
        bx1.plot(time_list, x_mag[:,1], color=plotcolor, linewidth=ln_wth)
        bx3.plot(time_list, y_mag[:,1], color=plotcolor, linewidth=ln_wth)
        cx1.plot(time_list, x_mag[:,2], color=plotcolor, linewidth=ln_wth)
        cx3.plot(time_list, y_mag[:,2], color=plotcolor, linewidth=ln_wth)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/save_img/'
    UTC_range = 'from_' + str(UTC_start) + '_to_' + str(UTC_end)

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + UTC_range + '_dBdt_multi.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + UTC_range + '_dBdt_multi.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + UTC_range + '_dBdt_multi.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + UTC_range + '_dBdt_multi.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig.savefig(png_file, bbox_inches='tight')

def ULF_subday_multistation_grad(yyyy, mm, dd, UTC_start, UTC_end, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: Basic sub-24 hr. plots of LYR, NAL, and HOR together (for 2dB/2dt and spectrograph)

    #Note: Should add plotting subdivisions down to 10 min, maybe interactive plot or GUI, and animation

    #NOTE: the smoothing may still be a bit wonky

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

    x_NAL_grad = np.gradient(x_NAL, d1)
    x_LYR_grad = np.gradient(x_LYR, d1)
    x_HOR_grad = np.gradient(x_HOR, d1)
    y_NAL_grad = np.gradient(y_NAL, d1)
    y_LYR_grad = np.gradient(y_LYR, d1)
    y_HOR_grad = np.gradient(y_HOR, d1)

    x_mag = np.column_stack((x_NAL_grad, x_LYR_grad, x_HOR_grad))
    y_mag = np.column_stack((y_NAL_grad, y_LYR_grad, y_HOR_grad))

    hours = UTC_end - UTC_start
    x_ticks = np.arange(0,(len(d1)),(len(d1)/10/hours))
    x_tick_labels = np.arange((UTC_start),(UTC_end+1),1)

    time_list = np.arange(0,(3600*hours),.1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**11 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/1.2 #overlap in fft segments, of length nfft
    f_max = 0.1 #max frequency plotting bound [Hz]
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    #----'--------------------------------------------------------------------------#
    #Main

    '''
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag, nfft, fs, noverlap = nov)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag, nfft, fs, noverlap = nov)

    p_min = np.amin(np.minimum(pxx1, pxx2))
    p_max = np.amax(np.maximum(pxx1, pxx2))
    '''

    v_min = -50
    v_max = 10

    my_dpi = 120
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = str(mm) + '/' + str(dd) + '/' + str(yyyy) + ' at NAL/LYR/HOR from ' + str(UTC_start) + ':00-' + str(UTC_end) + ':00 UTC'
    plt.suptitle(title)

    #First Column of plot
    ax1 = fig.add_subplot(4,3,1)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_tick_labels)
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('2dBx/2dT [nT/sec]')
    ax1.set_ylim(-b_max,b_max)

    ax2 = fig.add_subplot(4,3,4, sharex=ax1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag[:,0], nfft, fs, noverlap = nov, vmin=v_min, vmax=v_max, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('Freq [Hz]')
    ax2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    ax3 = fig.add_subplot(4,3,7, sharex=ax1)
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('2dBy/2dT [nT/sec]')
    ax3.set_ylim(-b_max,b_max)

    ax4 = fig.add_subplot(4,3,10, sharex=ax1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag[:,0], nfft, fs, noverlap = nov, vmin=v_min, vmax=v_max, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(ax4.get_xticklabels(), visible=True)
    ax4.set_ylabel('Freq [Hz]')
    ax4.set_ylim(0,f_max)

    #Second Column of plot
    bx1 = fig.add_subplot(4,3,2)
    bx1.set_xticks(x_ticks)
    bx1.set_xticklabels(x_tick_labels)
    plt.setp(bx1.get_xticklabels(), visible=False)
    plt.setp(bx1.get_yticklabels(), visible=False)
    bx1.set_ylim(-b_max,b_max)

    bx2 = fig.add_subplot(4,3,5, sharex=bx1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag[:,1], nfft, fs, noverlap = nov, vmin=v_min, vmax=v_max, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(bx2.get_xticklabels(), visible=False)
    plt.setp(bx2.get_yticklabels(), visible=False)
    bx2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    bx3 = fig.add_subplot(4,3,8, sharex=bx1)
    plt.setp(bx3.get_xticklabels(), visible=False)
    plt.setp(bx3.get_yticklabels(), visible=False)
    bx3.set_ylim(-b_max,b_max)

    bx4 = fig.add_subplot(4,3,11, sharex=bx1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag[:,1], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(bx4.get_xticklabels(), visible=True)
    plt.setp(bx4.get_yticklabels(), visible=False)
    bx4.set_xlabel('Time [UTC]')
    bx4.set_ylim(0,f_max)

    #First Column of plot
    cx1 = fig.add_subplot(4,3,3)
    cx1.set_xticks(x_ticks)
    cx1.set_xticklabels(x_tick_labels)
    plt.setp(cx1.get_xticklabels(), visible=False)
    plt.setp(cx1.get_yticklabels(), visible=False)
    cx1.set_ylim(-b_max,b_max)

    cx2 = fig.add_subplot(4,3,6, sharex=cx1)
    pxx1, freq1, t1, cax1 = pylab.specgram(x_mag[:,2], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    #plt.pcolormesh(t1, freq1, pxx1, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    plt.setp(cx2.get_xticklabels(), visible=False)
    plt.setp(cx2.get_yticklabels(), visible=False)
    cx2.set_ylim(0,f_max)

    #cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005) #shows spectrogram colorbar

    cx3 = fig.add_subplot(4,3,9, sharex=cx1)
    plt.setp(cx3.get_xticklabels(), visible=False)
    plt.setp(cx3.get_yticklabels(), visible=False)
    cx3.set_ylim(-b_max,b_max)

    cx4 = fig.add_subplot(4,3,12, sharex=cx1)
    #plt.pcolormesh(t2, freq2, pxx2, norm=colors.LogNorm(vmin=p_min, vmax=p_max), cmap=cm.jet)
    pxx2, freq2, t2, cax2 = pylab.specgram(y_mag[:,2], nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.setp(cx4.get_xticklabels(), visible=True)
    plt.setp(cx4.get_yticklabels(), visible=False)
    cx4.set_ylim(0,f_max)

    #cb2 = plt.colorbar(cax2, orientation='vertical',pad=.005) #shows spectrogram colorbar

    '''
    cbaxes = fig.add_axes([0.8, 0.1, 0.03, 0.8])
    cb = plt.colorbar(cax1, cax = cbaxes)
    '''

    if smoothed == True:
        x_mag_smoothed = savgol_filter(x_mag, 11, 0, axis=0)
        y_mag_smoothed = savgol_filter(y_mag, 11, 0, axis=0)
        ax1.plot(time_list, x_mag_smoothed[:,0], color=plotcolor, linewidth=ln_wth)
        ax3.plot(time_list, y_mag_smoothed[:,0], color=plotcolor, linewidth=ln_wth)
        bx1.plot(time_list, x_mag_smoothed[:,1], color=plotcolor, linewidth=ln_wth)
        bx3.plot(time_list, y_mag_smoothed[:,1], color=plotcolor, linewidth=ln_wth)
        cx1.plot(time_list, x_mag_smoothed[:,2], color=plotcolor, linewidth=ln_wth)
        cx3.plot(time_list, y_mag_smoothed[:,2], color=plotcolor, linewidth=ln_wth)
    else:
        ax1.plot(time_list, x_mag[:,0], color=plotcolor, linewidth=ln_wth)
        ax3.plot(time_list, y_mag[:,0], color=plotcolor, linewidth=ln_wth)
        bx1.plot(time_list, x_mag[:,1], color=plotcolor, linewidth=ln_wth)
        bx3.plot(time_list, y_mag[:,1], color=plotcolor, linewidth=ln_wth)
        cx1.plot(time_list, x_mag[:,2], color=plotcolor, linewidth=ln_wth)
        cx3.plot(time_list, y_mag[:,2], color=plotcolor, linewidth=ln_wth)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/DayPlotsGrad/'
    UTC_range = 'from_' + str(UTC_start) + '_to_' + str(UTC_end)

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + UTC_range + '_2dBdt_multi.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + UTC_range + '_2dBdt_multi.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + UTC_range + '_2dBdt_multi.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + UTC_range + '_2dBdt_multi.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig.savefig(png_file, bbox_inches='tight')

def run_range(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, station='LYR', third=False):

    #DESCRIPTION: Provides a stucture to loop over another function (from this script) set within

    #about 1.5 hrs. to run a full year at 1hr res, about 7.6 hrs. to run a decade at 8 hrs. res

    if third == False:
        for year in range(yyyy_start,(yyyy_end+1)):
            if year == yyyy_start and year == yyyy_end:
                for month in range(mm_start,(mm_end+1)):
                    if month == mm_end:
                        for day in range(dd_start,(dd_end+1)):
                            for hour in range(1,25):
                                ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                                plt.close('all')
                    else:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(1,(day_range+1)):
                            for hour in range(1,25):
                                ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                                plt.close('all')
            if year == yyyy_start:
                for month in range(mm_start,(12+1)):
                    if month == mm_start:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(dd_start,(day_range+1)):
                            for hour in range(1,25):
                                ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                                plt.close('all')
                    else:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(1,(day_range+1)):
                            for hour in range(1,25):
                                ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                                plt.close('all')
            if year == yyyy_end:
                for month in range(1,(mm_end+1)):
                    if month == mm_end:
                        for day in range(1,(dd_end+1)):
                            for hour in range(1,25):
                                ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                                plt.close('all')
                    else:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(1,(day_range+1)):
                            for hour in range(1,25):
                                ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                                plt.close('all')
            else:
                for month in range(1,(12+1)):
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        for hour in range(1,25):
                            ULF_subday(year, month, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True, station=station)
                            plt.close('all')
    else:
        for year in range(yyyy_start,(yyyy_end+1)):
            if year == yyyy_start and year == yyyy_end:
                for month in range(mm_start,(mm_end+1)):
                    if month == mm_end:
                        for day in range(dd_start,(dd_end+1)):
                            ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                            plt.close('all')
                    else:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(1,(day_range+1)):
                            ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                            plt.close('all')
            if year == yyyy_start:
                for month in range(mm_start,(12+1)):
                    if month == mm_start:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(dd_start,(day_range+1)):
                            ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                            plt.close('all')
                    else:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(1,(day_range+1)):
                            ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                            plt.close('all')
            if year == yyyy_end:
                for month in range(1,(mm_end+1)):
                    if month == mm_end:
                        for day in range(1,(dd_end+1)):
                            ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                    else:
                        start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                        for day in range(1,(day_range+1)):
                            ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                            ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                            plt.close('all')
            else:
                for month in range(1,(12+1)):
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        ULF_subday(year, month, day, 0, 8, showplot=False, smoothed=True, saveplot=True, station=station)
                        ULF_subday(year, month, day, 8, 16, showplot=False, smoothed=True, saveplot=True, station=station)
                        ULF_subday(year, month, day, 16, 24, showplot=False, smoothed=True, saveplot=True, station=station)
                        plt.close('all')


def ULF_filter(dat_arr, f_low, f_high, sample_rate_Hz=10):

    #Note: will only take a 1-D input vector for dat_arr

    ### Initial Parameters
    fL = f_low/sample_rate_Hz
    fH = f_high/sample_rate_Hz
    b = 0.08
    N = int(np.ceil((4 / b)))
    if not N % 2: N += 1  # Make sure that N is odd.
    n = np.arange(N)

    ### Low-pass filter
    hlpf = np.sinc(2 * fH * (n - (N - 1) / 2.))
    hlpf *= np.blackman(N)
    hlpf = hlpf / np.sum(hlpf)

    ### High-pass filter
    hhpf = np.sinc(2 * fL * (n - (N - 1) / 2.))
    hhpf *= np.blackman(N)
    hhpf = hhpf / np.sum(hhpf)
    hhpf = -hhpf
    hhpf[int((N - 1) / 2)] += 1

    ### Combine filters and original singal to form new signal
    h = np.convolve(hlpf, hhpf)
    new_signal = np.convolve(dat_arr, h)

    return new_signal

def TEST_CASE(yyyy, mm, dd, station='LYR'):

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**11 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/1.2 #overlap in fft segments, of length nfft

    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    x_dat = dat_arr[:,1]
    y_dat = dat_arr[:,2]

    low_f = 0.01
    high_f = 0.15
    x_new = ULF_filter(x_dat, low_f, high_f)
    y_new = ULF_filter(y_dat, low_f, high_f)

    f_max = 0.2
    v_min = -50
    v_max = 5

    plt.figure(1)
    plt.subplot(411)
    plt.ylabel('Magnitude [nT]')
    plt.xlabel('Samples')
    plt.title('Unfiltered X-axis')
    plt.plot(x_dat, color=plotcolor, linewidth=ln_wth)
    plt.subplot(412)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Samples')
    plt.title('Unfiltered X-axis')
    pylab.specgram(x_dat, nfft, fs, noverlap = nov, vmin=v_min, vmax=v_max, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.ylim(0,f_max)
    plt.subplot(413)
    plt.ylabel('Magnitude [nT]')
    plt.xlabel('Samples')
    plt.title('Filtered X-axis')
    plt.plot(x_new, color=plotcolor, linewidth=ln_wth)
    plt.subplot(414)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Samples')
    plt.title('Filtered X-axis')
    pylab.specgram(x_new, nfft, fs, noverlap = nov, vmin=-50, vmax=-30, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.ylim(0,f_max)

    plt.figure(2)
    plt.subplot(411)
    plt.ylabel('Magnitude [nT]')
    plt.xlabel('Samples')
    plt.title('Unfiltered Y-axis')
    plt.plot(y_dat, color=plotcolor, linewidth=ln_wth)
    plt.subplot(412)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Samples')
    plt.title('Unfiltered Y-axis')
    pylab.specgram(y_dat, nfft, fs, noverlap = nov, vmin=v_min, vmax=v_max, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.ylim(0,f_max)
    plt.subplot(413)
    plt.ylabel('Magnitude [nT]')
    plt.xlabel('Samples')
    plt.title('Filtered Y-axis')
    plt.plot(y_new, color=plotcolor, linewidth=ln_wth)
    plt.subplot(414)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Samples')
    plt.title('Filtered Y-axis')
    pylab.specgram(y_new, nfft, fs, noverlap = nov, vmin=-60, vmax=-45, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.ylim(0,f_max)
    plt.show()

def TEST_CASE_2(yyyy, mm, dd, station='LYR'):

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**11 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/1.2 #overlap in fft segments, of length nfft

    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    x_dat = dat_arr[:,1]
    y_dat = dat_arr[:,2]

    dat_cvl = np.correlate(x_dat, y_dat)

    f_max = 0.2
    v_min = -50
    v_max = 5

    plt.figure(1)
    plt.subplot(411)
    plt.ylabel('Magnitude [nT]')
    plt.xlabel('Samples')
    plt.title('Unfiltered X-axis')
    plt.plot(x_dat, color=plotcolor, linewidth=ln_wth)
    plt.subplot(412)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Samples')
    plt.title('Unfiltered X-axis')
    pylab.specgram(x_dat, nfft, fs, noverlap = nov, vmin=v_min, vmax=v_max, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.ylim(0,f_max)
    plt.subplot(413)
    plt.ylabel('Magnitude [nT]')
    plt.xlabel('Samples')
    plt.title('Convolved X/Y')
    plt.plot(dat_cvl, color=plotcolor, linewidth=ln_wth)
    plt.subplot(414)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Samples')
    plt.title('Convolved X/Y')
    pylab.specgram(dat_cvl, nfft, fs, noverlap = nov, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    plt.ylim(0,f_max)
    plt.show()
