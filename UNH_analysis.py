#!/usr/bin/env python37-32
#-*- coding: utf-8 -*-

"""
Created on Wed Oct 07 23:54:42 2018
Last modified on 11/16/2018

@author (orignial): tyler_chapman

This program is a development area for various ULF data analysis tools.
"""
#Generic
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import pylab
import datetime as datetime
import sys
import time

#Specific to this script
from pathlib import Path
import pandas as pd
import nolds
#import spacepy as sp #currently not installed (doesn't support pip install???)
#from spacepy import seapy
#import astropy  #not currently used...has useful physical constants
#import heliopy  #not currently used
#import geomagpy #not currently used
from scipy.signal import savgol_filter
from itertools import cycle
import calendar
from calendar import monthrange
from datetime import date
from pprint import pprint as pprint
from matplotlib.ticker import AutoMinorLocator
import os
import math

#------------------------------------------------------------------------------#
#Functions

__all__ = ['ULF_mud2txt','ULF_txt2npy','ULF_txt2npy_sampled','ULF_multi_npy',\
'ULF_B_field_plots','ULF_multi_plot','periodogram_multi_freq','periodogram_multi_freq_dHzdt',\
'total_power','total_power_plot','Pc_power_bands','ULF_day_spec','ULF_day_spec_grad',\
'total_power_multi','total_power_multi_grad','ACES_txt2npy','ACES_plots','OMNI_plots',\
'OMNI_min_asc2npy','OMNI_1m_select','OMNI_ULF_plots','OMNI_all_plots','OMNI_save_dir',\
'SuperMag_txt2npy','SuperMag_plots','SuperMag_spec','SuperMag_grad','ULF_combo_plots',\
'ULF_combo_grad','Run_Wrapper']

#File Generation
def ULF_mud2txt(yyyy, mm, dd, station='LYR'):

    #DESCRIPTION: This function opens a ULF MUD file and saves as a test file
    #Returns the full path file name of the specified numpy file

    data_dir = 'C:/Users/Tyler/Desktop/Project Lisbon/Datasets/ULF/' + station + '/MUD'

    #modifies the string to the correct format given the date
    if mm >= 10:
        mud_location = data_dir + station + '_' + str(yyyy) + '_' + str(mm) +  '.MUD'
    else:
        mud_location = data_dir + station + '_' + str(yyyy) + '_0' + str(mm)+ '.MUD'


    #check to make sure the specified text file exits and stops program otherwise
    txt_path = Path(mud_location)
    if txt_path.is_file() is False:
        print('Note: Specified ULF MUD file does not exist')
        print(mud_location)
        sys.exit()

    temp = txt_location.replace('.MUD', '_dbdt_v1.txt')
    np_file = temp.replace('MUD', 'TXT')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for that date')
    else:
        datapath = open(mud_location, 'r')
        dat = datapath.readlines()
        datapath.close()
        dat = dat[3:] #removes the header

        dataset = []
        for index, item in enumerate(dat,start=0):
            line = dat[index].split()
            line = [float(i) for i in line]
            dataset.append(line)

        ulf_dat = np.asarray(dataset, dtype='float32') #smaller than float64 type to save memory space

        np.save(np_file,ulf_dat)

    return np_file

    '''
    #IDL MUD read code, from Chutter

    function readmud, FilePath, FromSecond, ToSecond, sample_rate


    ;FilePath='c:\LYR_2007_01.MUD'
    ;FromSecond = long(long(3600) * 24)
    ;ToSecond = long(long(3600) * 24 * 2)

    ; open the MUD file:
    dat_fh = 1
    openr, dat_fh, filepath, /get_lun

    ; read header
    FileFlag = bytarr(4)
    NumChannels = 0
    SampleRate = 0
    SampleSize = 0
    StartYear = 0
    StartMonth = 0
    StartDay = 0
    StationName = bytarr(4)
    NumSeconds = long(0)
    Reserved1 = long(0)
    Reserved2 = long(0)

    readu, dat_fh, FileFlag
    readu, dat_fh, NumChannels
    readu, dat_fh, SampleRate
    readu, dat_fh, SampleSize
    readu, dat_fh, StartYear
    readu, dat_fh, StartMonth
    readu, dat_fh, StartDay
    readu, dat_fh, StationName
    readu, dat_fh, NumSeconds
    readu, dat_fh, Reserved1
    readu, dat_fh, Reserved2

    ; set sample buffer size
    SampleBufferSize = SampleRate * SampleSize / 8

    ; allocate data array
    Data_Array  = fltarr(2, (long(ToSecond)-long(FromSecond)) * sample_rate)

    ; allocate temp buffer
    Buffer_Array = intarr(10, 2)

    ; move to the 1st data position
    Offset = long(32)
    Offset = Offset + FromSecond * (4 + SampleBufferSize * NumChannels)
    POINT_LUN, dat_fh, Offset

    ; read samples
    Second = long(0)
    FOR n = long(0), long(ToSecond)-long(FromSecond)-1 DO BEGIN
        readu, dat_fh, Second

        ; read 10-sample chunk of 2 channels
        readu, dat_fh, Buffer_Array

        ; assign the samples to the data array
        FOR m = 0, 9 DO BEGIN
            Data_Array(0, n * 10 + m) = (Buffer_Array(m, 0) + 2048.0) / 4096.0 * 20.0 - 10.0;
            Data_Array(1, n * 10 + m) = (Buffer_Array(m, 1) + 2048.0) / 4096.0 * 20.0 - 10.0;
        ENDFOR

    ENDFOR

    ; close the file
    free_lun, dat_fh

    return, Data_Array
    end
    '''

def ULF_txt2npy(yyyy, mm, dd, station='LYR'):

    #DESCRIPTION: This function opens a ULF txt file and saves as a numpy file, from D: storage
    #Returns the full path file name of the specified numpy file

    data_dir = 'D:/' + station + '_All_Files/' + station +'_TXT/'

    #modifies the string to the correct format given the date
    if mm >= 10 and dd < 10:
        txt_location = data_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_dbdt_v1.txt'
    elif mm < 10 and dd >= 10:
        txt_location = data_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_dbdt_v1.txt'
    elif mm < 10 and dd < 10:
        txt_location = data_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_dbdt_v1.txt'
    else:
        txt_location = data_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_dbdt_v1.txt'

    #check to make sure the specified text file exits and stops program otherwise
    txt_path = Path(txt_location)
    if txt_path.is_file() is False:
        print('Note: Specified ULF data file does not exist')
        print(txt_location)
        sys.exit()

    temp = txt_location.replace('_dbdt_v1.txt', '.npy')
    np_file = temp.replace('TXT', 'NPY')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for that date')
    else:
        datapath = open(txt_location, 'r')
        dat = datapath.readlines()
        datapath.close()
        dat = dat[3:] #removes the header

        dataset = []
        for index, item in enumerate(dat,start=0):
            line = dat[index].split()
            line = [float(i) for i in line]
            dataset.append(line)

        ulf_dat = np.asarray(dataset, dtype='float32') #smaller than float64 type to save memory space

        np.save(np_file,ulf_dat)

    return np_file

def ULF_txt2npy_sampled(yyyy, mm, dd, station='LYR', d_type='float32', sample_sec=60):

    #DESCRIPTION: This function opens a ULF txt file and saves as a numpy file, from D: storage
    #Returns the full path file name of the specified numpy file (sampled per minute)

    data_dir = 'D:/' + station + '_All_Files/' + station +'_TXT/'

    #modifies the string to the correct format given the date
    if mm >= 10 and dd < 10:
        txt_location = data_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_dbdt_v1.txt'
    elif mm < 10 and dd >= 10:
        txt_location = data_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_dbdt_v1.txt'
    elif mm < 10 and dd < 10:
        txt_location = data_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_dbdt_v1.txt'
    else:
        txt_location = data_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_dbdt_v1.txt'

    #check to make sure the specified text file exits and stops program otherwise
    txt_path = Path(txt_location)
    if txt_path.is_file() is False:
        print('Note: Specified ULF data file does not exist')
        print(txt_location)
        sys.exit()

    temp = txt_location.replace('_dbdt_v1.txt', '_sampled.npy')
    np_file = temp.replace('TXT', 'NPY')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for that date...REPLACING!')
        os.remove(np_file)

    datapath = open(txt_location, 'r')
    dat = datapath.readlines()
    datapath.close()
    dat = dat[3:] #removes the header

    dataset = []
    for index, item in enumerate(dat,start=0):
        line = dat[index].split()
        line = [float(i) for i in line]
        dataset.append(line)

    ulf_dat = np.asarray(dataset, dtype=d_type) #smalled than float64 type to save memory space

    sample_rate = sample_sec*10
    temp = np.transpose(ulf_dat.reshape((int(len(dataset)/sample_rate),sample_rate,4)), axes=(1, 0, 2)) #reshapes array to ready for mean
    ulf_dat_avg = np.mean(temp, axis=0) #takes mean in 3rd dimension, reducing to 2d array with mean per minute
    ulf_dat_avg[:,0] = np.arange(int(len(dataset)/sample_rate)) #replaces time index column with minute of day values

    np.save(np_file,ulf_dat_avg)

    return np_file

def ULF_multi_npy(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, station='LYR', printfile=False, sampled=True, sample_sec=60):

    #DESCRIPTION: This function creates a multi-day numpy save, returns full file path

    data_dir = 'D:/LYR_All_Files/'
    save_name = station+ '_from_' + str(yyyy_start) + str(mm_start) + str(dd_start) + '_to_' + str(yyyy_end) + str(mm_end) + str(dd_end)
    if sampled == True:
        file_path = data_dir + save_name + '_sampled.npy'
    else:
        file_path = data_dir + save_name + '.npy'
    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(file_path)
    skipped = False
    if check_path.is_file():
        print('Note: Already have the numpy file for that date range')
        skipped = True
    else:
        ulf_dat = np.array([], dtype='float32') #creates an empty numpy array
        for year in range(yyyy_start,(yyyy_end+1)):
            for month in range(mm_start,(mm_end+1)):
                if month == mm_end:
                    for day in range(dd_start,(dd_end+1)):
                        if sampled == True:
                            np_file = ULF_txt2npy_sampled(year, month, day, station=station)
                        else:
                            np_file = ULF_txt2npy(year, month, day, station=station, sample_sec=sample_sec)
                        dat_arr = np.load(np_file)
                        ulf_dat = np.vstack([ulf_dat, dat_arr]) if ulf_dat.size else dat_arr #sets empty array equal to first entry
                else:
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        if sampled == True:
                            np_file = ULF_txt2npy_sampled(year, month, day, station=station, sample_sec=sample_sec)
                        else:
                            np_file = ULF_txt2npy(year, month, day, station=station)
                        dat_arr = np.load(np_file)
                        ulf_dat = np.vstack([ulf_dat, dat_arr]) if ulf_dat.size else dat_arr

        print('about to NUMPY!')
        #ulf_dat = np.array(ulf_dat, dtype='float16') #turn the list into a numpy array of type float16, to save memory space

        print('about to SAVE!')
        np.save(file_path,ulf_dat)

    if printfile == True and skipped == False:
        #save arrays to a text file for viewing
        header_text = 'ulf_data shape is: ' + str(ulf_dat.shape)
        save_file = save_name + '.txt'
        np.savetxt(save_file, ulf_dat, header=header_text)
    elif printfile == True and skipped == True:
        #save arrays to a text file for viewing
        ulf_dat = np.load(file_path)
        header_text = 'ulf_data shape is: ' + str(ulf_dat.shape)
        save_file = save_name + '.txt'
        np.savetxt(save_file, ulf_dat, header=header_text)

    return file_path


#B-field plots
def ULF_B_field_plots(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: Plotting B, dB/dt, and (dB/dt)^2 together

    #------------------------------------------------------------------------------#
    #Variables and settings

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    hour = np.arange(0,25,1)
    time_list = np.arange(0,86400,.1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft
    f_max = 0.1 #max frequency plotting bound [Hz]
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    #------------------------------------------------------------------------------#
    #Main

    #Power Plots (periodograms)

    Hz_1 = 0.02 #frequencies to search for and plot power

    temp_x = []
    temp_y = []
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)

    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        pwr_1 = pxx[(np.abs(fx-Hz_1)).argmin()]
        temp_x.append(pwr_1)
    x_pwr = np.asarray(temp_x)

    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        pwr_1 = pxy[(np.abs(fy-Hz_1)).argmin()]
        temp_y.append(pwr_1)
    y_pwr = np.asarray(temp_y)

    fxt, pxxt = signal.periodogram(x_mag,10e2) #plots of whole 24 hr period
    fyt, pxyt = signal.periodogram(y_mag,10e2)
    fxtw, pxxtw = signal.welch(x_mag,10e2) #plots of whole 24 hr period
    fytw, pxytw = signal.welch(y_mag,10e2)

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

    g_mag = np.sqrt(np.add(np.square(x_mag), np.square(y_mag))) #element-wise sqrt(x^2+y^2)
    g_2d = savgol_filter(np.gradient(g_mag, dt), 31, 0)
    #d_mag = savgol_filter(np.absolute(np.subtract(x_mag, y_mag)), 31, 0)
    d_mag = np.absolute(np.subtract(x_mag, y_mag))

    #plotting
    #plt.close('all') #closed all open figures (currently not functioning)

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

    fig1 = plt.figure()
    title1 = str(Hz_1) + ' Hz Power in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title1)

    pgx = fig1.add_subplot(211)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(np.arange(0,25,1))
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Power in dBx/dt')
    pgx.set_ylim(0,np.amax(x_pwr))
    pgx.plot(idx, x_pwr)

    pgy = fig1.add_subplot(212, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Power in dBy/dt')
    pgy.set_xlabel('Time [UTC]')
    pgy.set_ylim(0,np.amax(y_pwr))
    pgy.plot(idx, y_pwr)

    fig2 = plt.figure()
    title2 = '24 hr periodogram for ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title2)

    pt = fig2.add_subplot(111)
    #plt.axvline(x=5e-3, color='b') #[adds a verticle line across the axis]
    #plt.axvline(x=16e-3, color='pink')
    pxxt_s = savgol_filter(pxxt, 11, 0)
    pxyt_s = savgol_filter(pxyt, 11, 0)
    pxxtw_s = savgol_filter(pxxtw, 11, 0)
    pxytw_s = savgol_filter(pxytw, 11, 0)
    pt.semilogx(fxt,pxxt_s, label='X-Axis (periodogram)', linewidth=0.5)
    pt.semilogx(fyt,pxyt_s, label='Y-Axis (periodogram)', linewidth=0.5)
    pt.semilogx(fxtw,pxxtw_s, label='X-Axis (welch)', linewidth=0.5)
    pt.semilogx(fytw,pxytw_s, label='Y-Axis (welch)', linewidth=0.5)
    pylab.legend(loc='upper right')
    pt.set_xlabel('Frequency')
    pt.set_ylabel('Power')

    figG = plt.figure() #initalize the figure
    titleG = station + ' ULF dB(x,y,g)/dt on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(titleG)

    plt_clr = '#000000' #blue
    ln_wth= 0.1 #plot linewidth

    gx1 = figG.add_subplot(511)
    gx1.set_ylabel('dBx/dt')
    gx1.set_xticks(np.arange(0,86401,(14400/2)/2))
    gx1.set_xticklabels(np.arange(0,25,1))
    plt.setp(gx1.get_xticklabels(), visible=False)
    gx1.set_ylim(-b_max,b_max)
    gx1.plot(time_list, x_mag, plt_clr, linewidth=ln_wth)

    gx2 = figG.add_subplot(512, sharex=gx1)
    plt.setp(gx2.get_xticklabels(), visible=False)
    gx2.set_ylabel('dBy/dt')
    gx2.set_ylim(-b_max,b_max)
    gx2.plot(time_list, y_mag, plt_clr, linewidth=ln_wth)

    gx3 = figG.add_subplot(513, sharex=gx1)
    plt.setp(gx3.get_xticklabels(), visible=False)
    gx3.set_ylabel('dBg/dt')
    gx3.set_ylim(0,(b_max*2))
    gx3.plot(time_list, g_mag, plt_clr, linewidth=ln_wth)

    gx4 = figG.add_subplot(514, sharex=gx1)
    plt.setp(gx4.get_xticklabels(), visible=False)
    gx4.set_ylabel('2dBg/2dt')
    gx4.set_ylim(-b_max,b_max)
    gx4.plot(time_list, g_2d, plt_clr, linewidth=ln_wth)

    gx5 = figG.add_subplot(515, sharex=gx1)
    plt.setp(gx5.get_xticklabels(), visible=True)
    gx5.set_ylabel('dB(delta)/dt')
    gx5.set_ylim(0,0.5)
    gx5.set_xlabel('Time [UTC]')
    gx5.plot(time_list, d_mag, plt_clr, linewidth=ln_wth)

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

def ULF_multi_plot(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, station='LYR', showplot=False, saveplot=False, plot_dbdt=True, plot_pdgm=True, plot_Hz=True, sampled=True, sample_sec=60):

    #DESCRIPTION: This function creates plots from a multi-day numpy save (called from separate function)

    #------------------------------------------------------------------------------#
    #Variables and settings

    np_file =  ULF_multi_npy(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, sampled=sampled, sample_sec=sample_sec)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    hour = np.arange(0,25,1)
    time_list = np.arange(0,(len(d1)/10),.1) #creates a time list same length as the time of day array (for multiple days)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft
    f_max = 0.1 #max frequency plotting bound [Hz]
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    number_days = 1 + (date(yyyy_end, mm_end, dd_end) - date(yyyy_start, mm_start, dd_start)).days
    x_ticks = np.arange(0,(len(d1)/10+1),(len(d1)/(number_days*10)))
    x_tick_labels = np.arange(0,(number_days+1),1)

    #------------------------------------------------------------------------------#
    #Main

    if plot_Hz == True:
        #Power Plots

        Hz_1 = 0.07 #frequencies to search for and plot power

        temp_x = []
        temp_y = []
        n_pts = 864 #number of integration iteration
        idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)

        for val in range(0,len(idx)):
            if val == 0:
                fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
            elif val == (len(idx)-1):
                fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
            else:
                fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
            pwr_1 = pxx[(np.abs(fx-Hz_1)).argmin()]
            temp_x.append(pwr_1)
        x_pwr = np.asarray(temp_x)

        for val in range(0,len(idx)):
            #computes SOMETHING TO DO WITH PERIODOGRAMS
            if val == 0:
                fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
            elif val == (len(idx)-1):
                fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
            else:
                fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
            pwr_1 = pxy[(np.abs(fy-Hz_1)).argmin()]
            temp_y.append(pwr_1)
        y_pwr = np.asarray(temp_y)

        fig1 = plt.figure()
        title1 = str(Hz_1) + ' Hz Power in ' + station + ' ULF dB/dt data from ' + str(yyyy_start) + '/' + str(mm_start) + '/' + str(dd_start) + ' to ' + str(yyyy_end) + '/' + str(mm_end) + '/' + str(dd_end)
        plt.suptitle(title1)

        pgx = fig1.add_subplot(211)
        pgx.set_xticks(np.arange(0,(len(d1)+1),(len(d1)/number_days)))
        pgx.set_xticklabels(x_tick_labels)
        plt.setp(pgx.get_xticklabels(), visible=True)
        pgx.set_ylabel('Power in dBx/dt')
        pgx.set_ylim(0,np.amax(x_pwr))
        pgx.plot(idx, x_pwr)

        pgy = fig1.add_subplot(212, sharex=pgx)
        plt.setp(pgy.get_xticklabels(), visible=True)
        pgy.set_ylabel('Power in dBy/dt')
        pgy.set_xlabel('Days from Period Beginning')
        pgy.set_ylim(0,np.amax(y_pwr))
        pgy.plot(idx, y_pwr)

    if plot_pdgm == True:
        #Periodogram Plot
        fxt, pxxt = signal.periodogram(x_mag,10e2) #plots of whole 24 hr period
        fyt, pxyt = signal.periodogram(y_mag,10e2)
        #fxtw, pxxtw = signal.welch(x_mag,10e2) #plots of whole 24 hr period
        #fytw, pxytw = signal.welch(y_mag,10e2)

        fig2 = plt.figure()
        title2 = 'Periodogram for ' + station + ' ULF dB/dt data from ' + str(yyyy_start) + '/' + str(mm_start) + '/' + str(dd_start) + ' to ' + str(yyyy_end) + '/' + str(mm_end) + '/' + str(dd_end)
        plt.suptitle(title2)

        pt = fig2.add_subplot(111)
        #plt.axvline(x=5e-3, color='b') #[adds a verticle line across the axis]
        #plt.axvline(x=16e-3, color='pink')
        pxxt_s = savgol_filter(pxxt, 11, 0)
        pxyt_s = savgol_filter(pxyt, 11, 0)
        #pxxtw_s = savgol_filter(pxxtw, 11, 0)
        #pxytw_s = savgol_filter(pxytw, 11, 0)
        pt.semilogx(fxt,pxxt_s, label='X-Axis (periodogram)', linewidth=0.5)
        pt.semilogx(fyt,pxyt_s, label='Y-Axis (periodogram)', linewidth=0.5)
        #pt.semilogx(fxtw,pxxtw_s, label='X-Axis (welch)', linewidth=0.5)
        #pt.semilogx(fytw,pxytw_s, label='Y-Axis (welch)', linewidth=0.5)
        pylab.legend(loc='upper right')
        pt.set_xlabel('Frequency')
        pt.set_ylabel('Power')

    if plot_dbdt == True:
        #calculations for B-field

        Bx_field = [] #create empty lists
        By_field = []
        n_pts = 864 #number of integration iterations, equal to resolution of 100 sec
        idx = np.linspace(len(d1)/n_pts, len(d1), n_pts, dtype=int) #indexing vector (can't start at zero)

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
        title = station + ' ULF B-data from ' + str(yyyy_start) + '/' + str(mm_start) + '/' + str(dd_start) + ' to ' + str(yyyy_end) + '/' + str(mm_end) + '/' + str(dd_end)
        plt.suptitle(title)

        plt_clr = '#000000' #blue
        ln_wth= 0.1 #plot linewidth

        ax1 = fig.add_subplot(611) #adds a subplot to the figure, with 6 rows, 1 column, and first subplot at index 1 (i.e. on top)
        ax1.set_xticks(x_ticks) #arrange spacing of x-axis ticks the same length as data
        ax1.set_xticklabels(x_tick_labels) #creates x-axis tick labels for 24 UTC hours (redundant here)
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
        ax3.set_xticks(x_ticks)
        ax3.set_xticklabels(x_tick_labels)
        plt.setp(ax3.get_xticklabels(), visible=False)
        ax3.set_ylim(-b_max,b_max)
        ax3.plot(time_list, x_mag, plt_clr, linewidth=ln_wth)

        ax4 = fig.add_subplot(614, sharex=ax3)
        plt.setp(ax4.get_xticklabels(), visible=False)
        ax4.set_ylabel('dBy/dt')
        ax4.set_ylim(-b_max,b_max)
        ax4.plot(time_list, y_mag, plt_clr, linewidth=ln_wth)

        ax5 = fig.add_subplot(615)
        ax5.set_xticks(x_ticks)
        ax5.set_xticklabels(x_tick_labels)
        plt.setp(ax5.get_xticklabels(), visible=False)
        ax5.set_ylabel('(dBx/dt)^2')
        ax5.set_ylim(-nT_range,nT_range)
        #ax5.set_ylim(-np.amax(x_2d),np.amax(x_2d))
        ax5.plot(time_list, x_2d, plt_clr, linewidth=ln_wth)

        ax6 = fig.add_subplot(616, sharex=ax5)
        plt.setp(ax6.get_xticklabels(), visible=True)
        ax6.set_ylabel('(dBy/dt)^2')
        ax6.set_xlabel('Days from Period Beginning')
        ax6.set_ylim(-nT_range,nT_range)
        #ax6.set_ylim(-np.amax(y_2d),np.amax(y_2d))
        ax6.plot(time_list, y_2d, plt_clr, linewidth=ln_wth)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/'
    save_name = station+ '_from_' + str(yyyy_start) + str(mm_start) + str(dd_start) + '_to_' + str(yyyy_end) + str(mm_end) + str(dd_end)
    png_file = png_dir + save_name + '_B_field_plots.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig.savefig(png_file, bbox_inches='tight')


#Frequency/power plots
def periodogram_multi_freq(yyyy, mm, dd, freq, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #Note: the showplot functionality currently opens only one plot at a time...issue with plt.show() acting on all plots

    #------------------------------------------------------------------------------#
    #Variables and settings

    colors = cycle(["red", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "aqua", "silver", "teal", "yellow"])

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    hour = np.arange(0,25,1)
    time_list = np.arange(0,86400,.1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft
    f_max = 0.1 #max frequency plotting bound [Hz]
    b_max = 0.2 #+/- db/dt plotting range [nT/s]

    #------------------------------------------------------------------------------#
    #Calculations

    temp_x = []
    temp_y = []
    pwr = np.zeros((1, len(freq)))
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)
    for val in range(0,len(idx)):
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        for i in range(0,len(freq)):
            pwr[0,i] = pxx[(np.abs(fx-freq[i])).argmin()]
        if len(temp_x) == 0:
            temp_x = pwr
        else:
            temp_x = np.append(temp_x, pwr, axis=0)
    x_pwr = np.asarray(temp_x)

    for val in range(0,len(idx)):
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        for i in range(0,len(freq)):
            pwr[0,i] = pxy[(np.abs(fy-freq[i])).argmin()]
        if len(temp_y) == 0:
            temp_y = pwr
        else:
            temp_y = np.append(temp_y, pwr, axis=0)
    y_pwr = np.asarray(temp_y)

    #------------------------------------------------------------------------------#
    #plotting

    fig1 = plt.figure()
    ###fig1.tight_layout()
    title1 = 'Periodogram of Hz Power in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title1)

    pgx = fig1.add_subplot(211)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(np.arange(0,25,1))
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Power of dBx/dt')
    pgx.set_yscale('log')
    pgx.set_ylim(10e-6,10)

    pgy = fig1.add_subplot(212, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Power of dBy/dt')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    pgy.set_yscale('log')
    pgy.set_ylim(10e-6,10)

    for i, item in enumerate(freq):
        Hz_color = next(colors)
        x_pwr_smooth = savgol_filter(x_pwr[:,i], 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
        pgx.plot(idx, x_pwr_smooth, label=str(item) + ' Hz', color=Hz_color)
        y_pwr_smooth = savgol_filter(y_pwr[:,i], 11, 0)
        pgy.plot(idx, y_pwr_smooth, label=str(item) + ' Hz', color=Hz_color)

    pgx.legend(loc="upper right", fontsize=6)
    pgy.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving file

    if showplot == True:

            plt.show()

    if saveplot == True:

        png_dir = 'C:/Users/Tyler/Desktop/Periodograms/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_periodogram.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_periodogram.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_periodogram.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_periodogram.png'

        plt.savefig(png_file, bbox_inches='tight')

def periodogram_multi_freq_dHzdt(yyyy, mm, dd, freq, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #Note: the showplot functionality currently opens only one plot at a time...issue with plt.show() acting on all plots

    colors = cycle(["red", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "aqua", "silver", "teal", "yellow"])

    data_dir = 'C:/Users/Tyler/Desktop/Project Lisbon/Datasets/ULF/LYR/TXT/'

    #modifies the string to the correct format given the date
    if mm >= 10 and dd < 10: #single digit days
        data_location = data_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_dbdt_v1.txt'
    elif mm < 10 and dd >= 10: #single digit months
        data_location = data_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_dbdt_v1.txt'
    elif mm < 10 and dd < 10: #single digit days and months
        data_location = data_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_dbdt_v1.txt'
    else: #multi-digit days and months
        data_location = data_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_dbdt_v1.txt'

    #check to make sure the specified text file exits and stops program otherwise
    txt_path = Path(data_location)
    if txt_path.is_file() is False:
        print('Note: Specified ULF data file does not exist')
        sys.exit()

    temp = data_location.replace('_dbdt_v1.txt', '.npy')
    np_file = temp.replace('TXT', 'NPY')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for that date')
    else:
        datapath = open(data_location, 'r')
        dat = datapath.readlines()
        datapath.close()
        #sys.exit() #[this stops the program...may be a python2 vs. python3 issue]
        dat = dat[3:] #removes the header

        dataset = []
        for index, item in enumerate(dat,start=0):
            line = dat[index].split()
            line = [float(i) for i in line]
            dataset.append(line)

        ulf_dat = np.asarray(dataset)

        np.save(np_file,ulf_dat)

    #------------------------------------------------------------------------------#
    #Variables and settings

    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]

    #------------------------------------------------------------------------------#
    #Calculations

    temp_x = []
    temp_y = []
    pwr = np.zeros((1, len(freq)))
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)
    for val in range(0,len(idx)):
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        for i in range(0,len(freq)):
            pwr[0,i] = pxx[(np.abs(fx-freq[i])).argmin()]
        if len(temp_x) == 0:
            temp_x = pwr
        else:
            temp_x = np.append(temp_x, pwr, axis=0)
    x_pwr = np.asarray(temp_x)

    for val in range(0,len(idx)):
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        for i in range(0,len(freq)):
            pwr[0,i] = pxy[(np.abs(fy-freq[i])).argmin()]
        if len(temp_y) == 0:
            temp_y = pwr
        else:
            temp_y = np.append(temp_y, pwr, axis=0)
    y_pwr = np.asarray(temp_y)

    #------------------------------------------------------------------------------#
    #plotting

    fig2 = plt.figure()
    ###fig2.tight_layout()
    title1 = 'Periodogram of Hz Power in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title1)

    pgx = fig2.add_subplot(211)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(np.arange(0,25,1))
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Power Gradient of dBx/dt')
    #pgx.set_ylim(10e-6,10)

    pgy = fig2.add_subplot(212, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Power Gradient of dBy/dt')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    #pgy.set_ylim(10e-6,10)

    for i, item in enumerate(freq):
        Hz_color = next(colors)
        x_grad = np.gradient(x_pwr[:,i],idx)
        #x_grad_smooth = savgol_filter(x_grad, 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
        pgx.plot(idx, x_grad, label=str(item) + ' Hz', color=Hz_color)
        y_grad = np.gradient(y_pwr[:,i],idx)
        #y_grad_smooth = savgol_filter(y_grad, 11, 0)
        pgy.plot(idx, y_grad, label=str(item) + ' Hz', color=Hz_color)

    pgx.legend(loc="upper right", fontsize=6)
    pgy.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    if showplot == True:

        plt.show()

    if saveplot == True:

        png_dir = 'C:/Users/Tyler/Desktop/Periodograms/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_periodogram_dHzdt.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_periodogram_dHzdt.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_periodogram_dHzdt.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_periodogram_dHzdt.png'

        plt.savefig(png_file, bbox_inches='tight')

def total_power(yyyy, mm, dd, station='LYR', smoothed=False):

    #DESCRIPTION: returns two arrays of ULF freq data (first for max power freq and second for total power)

    #------------------------------------------------------------------------------#
    #Variables and settings

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]

    #------------------------------------------------------------------------------#
    #Calculations

    temp_x = []
    temp_y = []
    temp_x_int = []
    temp_y_int = []
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype='int')
    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        x_Hz_max = fx[np.argmax(pxx)]
        temp_x.append(x_Hz_max)
        temp_x_int.append(scipy.integrate.trapz(pxx, x=None, dx=(fx[1]-fx[0])))
    x_max = np.asarray(temp_x)
    x_int = np.asarray(temp_x_int)

    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        y_Hz_max = fy[np.argmax(pxy)]
        temp_y.append(y_Hz_max)
        temp_y_int.append(scipy.integrate.trapz(pxy, x=None, dx=(fy[1]-fy[0])))
    y_max = np.asarray(temp_y)
    y_int = np.asarray(temp_y_int)

    if smoothed == True:
        x_max_smooth = savgol_filter(x_max, 11, 0)
        y_max_smooth = savgol_filter(y_max, 11, 0)
        x_int_smooth = savgol_filter(x_int, 11, 0)
        y_int_smooth = savgol_filter(y_int, 11, 0)

        max_smoothed = np.column_stack((x_max_smooth, y_max_smooth))
        int_smoothed = np.column_stack((x_int_smooth, y_int_smooth))
        return max_smoothed, int_smoothed

    else:
        max = np.column_stack((x_max, y_max))
        int = np.column_stack((x_int, y_int))
        return max, int

def total_power_plot(yyyy, mm, dd, station='LYR', Hz_max=0.1, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    #------------------------------------------------------------------------------#
    #Variables and settings

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]

    #------------------------------------------------------------------------------#
    #Calculations

    temp_x = []
    temp_y = []
    temp_x_int = []
    temp_y_int = []
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype='int')
    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        x_Hz_max = fx[np.argmax(pxx)]
        temp_x.append(x_Hz_max)
        temp_x_int.append(scipy.integrate.trapz(pxx, x=None, dx=(fx[1]-fx[0])))
    x_max = np.asarray(temp_x)
    x_int = np.asarray(temp_x_int)

    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        y_Hz_max = fy[np.argmax(pxy)]
        temp_y.append(y_Hz_max)
        temp_y_int.append(scipy.integrate.trapz(pxy, x=None, dx=(fy[1]-fy[0])))
    y_max = np.asarray(temp_y)
    y_int = np.asarray(temp_y_int)

    #save arrays to a text file for viewing, but only if same length
    #np.savetxt('temp.txt', np.c_[fx,fy], header="frequencies")

    #------------------------------------------------------------------------------#
    #plotting

    fig4 = plt.figure()
    ###fig4.tight_layout()
    title4 = 'Total power and Max Power Frequency from 0-5 Hz in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title4)

    pgx = fig4.add_subplot(411)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(np.arange(0,25,1))
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Frequency in dBx/dt [Hz]')
    pgx.set_ylim(0,Hz_max)

    pgy = fig4.add_subplot(412, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Frequency in dBy/dt [Hz]')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    pgy.set_ylim(0,Hz_max)

    pgx2 = fig4.add_subplot(413)
    pgx2.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx2.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx2.set_xticklabels(np.arange(0,25,1))
    plt.setp(pgx2.get_xticklabels(), visible=True)
    pgx2.set_ylabel('Total Power of dBx/dt')
    pgx2.set_yscale('log')
    pgx2.set_ylim(10e-6,10)

    pgy2 = fig4.add_subplot(414, sharex=pgx2)
    plt.setp(pgy2.get_xticklabels(), visible=True)
    pgy2.set_ylabel('Total Power of dBy/dt')
    pgy2.set_xlabel('Time [UTC]', fontsize=10)
    pgy2.set_yscale('log')
    pgy2.set_ylim(10e-6,10)

    plotcolor= '#000000' #black

    if smoothed == True:
        x_max_smooth = savgol_filter(x_max, 11, 0)
        pgx.plot(idx, x_max_smooth, color=plotcolor)
        y_max_smooth = savgol_filter(y_max, 11, 0)
        pgy.plot(idx, y_max_smooth, color=plotcolor)
        x_int_smooth = savgol_filter(x_int, 11, 0)
        pgx2.plot(idx, x_int_smooth, color=plotcolor)
        y_int_smooth = savgol_filter(x_int, 11, 0)
        pgy2.plot(idx, y_int_smooth, color=plotcolor)

    else:
        pgx.plot(idx, x_max, color=plotcolor)
        pgy.plot(idx, y_max, color=plotcolor)
        pgx2.plot(idx, x_int, color=plotcolor)
        pgy2.plot(idx, y_int, color=plotcolor)

    #------------------------------------------------------------------------------#
    #saving file

    if showplot == True:

        plt.show()

    if saveplot == True:

        png_dir = 'C:/Users/Tyler/Desktop/plotz/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_total_power.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_total_power.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_total_power.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_total_power.png'

        plt.savefig(png_file, bbox_inches='tight')

def Pc_power_bands(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: Plots total power in Pc bands 1-5

    #Note: Currently smoothing is automatic and non-toggleable
    #min = 0.00998

    #Pc1 = 0.2 - 5 Hz --> manual indices 21-1001
    #Pc2 = 0.1 - 0.2 Hz --> manual indices 11-20
    #Pc3 = 0.022 - 0.1 Hz --> manual indices 3-10
    #Pc4 = 0.007 - 0.022 Hz --> manual indices 1-2
    #Pc5 = 0.002 - 0.007 Hz --> manual indices 0-1

    freq = [0.002, 0.007, 0.022, 0.01, 0.02]
    Pc_bands = ['Pc5 Band', 'Pc4 Band', 'Pc3 Band', 'Pc2 Band', 'Pc1 Band']
    freq_idx = np.array([[0, 1, 3, 11, 21],[1, 2, 10, 20, 1001]], dtype=int)

    #------------------------------------------------------------------------------#
    #Variables and settings

    colors = cycle(["red", "black", "blue", "fuchsia", "gray", "green", "lime", "maroon", "navy", "olive", "purple", "aqua", "silver", "teal", "yellow"])

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]

    #------------------------------------------------------------------------------#
    #Calculations

    temp_x = []
    temp_y = []
    pwr = np.zeros((1, len(freq)))
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)
    for val in range(0,len(idx)):
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)

        for i in range(0,len(freq)):
            #f_start = (np.abs(fx-freq[i])).argmin()
            #f_stop = (np.abs(fx-5)).argmin()
            f_start = freq_idx[0,i]
            f_stop = freq_idx[1,i]
            pwr[0,i] = scipy.integrate.trapz(pxx[f_start:f_stop], x=None, dx=(fx[1]-fx[0]))
        if len(temp_x) == 0:
            temp_x = pwr
        else:
            temp_x = np.append(temp_x, pwr, axis=0)
    x_pwr = np.asarray(temp_x)

    for val in range(0,len(idx)):
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)

        for i in range(0,len(freq)):
            #f_start = (np.abs(fx-freq[i])).argmin()
            #f_stop = (np.abs(fx-5)).argmin()
            f_start = freq_idx[0,i]
            f_stop = freq_idx[1,i]
            pwr[0,i] = scipy.integrate.trapz(pxy[f_start:f_stop], x=None, dx=(fy[1]-fy[0]))
        if len(temp_y) == 0:
            temp_y = pwr
        else:
            temp_y = np.append(temp_y, pwr, axis=0)
    y_pwr = np.asarray(temp_y)

    #------------------------------------------------------------------------------#
    #plotting

    fig1 = plt.figure()
    ###fig1.tight_layout()
    title1 = 'Periodogram of Power in Pc 1-5 Bands at ' + station + ' pn ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title1)

    pgx = fig1.add_subplot(211)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(np.arange(0,25,1))
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Power of dBx/dt')
    pgx.set_yscale('log')
    #pgx.set_ylim(10e-6,10)

    pgy = fig1.add_subplot(212, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Power of dBy/dt')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    pgy.set_yscale('log')
    #pgy.set_ylim(10e-6,10)

    for i, item in enumerate(Pc_bands):
        Hz_color = next(colors)
        x_pwr_smooth = savgol_filter(x_pwr[:,i], 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
        pgx.plot(idx, x_pwr_smooth, label=str(item) + ' Hz', color=Hz_color)
        y_pwr_smooth = savgol_filter(y_pwr[:,i], 11, 0)
        pgy.plot(idx, y_pwr_smooth, label=str(item) + ' Hz', color=Hz_color)

    #np.savetxt('freq.txt', np.c_[fx, fy])
    #np.savetxt('text.txt', x_pwr)

    pgx.legend(loc="upper right", fontsize=6)
    pgy.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving file

    if showplot == True:
            plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/Periodograms/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_periodogram.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_periodogram.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_periodogram.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_periodogram.png'

        plt.savefig(png_file, bbox_inches='tight')


#Multi-day frequency/power plots
def ULF_day_spec(yyyy, mm, dd, station='LYR', printfile=False):

    #DESCRIPTION: This function generates two arrays of max frequency and integrated power for a 24 hr period, saves as numpy files, and returns full file path

    #------------------------------------------------------------------------------#
    #Variables and settings

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    #####d_type = 'float32' #to sace numpy file as smaller data type

    #------------------------------------------------------------------------------#
    #Calculations

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    temp_x = []
    temp_y = []
    temp_x_int = []
    temp_y_int = []
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)
    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        x_Hz_max = fx[np.argmax(pxx)]
        temp_x.append(x_Hz_max)
        temp_x_int.append(scipy.integrate.trapz(pxx, x=None, dx=(fx[1]-fx[0])))
    x_max = np.asarray(temp_x)
    x_int = np.asarray(temp_x_int)

    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        y_Hz_max = fy[np.argmax(pxy)]
        temp_y.append(y_Hz_max)
        temp_y_int.append(scipy.integrate.trapz(pxy, x=None, dx=(fy[1]-fy[0])))
    y_max = np.asarray(temp_y)
    y_int = np.asarray(temp_y_int)

    max_arr = np.column_stack((idx, x_max, y_max))
    int_arr = np.column_stack((idx, x_int, y_int))

    max_file = np_file.replace('.npy', '_max_Hz.npy')
    int_file = np_file.replace('.npy', '_int_power.npy')

    if printfile == True:
        np.savetxt('freq.txt', np.c_[max_arr, int_arr])

    np.save(max_file, max_arr)
    np.save(int_file, int_arr)

    return max_file, int_file

def ULF_day_spec_grad(yyyy, mm, dd, station='LYR', printfile=False):

    #DESCRIPTION: This function generates two arrays of max frequency and integrated power for a 24 hr period, saves as numpy files, and returns full file path

    #------------------------------------------------------------------------------#
    #Variables and settings

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    #####d_type = 'float32' #to sace numpy file as smaller data type

    #------------------------------------------------------------------------------#
    #Calculations

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    temp_x = []
    temp_y = []
    temp_x_int = []
    temp_y_int = []
    n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)
    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fx, pxx = signal.periodogram(x_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val]],fs)
        else:
            fx, pxx = signal.periodogram(x_mag[idx[val-1]:idx[val+1]],fs)
        x_Hz_max = fx[np.argmax(pxx)]
        temp_x.append(x_Hz_max)
        temp_x_int.append(scipy.integrate.trapz(pxx, x=None, dx=(fx[1]-fx[0])))
    x_max = np.asarray(np.gradient(temp_x, idx))
    x_int = np.asarray(np.gradient(temp_x_int, idx))

    for val in range(0,len(idx)):
        #computes SOMETHING TO DO WITH PERIDOGRAMS
        if val == 0:
            fy, pxy = signal.periodogram(y_mag[0:idx[val+1]],fs)
        elif val == (len(idx)-1):
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val]],fs)
        else:
            fy, pxy = signal.periodogram(y_mag[idx[val-1]:idx[val+1]],fs)
        y_Hz_max = fy[np.argmax(pxy)]
        temp_y.append(y_Hz_max)
        temp_y_int.append(scipy.integrate.trapz(pxy, x=None, dx=(fy[1]-fy[0])))
    y_max = np.asarray(np.gradient(temp_y, idx))
    y_int = np.asarray(np.gradient(temp_y_int, idx))

    max_arr = np.column_stack((idx, x_max, y_max))
    int_arr = np.column_stack((idx, x_int, y_int))

    max_file = np_file.replace('.npy', '_max_Hz_grad.npy')
    int_file = np_file.replace('.npy', '_int_power_grad.npy')

    if printfile == True:
        np.savetxt('grad.txt', np.c_[max_arr, int_arr])

    np.save(max_file, max_arr)
    np.save(int_file, int_arr)

    return max_file, int_file

def total_power_multi(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, station='LYR', Hz_max=0.1, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: This function plots max power frequency and integrated frequency power over a date range

    #------------------------------------------------------------------------------#
    #Variables and settings

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]

    #------------------------------------------------------------------------------#
    #Calculations

    max_total = np.array([], dtype='float32') #creates an empty numpy array
    int_total = np.array([], dtype='float32') #creates an empty numpy array
    for year in range(yyyy_start,(yyyy_end+1)):
        for month in range(mm_start,(mm_end+1)):
            if month == mm_end:
                for day in range(dd_start,(dd_end+1)):
                    np_file1, np_file2 = ULF_day_spec(year, month, day)
                    max_arr = np.load(np_file1)
                    int_arr = np.load(np_file2)
                    max_total = np.vstack([max_total, max_arr]) if max_total.size else max_arr
                    int_total = np.vstack([int_total, int_arr]) if int_total.size else int_arr
            else:
                start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                for day in range(1,(day_range+1)):
                    np_file1, np_file2 = ULF_day_spec(year, month, day)
                    max_arr = np.load(np_file1)
                    int_arr = np.load(np_file2)
                    max_total = np.vstack([max_total, max_arr]) if max_total.size else max_arr
                    int_total = np.vstack([int_total, int_arr]) if int_total.size else int_arr

    np.savetxt('text.txt', np.c_[max_total, int_total])

    idx = np.arange(0, len(max_total[:,0]))
    x_max = max_total[:,1]
    y_max = max_total[:,2]
    x_int = int_total[:,1]
    y_int = int_total[:,2]

    number_days = 1 + (date(yyyy_end, mm_end, dd_end) - date(yyyy_start, mm_start, dd_start)).days
    x_ticks = np.arange(0,(len(idx)+1),(len(idx)/(number_days)))
    x_tick_labels = np.arange(0,(number_days+1),1)

    #------------------------------------------------------------------------------#
    #plotting

    fig4 = plt.figure()
    ###fig4.tight_layout()
    title = 'Total power and Max Power Frequency from 0-5 Hz at ' + station + ' from ' + str(yyyy_start) + '/' + str(mm_start) + '/' + str(dd_start) + ' to ' + str(yyyy_end) + '/' + str(mm_end) + '/' + str(dd_end)
    plt.suptitle(title)

    pgx = fig4.add_subplot(411)
    pgx.set_xticks(x_ticks)
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(x_tick_labels)
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Frequency [Hz]')
    pgx.set_ylim(0,Hz_max)

    pgy = fig4.add_subplot(412, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Frequency [Hz]')
    pgy.set_ylim(0,Hz_max)

    pgx2 = fig4.add_subplot(413)
    pgx2.set_xticks(x_ticks)
    #pgx2.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx2.set_xticklabels(x_tick_labels)
    plt.setp(pgx2.get_xticklabels(), visible=True)
    pgx2.set_ylabel('Total Power of dBx/dt')
    pgx2.set_yscale('log')
    pgx2.set_ylim(10e-6,10)

    pgy2 = fig4.add_subplot(414, sharex=pgx2)
    plt.setp(pgy2.get_xticklabels(), visible=True)
    pgy2.set_ylabel('Total Power of dBy/dt')
    pgy2.set_xlabel('Days from Period Beginning', fontsize=10)
    pgy2.set_yscale('log')
    pgy2.set_ylim(10e-6,10)

    plotcolor= '#000000' #black

    if smoothed == True:
        x_max_smooth = savgol_filter(x_max, 11, 0)
        pgx.plot(idx, x_max_smooth, color=plotcolor)
        y_max_smooth = savgol_filter(y_max, 11, 0)
        pgy.plot(idx, y_max_smooth, color=plotcolor)
        x_int_smooth = savgol_filter(x_int, 11, 0)
        pgx2.plot(idx, x_int_smooth, color=plotcolor)
        y_int_smooth = savgol_filter(x_int, 11, 0)
        pgy2.plot(idx, y_int_smooth, color=plotcolor)

    else:
        pgx.plot(idx, x_max, color=plotcolor)
        pgy.plot(idx, y_max, color=plotcolor)
        pgx2.plot(idx, x_int, color=plotcolor)
        pgy2.plot(idx, y_int, color=plotcolor)

    #------------------------------------------------------------------------------#
    #saving file

    if showplot == True:
        plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/plotz/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_total_power.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_total_power.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_total_power.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_total_power.png'

        plt.savefig(png_file, bbox_inches='tight')

def total_power_multi_grad(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, station='LYR', Hz_max=0.1, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: This function plots max power frequency and integrated frequency power over a date range

    #------------------------------------------------------------------------------#
    #Variables and settings

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]

    #------------------------------------------------------------------------------#
    #Calculations

    max_total = np.array([], dtype='float32') #creates an empty numpy array
    int_total = np.array([], dtype='float32') #creates an empty numpy array
    for year in range(yyyy_start,(yyyy_end+1)):
        for month in range(mm_start,(mm_end+1)):
            if month == mm_end:
                for day in range(dd_start,(dd_end+1)):
                    np_file1, np_file2 = ULF_day_spec_grad(year, month, day)
                    max_arr = np.load(np_file1)
                    int_arr = np.load(np_file2)
                    max_total = np.vstack([max_total, max_arr]) if max_total.size else max_arr
                    int_total = np.vstack([int_total, int_arr]) if int_total.size else int_arr
            else:
                start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                for day in range(1,(day_range+1)):
                    np_file1, np_file2 = ULF_day_spec_grad(year, month, day)
                    max_arr = np.load(np_file1)
                    int_arr = np.load(np_file2)
                    max_total = np.vstack([max_total, max_arr]) if max_total.size else max_arr
                    int_total = np.vstack([int_total, int_arr]) if int_total.size else int_arr

    np.savetxt('text.txt', np.c_[max_total, int_total])

    idx = np.arange(0, len(max_total[:,0]))
    x_max = max_total[:,1]
    y_max = max_total[:,2]
    x_int = int_total[:,1]
    y_int = int_total[:,2]

    number_days = 1 + (date(yyyy_end, mm_end, dd_end) - date(yyyy_start, mm_start, dd_start)).days
    x_ticks = np.arange(0,(len(idx)+1),(len(idx)/(number_days)))
    x_tick_labels = np.arange(0,(number_days+1),1)

    #------------------------------------------------------------------------------#
    #plotting

    fig4 = plt.figure()
    ###fig4.tight_layout()
    title = 'Gradient of Total power and Max Power Frequency from 0-5 Hz at ' + station + ' from ' + str(yyyy_start) + '/' + str(mm_start) + '/' + str(dd_start) + ' to ' + str(yyyy_end) + '/' + str(mm_end) + '/' + str(dd_end)
    plt.suptitle(title)

    pgx = fig4.add_subplot(411)
    pgx.set_xticks(x_ticks)
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(x_tick_labels)
    plt.setp(pgx.get_xticklabels(), visible=True)
    pgx.set_ylabel('Frequency [Hz]')
    #pgx.set_ylim(-Hz_max,Hz_max)

    pgy = fig4.add_subplot(412, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Frequency [Hz]')
    #pgy.set_ylim(-Hz_max,Hz_max)

    pgx2 = fig4.add_subplot(413)
    pgx2.set_xticks(x_ticks)
    #pgx2.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx2.set_xticklabels(x_tick_labels)
    plt.setp(pgx2.get_xticklabels(), visible=True)
    pgx2.set_ylabel('Gradient of Total Power of dBx/dt')
    #pgx2.set_yscale('log')
    #pgx2.set_ylim(10e-8,10e-2)

    pgy2 = fig4.add_subplot(414, sharex=pgx2)
    plt.setp(pgy2.get_xticklabels(), visible=True)
    pgy2.set_ylabel('Gradient of Total Power of dBy/dt')
    pgy2.set_xlabel('Days from Period Beginning', fontsize=10)
    #pgy2.set_yscale('log')
    #pgy2.set_ylim(10e-8,10e-2)

    plotcolor= '#000000' #black

    if smoothed == True:
        x_max_smooth = savgol_filter(x_max, 11, 0)
        pgx.plot(idx, x_max_smooth, color=plotcolor)
        y_max_smooth = savgol_filter(y_max, 11, 0)
        pgy.plot(idx, y_max_smooth, color=plotcolor)
        x_int_smooth = savgol_filter(x_int, 11, 0)
        pgx2.plot(idx, x_int_smooth, color=plotcolor)
        y_int_smooth = savgol_filter(x_int, 11, 0)
        pgy2.plot(idx, y_int_smooth, color=plotcolor)

    else:
        pgx.plot(idx, x_max, color=plotcolor)
        pgy.plot(idx, y_max, color=plotcolor)
        pgx2.plot(idx, x_int, color=plotcolor)
        pgy2.plot(idx, y_int, color=plotcolor)

    #------------------------------------------------------------------------------#
    #saving file

    if showplot == True:
        plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/plotz/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_' + station + '_total_power.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_' + station + '_total_power.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_' + station + '_total_power.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_' + station + '_total_power.png'

        plt.savefig(png_file, bbox_inches='tight')


#Geo-solar data analysis
def ACES_txt2npy(yyyy, mm):

    #DESCRIPTION: This function opens a NASA ACE txt file and saves as a numpy file
    #Returns the full path file name of the specified numpy file

    data_dir = 'C:/Users/Tyler/Desktop/Project Lisbon/Datasets/Geophysical/NASA_ACES/'

    #modifies the string to the correct format given the date
    if mm >= 10:
        txt_location = data_dir + str(yyyy) + str(mm) + '_ace_mag_1h.txt'
    elif mm < 10:
        txt_location = data_dir + str(yyyy) + '0' + str(mm) + '_ace_mag_1h.txt'

    #check to make sure the specified text file exits and stops program otherwise
    txt_path = Path(txt_location)
    if txt_path.is_file() is False:
        print('Note: Specified ACE data file does not exist')
        print(txt_location)
        sys.exit()

    np_file = txt_location.replace('_ace_mag_1h.txt', '.npy')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for that date')
    else:
        datapath = open(txt_location, 'r')
        dat = datapath.readlines()
        datapath.close()
        dat = dat[20:] #removes the header

        dataset = []
        for index, item in enumerate(dat,start=0):
            line = dat[index].split()
            line = [float(i) for i in line]
            dataset.append(line)

        ulf_dat = np.asarray(dataset, dtype='float32') #smaller than float64 type to save memory space

        np.save(np_file,ulf_dat)

    return np_file

def ACES_plots(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION:

    #Data taken from: ftp://ftp.ngdc.noaa.gov/STP/SOLAR_DATA/SATELLITE_ENVIRONMENT/ACE_SATELLITE_PLOTS/Interplanetary_Magnetic_Field/

    #NOTE: for the last day in month, there is a 1 hr. UTC offset due to only having 0-23, not 0-24

    #------------------------------------------------------------------------------#
    #Variables and settings

    np_file = ACES_txt2npy(yyyy, mm)
    dat_arr = np.load(np_file)

    start_DOTW, day_range = monthrange(yyyy,mm)
    if dd == day_range: #doesn't take the 0000 (2400) UTC measurement from next day if last in month
        dd_start = 24*(dd-1)
        dd_end = 24*dd #corresponding to index 0-23 for hours 0-23
        time_list = np.arange(0,24,1)
    else:
        dd_start = 24*(dd-1)
        dd_end = 24*dd + 1 #corresponding to index 0-24 for hours 0-24
        time_list = np.arange(0,25,1)


    #grabs desired section of numpy file corresponding to the day specified
    year = dat_arr[dd_start:dd_end,0]
    month = dat_arr[dd_start:dd_end,1]
    day = dat_arr[dd_start:dd_end,2]
    HHMM = dat_arr[dd_start:dd_end,3] #time of measurement in HHMM format
    julian = dat_arr[dd_start:dd_end,4] #Julian calandar date
    SOD = dat_arr[dd_start:dd_end,5] #seconds of the day (0-82800)
    S = dat_arr[dd_start:dd_end,6] #Status(S): 0 = nominal data, 1 to 8 = bad data record, 9 = no data
    Bx = dat_arr[dd_start:dd_end,7] #solar wind magnetic field component
    By = dat_arr[dd_start:dd_end,8] #solar wind magnetic field component
    Bz = dat_arr[dd_start:dd_end,9] #solar wind magnetic field component
    Bt = dat_arr[dd_start:dd_end,10] #solar wind magnetic field component
    Lat = dat_arr[dd_start:dd_end,11] #satellite location at measurement in GSE
    Long = dat_arr[dd_start:dd_end,12] #satellite location at measurement in GSE

    np.savetxt('text.txt', np.c_[year, month, day, HHMM, julian, SOD, S, Bx, By, Bz, Bt, Lat, Long])

    hour = np.arange(0,25,1)

    dt = 3600 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    pwr_max, pwr_int = total_power(yyyy, mm, dd, station=station, smoothed=True)
    idx = np.linspace(0, (864000-1), len(pwr_max))

    ULF_calc = 6 * Bt / 1000 #expected ULF main freq in Hz, from eqn: f (mHz) ≈ 6 BIMF (nT) [Takahashi et al., 1984]

    #plotting
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'ACES solar wind B(x,y,z,t) on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    x1 = fig.add_subplot(411)
    x1.set_ylabel('Bx [nT]')
    x1.set_xticks(time_list)
    x1.set_xticklabels(hour)
    plt.setp(x1.get_xticklabels(), visible=False)
    #x1.set_ylim(-b_max,b_max)
    x1.plot(time_list, Bx, plt_clr, linewidth=ln_wth)

    x2 = fig.add_subplot(412, sharex=x1)
    plt.setp(x2.get_xticklabels(), visible=False)
    x2.set_ylabel('By [nT]')
    #x2.set_ylim(-b_max,b_max)
    x2.plot(time_list, By, plt_clr, linewidth=ln_wth)

    x3 = fig.add_subplot(413, sharex=x1)
    plt.setp(x3.get_xticklabels(), visible=False)
    x3.set_ylabel('Bz [nT]')
    #x3.set_ylim(0,(b_max*2))
    x3.plot(time_list, Bz, plt_clr, linewidth=ln_wth)

    x4 = fig.add_subplot(414, sharex=x1)
    plt.setp(x4.get_xticklabels(), visible=True)
    x4.set_ylabel('Bt [nT]')
    x4.set_xlabel('Time [UTC]')
    #x4.set_ylim(-b_max,b_max)
    x4.plot(time_list, Bt, plt_clr, linewidth=ln_wth)


    fig2 = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi)
    ###fig4.tight_layout()
    title2 = 'ULF Max Power Frequency from 0-5 Hz in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title2)

    px = fig2.add_subplot(311)
    px.set_xticks(time_list)
    px.set_xticklabels(hour)
    plt.setp(px.get_xticklabels(), visible=False)
    px.set_ylabel('Expected ULF main freq. from solar wind [Hz]')
    px.set_ylim(0,(max(ULF_calc)*1.1))
    px.plot(time_list, ULF_calc, plt_clr, linewidth=ln_wth)

    pgx = fig2.add_subplot(312)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(hour)
    plt.setp(pgx.get_xticklabels(), visible=False)
    pgx.set_ylabel('Freq in dBx/dt [Hz]')
    pgx.set_ylim(0,(max(ULF_calc)*1.1))
    pgx.plot(idx, pwr_max[:,0], color=plt_clr)

    pgy = fig2.add_subplot(313, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Freq in dBy/dt [Hz]')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    pgy.set_ylim(0,(max(ULF_calc)*1.1))
    pgy.plot(idx, pwr_max[:,1], color=plt_clr)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/IMF/'

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_ACES_mag.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_ACES_mag.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_ACES_mag.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_ACES_mag.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig2.savefig(png_file, bbox_inches='tight')


def OMNI_plots(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: Same as "ACES_plots" but using OMNI data with better time resolution and no offset

    #NOTE: for the last day in month, there is a 1 hr. UTC offset due to only having 0-23, not 0-24?????????

    #------------------------------------------------------------------------------#
    #Variables and settings

    grab = ['Bx_GSEM', 'By_GSE', 'Bz_GSE']
    dat_arr, ids = OMNI_1m_select(yyyy, mm, dd, grab)

    #cleaning the data (9999.99+ given as placeholder, among others)
    for index, val in np.ndenumerate(dat_arr):
        if val >= 99 and val < 100 or val >= 999 and val < 1000 or val >= 9999 and val < 10000 or val >= 99999 and val < 100000 or val >= 999999 and val < 1000000 or val >= 9999999 and val < 10000000:
            dat_arr[index] = np.nan

    Bx = dat_arr[:,0]
    By = dat_arr[:,1]
    Bz = dat_arr[:,2]

    time_list = np.linspace(0, (864000-1), len(Bx))
    hour = np.arange(0,25,1)

    dt = 1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    Bt = np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz))

    theta = np.zeros(len(Bt))
    for i in range(0,len(Bt)):
        theta[i] = math.degrees(math.acos(Bx[i]/Bt[i]))

    pwr_max, pwr_int = total_power(yyyy, mm, dd, station=station, smoothed=True)
    idx = np.linspace(0, (864000-1), len(pwr_max))

    ULF_calc = 6 * Bt / 1000 #expected ULF main freq in Hz, from eqn: f (mHz) ≈ 6 BIMF (nT) [Takahashi et al., 1984]

    #plotting
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'OMNI Solar Wind B(x,y,z,t) on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    x1 = fig.add_subplot(411)
    x1.set_ylabel('Bx [nT]')
    x1.set_xticks(np.arange(0,864001,(144000/2)/2))
    x1.set_xticklabels(hour)
    plt.setp(x1.get_xticklabels(), visible=False)
    #x1.set_ylim(-b_max,b_max)
    x1.plot(time_list, Bx, plt_clr, linewidth=ln_wth)

    x2 = fig.add_subplot(412, sharex=x1)
    plt.setp(x2.get_xticklabels(), visible=False)
    x2.set_ylabel('By [nT]')
    #x2.set_ylim(-b_max,b_max)
    x2.plot(time_list, By, plt_clr, linewidth=ln_wth)

    x3 = fig.add_subplot(413, sharex=x1)
    plt.setp(x3.get_xticklabels(), visible=False)
    x3.set_ylabel('Bz [nT]')
    #x3.set_ylim(0,(b_max*2))
    x3.plot(time_list, Bz, plt_clr, linewidth=ln_wth)

    x4 = fig.add_subplot(414, sharex=x1)
    plt.setp(x4.get_xticklabels(), visible=True)
    x4.set_ylabel('Bt [nT]')
    x4.set_xlabel('Time [UTC]')
    #x4.set_ylim(-b_max,b_max)
    x4.plot(time_list, Bt, plt_clr, linewidth=ln_wth)


    fig2 = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi)
    ###fig4.tight_layout()
    title2 = 'ULF Max Power Frequency from 0-5 Hz in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title2)

    px = fig2.add_subplot(311)
    px2 = px.twinx()
    px.set_xticks(np.arange(0,864001,(144000/2)/2))
    px.set_xticklabels(hour)
    plt.setp(px.get_xticklabels(), visible=False)
    px.set_ylabel('Expected ULF main freq. from solar wind [Hz]')
    #px.set_ylim(0,(max(ULF_calc)*1.1))
    px.plot(time_list, ULF_calc, plt_clr, linewidth=ln_wth)
    px2.plot(time_list, theta, plt_clr, linewidth=ln_wth, color='r')
    px2.axhline(y=45, linewidth=ln_wth, color='g', linestyle='-')
    px2.set_ylabel('Solar Wind Theta [degrees]')


    pgx = fig2.add_subplot(312)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    pgx.set_xticklabels(hour)
    plt.setp(pgx.get_xticklabels(), visible=False)
    pgx.set_ylabel('Freq in dBx/dt [Hz]')
    #pgx.set_ylim(0,(max(pwr_max)*1.1))
    pgx.plot(idx, pwr_max[:,0], color=plt_clr)

    pgy = fig2.add_subplot(313, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Freq in dBy/dt [Hz]')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    #pgy.set_ylim(0,(max(pwr_max)*1.1))
    pgy.plot(idx, pwr_max[:,1], color=plt_clr)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/IMF/'

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_ACES_mag.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_ACES_mag.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_ACES_mag.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_ACES_mag.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig2.savefig(png_file, bbox_inches='tight')

def OMNI_min_asc2npy(yyyy, mm, dd):

    #DESCRIPTION: This function opens a NASA ACE txt file and saves as a numpy file
    #Returns the full path file name of the specified numpy file

    data_dir = 'C:/Users/Tyler/Desktop/Project Lisbon/Datasets/Geophysical/OMNI/'

    asc_location = data_dir + 'ASC/' + 'omni_min' + str(yyyy) + '.asc'

    #check to make sure the specified text file exits and stops program otherwise
    asc_path = Path(asc_location)
    if asc_path.is_file() is False:
        print('Note: Specified OMNI data file does not exist')
        print(asc_location)
        sys.exit()

    #modifies the string to the correct format given the date
    if mm >= 10 and dd < 10:
        np_file = data_dir + 'NPY/' + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_omni_min.npy'
    elif mm < 10 and dd >= 10:
        np_file = data_dir + 'NPY/' + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_omni_min.npy'
    elif mm < 10 and dd < 10:
        np_file = data_dir + 'NPY/' + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_omni_min.npy'
    else:
        np_file = data_dir + 'NPY/' + str(yyyy) + '_' + str(mm) + '_' + str(dd) +'_omni_min.npy'

    temp = asc_location.replace('.asc', '.npy')
    np_file_full = temp.replace('ASC', 'NPY')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file_full)
    if check_path.is_file():
        print('Note: Already have the numpy file for that OMNI year')
    else:
        datapath = open(asc_location, 'r')
        dat = datapath.readlines()
        datapath.close()

        dataset = []
        for index, item in enumerate(dat,start=0):
            line = dat[index].split()
            line = [float(i) for i in line]
            dataset.append(line)

        omni_full = np.asarray(dataset, dtype='float32') #smaller than float64 type to save memory space
        np.save(np_file_full,omni_full)

    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for that OMNI day')
    else:

        omni_full = np.load(np_file_full)

        DOY = (datetime.date(yyyy, mm, dd) - datetime.date(yyyy,1,1)).days + 1 #calculates the day of the year of date query

        start_idx = 1440*(DOY-1) #starting and ending indexes for minutes of the date, for specified day
        end_idx = 1440*DOY

        omni_dat = omni_full[start_idx:end_idx,:]

        np.save(np_file,omni_dat)

    return np_file

def OMNI_1m_select(yyyy, mm, dd, grab, all=False):

    #DESCRIPTION: This funciton calls OMNI_min_asc2npy to produce a numpy file for a single day
    #from the high res OMNI data (yearly) and returns an array of only the desired components
    #(in the order they were passed); throws error if invalid query

    np_file = OMNI_min_asc2npy(yyyy, mm, dd)
    omni_dat = np.load(np_file)

    omni_dict = {} #create dictionary

    omni_dict['Year'] = omni_dat[:,0]         #(1995 to 2006)
    omni_dict['Day'] = omni_dat[:,1] 	      #(365 or 366)
    omni_dict['Hour'] = omni_dat[:,2] 	      #(0 to 23)
    omni_dict['Minute'] = omni_dat[:,3]       #(0 to 59)
    omni_dict['ID_IMF'] = omni_dat[:,4]       #ID for IMF spacecraft
    omni_dict['ID_SW'] = omni_dat[:,5]        #ID for SW Plasma spacecraft
    omni_dict['N_IMF'] = omni_dat[:,6]        #Number of points in IMF averages
    omni_dict['N_SW'] = omni_dat[:,7]         #Number of points in Plasma averages
    omni_dict['Per_Int'] = omni_dat[:,8]      #Percent interp
    omni_dict['Timeshift'] = omni_dat[:,9]    #[sec]
    omni_dict['RMS_Time'] = omni_dat[:,10]    #Timeshift
    omni_dict['RMS_Phase'] = omni_dat[:,11]   #Phase front normal
    omni_dict['Delta_Time'] = omni_dat[:,12]  #Time btwn observations
    omni_dict['F_Mag_Avg'] = omni_dat[:,13]   #Field magnitude average [nT]
    omni_dict['Bx_GSEM'] = omni_dat[:,14]     #[nT] (GSE and GSM)
    omni_dict['By_GSE'] = omni_dat[:,15]      #[nT] (GSE)
    omni_dict['Bz_GSE'] = omni_dat[:,16]      #[nT] (GSE)
    omni_dict['By_GSM'] = omni_dat[:,17]      #[nT] (GSE)
    omni_dict['Bz_GSM'] = omni_dat[:,18]      #[nT] (GSE)
    omni_dict['RMS_Scalar'] = omni_dat[:,19]  #RMS SD B Scalar [nT]
    omni_dict['RMS_Vector'] = omni_dat[:,20]  #RMS SD field vector [nT]
    omni_dict['Flow_Speed'] = omni_dat[:,21]  #[km/s]
    omni_dict['Vx_GSE'] = omni_dat[:,22]      #Velocity [km/s]
    omni_dict['Vy_GSE'] = omni_dat[:,23]      #Velocity [km/s]
    omni_dict['Vz_GSE'] = omni_dat[:,24] 	  #Velocity [km/s]
    omni_dict['P_Demsity'] = omni_dat[:,25]   #Proton Density [n/cc]
    omni_dict['Temp_K'] = omni_dat[:,26]      #Temperature [K]
    omni_dict['Flow_Pres'] = omni_dat[:,27]   #Flow pressure [nPa]
    omni_dict['E_Field'] = omni_dat[:,28]     #Electric Field [mV/m]
    omni_dict['Plasma_Beta'] = omni_dat[:,29]
    omni_dict['Alfven_Mach'] = omni_dat[:,30] #Mach Number
    omni_dict['X_SC'] = omni_dat[:,31]        #(s/c), GSE, Re
    omni_dict['Y_SC'] = omni_dat[:,32]        #(s/c), GSE, Re
    omni_dict['Z_SC'] = omni_dat[:,33]        #(s/c), GSE, Re
    omni_dict['BSN_X_GSE'] = omni_dat[:,34]   #BSN location, Re
    omni_dict['BSN_Y_GSE'] = omni_dat[:,35]   #BSN location, Re
    omni_dict['BSN_Z_GSE'] = omni_dat[:,36]   #BSN location, Re
    omni_dict['AE_IDX'] = omni_dat[:,37]      #Index [nT]
    omni_dict['AL_IDX'] = omni_dat[:,38]      #Index [nT]
    omni_dict['AU_IDX'] = omni_dat[:,39]      #Index [nT]
    omni_dict['SYM_D_IDX'] = omni_dat[:,40]   #Index [nT]
    omni_dict['SYM_H_IDX'] = omni_dat[:,41]   #Index [nT]
    omni_dict['ASY_D_IDX'] = omni_dat[:,42]   #Index [nT]
    omni_dict['ASY_H_IDX'] = omni_dat[:,43]   #Index [nT]
    omni_dict['PC_N_IDX'] = omni_dat[:,44]    #Index [nT]
    omni_dict['Mag_Mach'] = omni_dat[:,45]    #Magnetosonic mach number

    if all == True:
        grab = omni_dict

    for key in grab:
        if key not in omni_dict :
            err_mes = 'ERROR: invalid input argument (' + key + ')'
            print(err_mes)
            sys.exit()

    omni_out = np.array([], dtype='float32')
    for key in grab:
        temp = omni_dict[key]
        omni_out = np.c_[omni_out, temp] if omni_out.size else temp

    ids = list(omni_dict.keys())

    return omni_out, ids

def OMNI_ULF_plots(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: plots OMNI data for specifically chosen parameters

    #Data taken from: ftp://cdaweb.gsfc.nasa.gov/pub/data/omni/high_res_omni/

    #------------------------------------------------------------------------------#
    #Variables and settings

    grab = ['Bx_GSEM', 'By_GSE', 'Bz_GSE']
    dat_arr, ids = OMNI_1m_select(yyyy, mm, dd, grab)

    #cleaning the data (9999.99+ given as placeholder)
    for index, val in np.ndenumerate(dat_arr):
        if val >= 99 and val <= 100 or val >= 999 and val <= 1000 or val >= 9999 and val <= 10000 or val >= 99999 and val <= 100000 or val >= 999999 and val <= 1000000 or val >= 9999999 and val <= 10000000:
            dat_arr[index] = np.nan

    Bx = dat_arr[:,0]
    By = dat_arr[:,1]
    Bz = dat_arr[:,2]

    time_list = np.linspace(0, (864000-1), len(Bx))
    hour = np.arange(0,25,1)

    dt = 0.1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    pwr_max, pwr_int = total_power(yyyy, mm, dd, station=station, smoothed=True)
    idx = np.linspace(0, (864000-1), len(pwr_max))

    Bt = np.sqrt(np.square(Bx) + np.square(By) + np.square(Bz))
    cone_angle = np.degrees(np.arctan(Bx / np.sqrt(np.square(By) + np.square(Bz)))) #Solar Wind Cone angle: arctan(Bx/(sqrt(By**2 + Bz**2)))
    ULF_calc = 6.1 * Bt / 1000 #expected ULF main freq in Hz, from eqn: f (mHz) ≈ 6 BIMF (nT) [Takahashi et al., 1984]

    for i in range(0,len(Bt)):
        if cone_angle[i] > 45:
            ULF_calc[i] = 0

    #plotting
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'OMNI B(x,y,z,t) on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    x1 = fig.add_subplot(411)
    x1.set_ylabel('Bx [nT]')
    x1.set_xticks(np.arange(0,864001,(144000/2)/2))
    x1.set_xticklabels(hour)
    plt.setp(x1.get_xticklabels(), visible=False)
    #x1.set_ylim(-b_max,b_max)
    x1.plot(time_list, Bx, plt_clr, linewidth=ln_wth)

    x2 = fig.add_subplot(412, sharex=x1)
    plt.setp(x2.get_xticklabels(), visible=False)
    x2.set_ylabel('By [nt]')
    #x2.set_ylim(-b_max,b_max)
    x2.plot(time_list, By, plt_clr, linewidth=ln_wth)

    x3 = fig.add_subplot(413, sharex=x1)
    plt.setp(x3.get_xticklabels(), visible=False)
    x3.set_ylabel('Bz [nT]')
    #x3.set_ylim(0,(b_max*2))
    x3.plot(time_list, Bz, plt_clr, linewidth=ln_wth)

    x4 = fig.add_subplot(414, sharex=x1)
    plt.setp(x4.get_xticklabels(), visible=True)
    x4.set_ylabel('Bt [nT]')
    x4.set_xlabel('Time [UTC]')
    #x4.set_ylim(-b_max,b_max)
    x4.plot(time_list, Bt, plt_clr, linewidth=ln_wth)


    fig2 = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi)
    ###fig4.tight_layout()
    title2 = 'ULF Max Power Frequency from 0-5 Hz in ' + station + ' ULF dB/dt data on ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title2)

    px = fig2.add_subplot(311)
    px.set_xticks(np.arange(0,864001,(144000/2)/2))
    px.set_xticklabels(hour)
    plt.setp(px.get_xticklabels(), visible=False)
    px.set_ylabel('Expected ULF main freq. from solar wind [Hz]')
    #px.set_ylim(0,(max(ULF_calc)*1.1))
    px.set_ylim(0,0.1)
    px.plot(time_list, ULF_calc, plt_clr, linewidth=ln_wth)

    pgx = fig2.add_subplot(312)
    pgx.set_xticks(np.arange(0,864001,(144000/2)/2))
    #pgx.set_xticks(np.arange(0,(len(idx)+1),n_pts)) #this does not work an I'm baffeled as to why not...while the above does!?
    pgx.set_xticklabels(hour)
    plt.setp(pgx.get_xticklabels(), visible=False)
    pgx.set_ylabel('Freq in dBx/dt [Hz]')
    pgx.set_ylim(0,0.1)
    pgx.plot(idx, pwr_max[:,0], color=plt_clr, linewidth=ln_wth)

    pgy = fig2.add_subplot(313, sharex=pgx)
    plt.setp(pgy.get_xticklabels(), visible=True)
    pgy.set_ylabel('Freq in dBy/dt [Hz]')
    pgy.set_xlabel('Time [UTC]', fontsize=10)
    pgy.set_ylim(0,0.1)
    pgy.plot(idx, pwr_max[:,1], color=plt_clr, linewidth=ln_wth)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/IMF/'

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_OMNI_mag.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_OMNI_mag.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_OMNI_mag.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_OMNI_mag.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig2.savefig(png_file, bbox_inches='tight')

def OMNI_all_plots(yyyy, mm, dd, showplot=False, saveplot=False, save_dir='none'):

    #DESCRIPTION: plots all OMNI data for a given date

    #Data taken from: ftp://cdaweb.gsfc.nasa.gov/pub/data/omni/high_res_omni/

    #------------------------------------------------------------------------------#
    #Variables and settings

    grab = []
    dat_arr, ids = OMNI_1m_select(yyyy, mm, dd, grab, all=True)

    #cleaning the data (9999.99+ given as placeholder, among others)
    for index, val in np.ndenumerate(dat_arr):
        if val >= 99 and val <= 100 or val >= 999 and val <= 1000 or val >= 9999 and val <= 10000 or val >= 99999 and val <= 100000 or val >= 999999 and val <= 1000000 or val >= 9999999 and val <= 10000000:
            dat_arr[index] = np.nan

    time = dat_arr[:,0]
    time_list = np.linspace(0, (864000-1), len(time))
    hour = np.arange(0,25,1)

    dt = 0.1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    #plotting

    for index, val in enumerate(ids):
        fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
        title = 'OMNI ' + val + ' data for ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
        plt.suptitle(title)

        plt_clr = '#000000' #blue
        ln_wth= 0.5 #plot linewidth

        x1 = fig.add_subplot(111)
        x1.set_ylabel(val)
        x1.set_xlabel('Time [UTC]')
        x1.set_xticks(np.arange(0,864001,(144000/2)/2))
        x1.set_xticklabels(hour)
        plt.setp(x1.get_xticklabels(), visible=True)
        x1.plot(time_list, dat_arr[:,index], plt_clr, linewidth=ln_wth)

        if save_dir == 'none':
            png_dir = 'C:/Users/Tyler/Desktop/OMNI/'
        else:
            png_dir = save_dir + '/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + ids[index] + '_' + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_OMNI.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + ids[index] + '_' + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_OMNI.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + ids[index] + '_' + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_OMNI.png'
        else:
            png_file = png_dir + ids[index] + '_' + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_OMNI.png'

        #------------------------------------------------------------------------------#
        #saving and displaying plots

        if showplot == True:
            plt.show()

        if saveplot == True:
            fig.savefig(png_file, bbox_inches='tight')

def OMNI_save_dir(yyyy, mm, dd):

    #NOTE: currently an error relating to the file path check...different than usual

    data_dir = 'C:/Users/Tyler/Desktop/OMNI/'

    #modifies the string to the correct format given the date
    if mm >= 10 and dd < 10:
        file_location = data_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd)
    elif mm < 10 and dd >= 10:
        file_location = data_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd)
    elif mm < 10 and dd < 10:
        file_location = data_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd)
    else:
        file_location = data_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd)

    #check to make sure the specified text file exits and stops program otherwise
    file_path = Path(file_location)
    if os.path.isfile(file_path) is False:
        os.mkdir(file_location) #creates directory/folder
    else:
        print('Note: Specified OMNI directory slready exists')
        print(file_location)
        sys.exit()

    OMNI_all_plots(yyyy, mm, dd, showplot=False, saveplot=True, save_dir=file_location)


def SuperMag_txt2npy(yyyy, mm, dd):

    #DESCRIPTION: This function opens a SuperMag txt file and saves a cleaned and ordered npy file
    #Returns the full path file name of the specified numpy file

    #NOTE: Only configured for a SuperMag text file containing 5 stations (nominally BJN, HOR, HRN, LYR, NAL)
    #NOTE: SuperMag data for leap years has same number of data points, since it doesn't contain Dec. 31st

    def is_float(n):
        try:
            float(n)
            return True
        except:
            return False

    data_dir = 'C:/Users/Tyler/Desktop/Project Lisbon/Datasets/Geophysical/SuperMag/'
    txt_location = data_dir + str(yyyy) + '_SuperMag.txt'

    #check to make sure the specified text file exits and stops program otherwise
    txt_path = Path(txt_location)
    if txt_path.is_file() is False:
        print('Note: Specified SuperMag data file does not exist')
        print(txt_location)
        sys.exit()

    np_file = txt_location.replace('.txt', '.npy')

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(np_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for ' + str(yyyy))
    else:
        datapath = open(txt_location, 'r')
        dat = datapath.readlines()
        datapath.close()
        dat = dat[77:] #removes the header

        #insert missing station values into dat as NaNs
        check_arr = [str(yyyy), 'BJN', 'HOR', 'HRN', 'LYR', 'NAL']
        idx = 0
        dataset1 = []
        dataset2 = []
        for index, item in enumerate(dat,start=0):
            check_str = dat[index].split()[0]
            if check_str != check_arr[idx]:
                while check_str != check_arr[idx]:
                    line_x = ['999', '999', '999', '999', '999', '999', '999']
                    line_x = [float(i) for i in line_x]
                    dataset1.append(line_x) if index <= 1600000 else dataset2.append(line_x)
                    idx = idx+1
                    if idx == len(check_arr): idx = 0
                idx = idx+1
                if idx == len(check_arr): idx = 0
            else:
                idx = idx+1
                if idx == len(check_arr): idx = 0

            line = dat[index].split()
            line = [float(i) for i in line if is_float(i)]
            dataset1.append(line) if index <= 1600000 else dataset2.append(line)
            if index == (len(dat)-1): #to fill in the rest of the sequence for the final lines
                for i in range ((idx),len(check_arr)):
                    line_x = ['999', '999', '999', '999', '999', '999', '999']
                    line_x = [float(i) for i in line_x]
                    dataset1.append(line_x) if index <= 1600000 else dataset2.append(line_x)

        #the split between sets avoids memory error when appending
        dataset_1 = np.asarray(dataset1, dtype='float16') #smaller than float64 type to save memory space
        dataset_2 = np.asarray(dataset2, dtype='float16') #smaller than float64 type to save memory space
        SuperMag_dat = np.concatenate((dataset_1,dataset_2), axis=0) #smaller than float64 type to save memory space

        #below the data is reformatted, where each line is one minute of time
        row, col = np.shape(SuperMag_dat)
        ordered_dat = np.array([int(row/6), 6, col], dtype='float16') #creates an empty numpy array
        ordered_dat = np.transpose(SuperMag_dat.reshape(int(row/6), 6, col), axes=(0,1,2)) #reshapes array to allow for easier access
        #This makes each row a single minute of data, with the first colum holding ,an array of,, time info and the remaining columns each holing one array of mag data from each station

        np.save(np_file,ordered_dat,allow_pickle=False)

    #modifies the string to the correct format given the date
    if mm >= 10 and dd < 10: date_file = data_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_SuperMag.npy'
    elif mm < 10 and dd >= 10: date_file = data_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_SuperMag.npy'
    elif mm < 10 and dd < 10: date_file = data_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_SuperMag.npy'
    else: date_file = data_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) +'_SuperMag.npy'

    #check if numpy file for the specified date already exists, creates a new one if not
    check_path = Path(date_file)
    if check_path.is_file():
        print('Note: Already have the numpy file for ' + str(yyyy) + '/' + str(mm) + '/' + str(dd))
    else:
        SuperMag_full = np.load(np_file)
        DOY = (datetime.date(yyyy, mm, dd) - datetime.date(yyyy,1,1)).days + 1 #calculates the day of the year of date query

        start_idx = 1440*(DOY-1) #starting and ending indexes for minutes of the date, for specified day
        end_idx = 1440*DOY

        SuperMag_dat = SuperMag_full[start_idx:end_idx,:,:]
        np.save(date_file,SuperMag_dat)

    return date_file

def SuperMag_plots(yyyy, mm, dd, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: plots SuperMag data

    #------------------------------------------------------------------------------#
    #Variables and settings

    dat_dir = SuperMag_txt2npy(yyyy, mm, dd)
    dat_arr = np.load(dat_dir)

    #cleaning the data (999 given as placeholder)
    for index, val in np.ndenumerate(dat_arr):
        if val == 999:
            dat_arr[index] = np.nan

    shape = np.shape(dat_arr)
    time_list = np.linspace(0, 24, shape[0])
    hour = np.arange(0,25,1)

    dt = 0.1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    colors = cycle(["red", "black", "blue", "gray", "green"])

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    #plotting

    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'SuperMag data for ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    ax1 = fig.add_subplot(311)
    ax1.set_ylabel('Bx (North) [nT]')
    ax1.set_xticks(hour)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    plt.setp(ax1.get_xticklabels(which='both'), visible=False)
    ax1.axhline(linewidth=0.5, color='fuchsia')

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.set_ylabel('By (East) [nT]')
    plt.setp(ax2.get_xticklabels(which='both'), visible=False)
    ax2.axhline(linewidth=0.5, color='fuchsia')


    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.set_ylabel('Bz (Verticle) [nT]')
    plt.setp(ax3.get_xticklabels(which='major'), visible=True)
    plt.setp(ax3.get_xticklabels(which='minor'), visible=False)
    ax3.set_xlabel('Time [UTC]')
    ax3.axhline(linewidth=0.5, color='fuchsia')

    stations = ['BJN', 'HOR', 'HRN', 'LYR', 'NAL']

    for i, item in enumerate(stations):
        plt_color = next(colors)
        if smoothed == True:
            Bx_smooth = savgol_filter(dat_arr[:,(i+1),0].flatten(order='C'), 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
            ax1.plot(time_list, Bx_smooth, label=str(item), color=plt_color, linewidth=ln_wth)
            By_smooth = savgol_filter(dat_arr[:,(i+1),1].flatten(order='C'), 11, 0)
            ax2.plot(time_list, By_smooth, label=str(item), color=plt_color, linewidth=ln_wth)
            Bz_smooth = savgol_filter(dat_arr[:,(i+1),2].flatten(order='C'), 11, 0)
            ax3.plot(time_list, Bz_smooth, label=str(item), color=plt_color, linewidth=ln_wth)
        else:
            ax1.plot(time_list, dat_arr[:,(i+1),0], label=str(item), color=plt_color, linewidth=ln_wth)
            ax2.plot(time_list, dat_arr[:,(i+1),1], label=str(item), color=plt_color, linewidth=ln_wth)
            ax3.plot(time_list, dat_arr[:,(i+1),2], label=str(item), color=plt_color, linewidth=ln_wth)

    ax1.legend(loc="upper right", fontsize=6)
    ax2.legend(loc="upper right", fontsize=6)
    ax3.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    if showplot == True:
        plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/SuperMag/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_SuperMag.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_SuperMag.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_SuperMag.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_SuperMag.png'

        fig.savefig(png_file, bbox_inches='tight')

def SuperMag_spec(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: plots SuperMag data

    #------------------------------------------------------------------------------#
    #Variables and settings

    dat_dir = SuperMag_txt2npy(yyyy, mm, dd)
    dat_arr = np.load(dat_dir)

    #cleaning the data (999 given as placeholder)
    for index, val in np.ndenumerate(dat_arr):
        if val == 999:
            dat_arr[index] = np.nan

    shape = np.shape(dat_arr)
    time_list = np.linspace(0, 24, shape[0])
    hour = np.arange(0,25,1)

    dt = 5 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**3 #3=8, 4=16, 5=32, 6=64, 7=128, 8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    colors = cycle(["red", "black", "blue", "gray", "green"])

    f_max = 0.1
    plotcolor = '#000000' #black
    ln_wth = 0.5 #plot line width
    cmap=cm.jet #sets the colormap spectrum @https://matplotlib.org/users/colormaps.html

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    detrend = 'default' #function applied before fft
    scale = 'dB' #scaling of calcuated values
    mode = 'psd' #spectrum mode (also has magnitude, angle, and phase)

    stations = ['BJN', 'HOR', 'HRN', 'LYR', 'NAL']
    idx = 999
    for index, item in enumerate(stations,start=0):
        if item == station: idx = index
    if idx == 999: print('Note: Given station does not appear in the SuperMag collection available.')

    #plotting

    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'SuperMag data from ' + station + ' for '+ str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    ax1 = fig.add_subplot(311)
    pxx, freqx, tx, cax = pylab.specgram(dat_arr[:,(idx+1),0].flatten(order='C'), nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    ax1.set_ylabel('Bx (North)')
    ax1.set_xticks(hour)
    #ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    plt.setp(ax1.get_xticklabels(which='both'), visible=False)
    ax1.axhline(linewidth=0.5, color='fuchsia')
    ax1.set_ylim(0,f_max)

    ax2 = fig.add_subplot(312, sharex=ax1)
    pxy, freqy, ty, cay = pylab.specgram(dat_arr[:,(idx+1),1].flatten(order='C'), nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    ax2.set_ylabel('By (East)')
    plt.setp(ax2.get_xticklabels(which='both'), visible=False)
    ax2.axhline(linewidth=0.5, color='fuchsia')
    ax2.set_ylim(0,f_max)

    ax3 = fig.add_subplot(313, sharex=ax1)
    pxz, freqz, tz, caz = pylab.specgram(dat_arr[:,(idx+1),2].flatten(order='C'), nfft, fs, noverlap = nov, vmin=-50, vmax=5, cmap=cmap, detrend=detrend, scale=scale, mode=mode)
    ax3.set_ylabel('Bz (Verticle)')
    plt.setp(ax3.get_xticklabels(which='major'), visible=True)
    plt.setp(ax3.get_xticklabels(which='minor'), visible=False)
    ax3.set_xlabel('Time [UTC]')
    ax3.axhline(linewidth=0.5, color='fuchsia')
    ax3.set_ylim(0,f_max)

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/SuperMag_spec/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_SuperMag_spec.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_SuperMag_spec.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_SuperMag_spec.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_SuperMag_spec.png'

        fig.savefig(png_file, bbox_inches='tight')

    fig2 = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title2 = 'SuperMag data from ' + station + ' for '+ str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    bx1 = fig2.add_subplot(311)
    bx1.set_ylabel('Bx (North) Power')
    plt.setp(bx1.get_xticklabels(which='both'), visible=False)

    bx2 = fig2.add_subplot(312, sharex=bx1)
    bx2.set_ylabel('By (East) Power')
    plt.setp(bx2.get_xticklabels(which='both'), visible=False)

    bx3 = fig2.add_subplot(313, sharex=bx1)
    bx3.set_ylabel('Bz (Verticle) Power')
    plt.setp(bx3.get_xticklabels(which='major'), visible=True)
    bx3.set_xlabel('Frequency [Hz]')

    stations = ['BJN', 'HOR', 'HRN', 'LYR', 'NAL']

    for i, item in enumerate(stations):
        plt_color = next(colors)
        N = 10e0
        fx, pxt = signal.periodogram(dat_arr[:,(i+1),0],N)
        fy, pyt = signal.periodogram(dat_arr[:,(i+1),1],N)
        fz, pzt = signal.periodogram(dat_arr[:,(i+1),0],N)
        #pxxt_s = savgol_filter(pxxt, 11, 0)
        #pxyt_s = savgol_filter(pxyt, 11, 0)
        bx1.semilogx(fx,pxt, label=item, linewidth=0.5)
        bx2.semilogx(fy,pyt, label=item, linewidth=0.5)
        bx3.semilogx(fz,pzt, label=item, linewidth=0.5)

    bx1.legend(loc='upper right')

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    if showplot == True:
        plt.show()

def SuperMag_grad(yyyy, mm, dd, showplot=False, saveplot=False, smoothed=False):

    #DESCRIPTION: plots SuperMag data

    #------------------------------------------------------------------------------#
    #Variables and settings

    dat_dir = SuperMag_txt2npy(yyyy, mm, dd)
    dat_arr = np.load(dat_dir)

    #cleaning the data (999 given as placeholder)
    for index, val in np.ndenumerate(dat_arr):
        if val == 999:
            dat_arr[index] = np.nan

    shape = np.shape(dat_arr)
    time_list = np.linspace(0, 24, shape[0])
    hour = np.arange(0,25,1)

    dt = 0.1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    colors = cycle(["red", "black", "blue", "gray", "green"])

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    #plotting

    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'SuperMag data gradient for ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    ax1 = fig.add_subplot(311)
    ax1.set_ylabel('Bx (North) [nT]')
    ax1.set_xticks(hour)
    ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    plt.setp(ax1.get_xticklabels(which='both'), visible=False)
    ax1.axhline(linewidth=0.5, color='fuchsia')

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.set_ylabel('By (East) [nT]')
    plt.setp(ax2.get_xticklabels(which='both'), visible=False)
    ax2.axhline(linewidth=0.5, color='fuchsia')


    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.set_ylabel('Bz (Verticle) [nT]')
    plt.setp(ax3.get_xticklabels(which='major'), visible=True)
    plt.setp(ax3.get_xticklabels(which='minor'), visible=False)
    ax3.set_xlabel('Time [UTC]')
    ax3.axhline(linewidth=0.5, color='fuchsia')

    stations = ['BJN', 'HOR', 'HRN', 'LYR', 'NAL']
    #n_pts = 864 #number of integration iteration, equal to resolution of 100 sec
    idx = np.linspace(0, 1440, 1440,  dtype=int)


    for i, item in enumerate(stations):
        plt_color = next(colors)
        if smoothed == True:
            x_grad = np.asarray(np.gradient(dat_arr[:,(i+1),0].flatten(order='C'), idx))
            x_smooth = savgol_filter(x_grad, 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
            ax1.plot(time_list, x_smooth, label=str(item), color=plt_color, linewidth=ln_wth)
            y_grad = np.asarray(np.gradient(dat_arr[:,(i+1),1].flatten(order='C'), idx))
            y_smooth = savgol_filter(y_grad, 11, 0)
            ax2.plot(time_list, y_smooth, label=str(item), color=plt_color, linewidth=ln_wth)
            z_grad = np.asarray(np.gradient(dat_arr[:,(i+1),2].flatten(order='C'), idx))
            z_smooth = savgol_filter(z_grad, 11, 0)
            ax3.plot(time_list, z_smooth, label=str(item), color=plt_color, linewidth=ln_wth)
        else:
            x_grad = np.asarray(np.gradient(dat_arr[:,(i+1),0].flatten(order='C'), idx))
            ax1.plot(time_list, dat_arr[:,(i+1),0], label=str(item), color=plt_color, linewidth=ln_wth)
            y_grad = np.asarray(np.gradient(dat_arr[:,(i+1),1].flatten(order='C'), idx))
            ax2.plot(time_list, dat_arr[:,(i+1),1], label=str(item), color=plt_color, linewidth=ln_wth)
            z_grad = np.asarray(np.gradient(dat_arr[:,(i+1),2].flatten(order='C'), idx))
            ax3.plot(time_list, dat_arr[:,(i+1),2], label=str(item), color=plt_color, linewidth=ln_wth)

    ax1.legend(loc="upper right", fontsize=6)
    ax2.legend(loc="upper right", fontsize=6)
    ax3.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    if showplot == True:
        plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/SuperMag_grad/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_SuperMag_grad.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_SuperMag_grad.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_SuperMag_grad.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_SuperMag_grad.png'

        fig.savefig(png_file, bbox_inches='tight')


def ULF_combo_plots(yyyy, mm, dd, choose='both', limit_same=False, showplot=False, saveplot=False, smoothed=False, sample_sec=0.1):

    #------------------------------------------------------------------------------#
    #Variables and settings

    dat_dir = SuperMag_txt2npy(yyyy, mm, dd)
    SuperMag_arr = np.load(dat_dir)

    #cleaning the data (999 given as placeholder)
    for index, val in np.ndenumerate(SuperMag_arr):
        if val == 999:
            SuperMag_arr[index] = np.nan

    hour = np.arange(0,25,1)

    dt = 0.1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    colors = cycle(["red", "black", "blue", "gray", "green"])

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    stations = ['BJN', 'HOR', 'HRN', 'LYR', 'NAL']

    row, col, depth = np.shape(SuperMag_arr)

    SM_pol_x = np.zeros((row, col))
    SM_pol_y = np.zeros((row, col))
    SM_pol_z = np.zeros((row, col))
    ULF_pol_x = np.zeros((int(864000/(sample_sec*10)), col))
    ULF_pol_y = np.zeros((int(864000/(sample_sec*10)), col))
    ULF_pol_z = np.zeros((int(864000/(sample_sec*10)), col))

    for i, item in enumerate(stations, start=0):
        #NOTE: only doing x and y component for three component system here
        x_pol = np.abs(SuperMag_arr[:,(i+1),0])/np.sqrt(np.add(np.square(SuperMag_arr[:,(i+1),0]), np.square(SuperMag_arr[:,(i+1),1])))
        y_pol = np.abs(SuperMag_arr[:,(i+1),1])/np.sqrt(np.add(np.square(SuperMag_arr[:,(i+1),0]), np.square(SuperMag_arr[:,(i+1),1])))
        z_pol = np.abs(SuperMag_arr[:,(i+1),2])/np.sqrt(np.add(np.square(SuperMag_arr[:,(i+1),0]), np.square(SuperMag_arr[:,(i+1),1])))
        SM_pol_x[:,i] = x_pol
        SM_pol_y[:,i] = y_pol
        SM_pol_z[:,i] = z_pol

        if item == 'LYR' or item == 'HOR':
            np_file = ULF_txt2npy_sampled(yyyy, mm, dd, station=item, sample_sec=sample_sec)
            ULF_arr = np.load(np_file)
            x_mag = ULF_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
            y_mag = ULF_arr[:,2] #y-axis signal

            x_pol = np.abs(x_mag)/np.sqrt(np.add(np.square(x_mag), np.square(y_mag)))
            y_pol = np.abs(y_mag)/np.sqrt(np.add(np.square(x_mag), np.square(y_mag)))
            ULF_pol_x[:,i] = x_pol
            ULF_pol_y[:,i] = y_pol

    ULF_time = np.linspace(0, 24, len(ULF_arr[:,0]))
    SM_time = np.linspace(0, 24, np.shape(SuperMag_arr)[0])

    for index, val in np.ndenumerate(SM_pol_x):
        if val == 0: SM_pol_x[index] = np.nan
    for index, val in np.ndenumerate(ULF_pol_x):
        if val == 0: ULF_pol_x[index] = np.nan
    for index, val in np.ndenumerate(SM_pol_y):
        if val == 0: SM_pol_y[index] = np.nan
    for index, val in np.ndenumerate(ULF_pol_y):
        if val == 0: ULF_pol_y[index] = np.nan
    for index, val in np.ndenumerate(SM_pol_z):
        if val == 0: SM_pol_z[index] = np.nan
    for index, val in np.ndenumerate(ULF_pol_z):
        if val == 0: ULF_pol_z[index] = np.nan

    #plotting

    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'Polarization ratios for ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    ax1 = fig.add_subplot(311)
    ax1.set_ylabel('Bx (North) ratio')
    ax1.set_xticks(hour)
    #ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    plt.setp(ax1.get_xticklabels(which='both'), visible=False)
    ax1.axhline(linewidth=0.5, color='fuchsia')

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.set_ylabel('By (East) ratio')
    plt.setp(ax2.get_xticklabels(which='both'), visible=False)
    ax2.axhline(linewidth=0.5, color='fuchsia')

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.set_ylabel('Bz (Verticle) ratio')
    plt.setp(ax3.get_xticklabels(which='major'), visible=True)
    plt.setp(ax3.get_xticklabels(which='minor'), visible=False)
    ax3.set_xlabel('Time [UTC]')
    ax3.axhline(linewidth=0.5, color='fuchsia')
    ax3.set_xlim(0,24)

    if limit_same == True: #eliminates data from SuperMag array if not present for ULF station
        for i, item in enumerate(stations,start=0):
            if all(np.isnan(ULF_pol_x[:,i])) == True:
                SM_pol_x[:,i] = np.nan
                SM_pol_y[:,i] = np.nan
                SM_pol_z[:,i] = np.nan
            plt_color = next(colors)
            if smoothed == True:
                if choose == 'both' or choose == 'SuperMag':
                    SM_x_smooth = savgol_filter(SM_pol_x[:,i], 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
                    SM_y_smooth = savgol_filter(SM_pol_y[:,i], 11, 0)
                    SM_z_smooth = savgol_filter(SM_pol_z[:,i], 11, 0)
                    ax1.plot(SM_time, SM_x_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax2.plot(SM_time, SM_y_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax3.plot(SM_time, SM_z_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                if choose == 'both' or choose == 'ULF':
                    ULF_x_smooth = savgol_filter(ULF_pol_x[:,i], 11, 0)
                    ULF_y_smooth = savgol_filter(ULF_pol_y[:,i], 11, 0)
                    ax1.plot(ULF_time, ULF_x_smooth, label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6,2])
                    ax2.plot(ULF_time, ULF_y_smooth, label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6,2])
                    ax3.plot(ULF_time, ULF_pol_z[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6,2])
            else:
                if choose == 'both' or choose == 'SuperMag':
                    ax1.plot(SM_time, SM_pol_x[:,i], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax2.plot(SM_time, SM_pol_y[:,i], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax3.plot(SM_time, SM_pol_z[:,i], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                if choose == 'both' or choose == 'ULF':
                    ax1.plot(ULF_time, ULF_pol_x[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth) #dashes=[6, 2]
                    ax2.plot(ULF_time, ULF_pol_y[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth)
                    ax3.plot(ULF_time, ULF_pol_z[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth)

    else:
        for i, item in enumerate(stations,start=0):
            plt_color = next(colors)
            if smoothed == True:
                if choose == 'both' or choose == 'SuperMag':
                    SM_x_smooth = savgol_filter(SM_pol_x[:,i], 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
                    SM_y_smooth = savgol_filter(SM_pol_y[:,i], 11, 0)
                    SM_z_smooth = savgol_filter(SM_pol_z[:,i], 11, 0)
                    ax1.plot(SM_time, SM_x_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax2.plot(SM_time, SM_y_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax3.plot(SM_time, SM_z_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                if choose == 'both' or choose == 'ULF':
                    ULF_x_smooth = savgol_filter(ULF_pol_x[:,i], 11, 0)
                    ULF_y_smooth = savgol_filter(ULF_pol_y[:,i], 11, 0)
                    ax1.plot(ULF_time, ULF_x_smooth, label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6,2])
                    ax2.plot(ULF_time, ULF_y_smooth, label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6,2])
                    ax3.plot(ULF_time, ULF_pol_z[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6,2])
            else:
                if choose == 'both' or choose == 'SuperMag':
                    ax1.plot(SM_time, SM_pol_x[:,i], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax2.plot(SM_time, SM_pol_y[:,i], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                    ax3.plot(SM_time, SM_pol_z[:,i], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                if choose == 'both' or choose == 'ULF':
                    ax1.plot(ULF_time, ULF_pol_x[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth) #dashes=[6, 2]
                    ax2.plot(ULF_time, ULF_pol_y[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth)
                    ax3.plot(ULF_time, ULF_pol_z[:,i], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth)

    ax1.legend(loc="upper right", fontsize=6)
    ax2.legend(loc="upper right", fontsize=6)
    ax3.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    if showplot == True:
        plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/Polarization/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_mag_pol.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_mag_pol.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_mag_pol.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_mag_pol.png'

        fig.savefig(png_file, bbox_inches='tight')

def ULF_combo_grad(yyyy, mm, dd, choose='both', showplot=False, saveplot=False, smoothed=False):

    #------------------------------------------------------------------------------#
    #Variables and settings

    dat_dir = SuperMag_txt2npy(yyyy, mm, dd)
    SuperMag_arr = np.load(dat_dir)

    #cleaning the data (999 given as placeholder)
    for index, val in np.ndenumerate(SuperMag_arr):
        if val == 999:
            SuperMag_arr[index] = np.nan

    hour = np.arange(0,25,1)

    dt = 0.1 #timestep [sec]
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi = 120

    colors = cycle(["red", "black", "blue", "gray", "green"])

    #------------------------------------------------------------------------------#
    #Main

    #calculations

    stations = ['BJN', 'HOR', 'HRN', 'LYR', 'NAL']

    row, col, depth = np.shape(SuperMag_arr)

    SM_pol_x = np.zeros((row, col))
    SM_pol_y = np.zeros((row, col))
    SM_pol_z = np.zeros((row, col))
    ULF_pol_x = np.zeros((864000, col))
    ULF_pol_y = np.zeros((864000, col))
    ULF_pol_z = np.zeros((864000, col))

    grad_idx = np.linspace(0, 24, np.shape(SuperMag_arr)[0])

    for i, item in enumerate(stations, start=0):
        #NOTE: only doing x and y component for three component system here
        SM_pol_x[:,i] = np.gradient(SuperMag_arr[:,(i+1),0],grad_idx)
        SM_pol_y[:,i] = np.gradient(SuperMag_arr[:,(i+1),1],grad_idx)
        SM_pol_z[:,i] = np.gradient(SuperMag_arr[:,(i+1),2],grad_idx)

        if item == 'LYR' or item == 'HOR':
            np_file = ULF_txt2npy(yyyy, mm, dd, station=item)
            ULF_arr = np.load(np_file)
            ULF_pol_x[:,i] = ULF_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
            ULF_pol_y[:,i] = ULF_arr[:,2] #y-axis signal

    ULF_time = np.linspace(0, 24, len(ULF_arr[:,0]))
    SM_time = np.linspace(0, 24, np.shape(SuperMag_arr)[0])

    for index, val in np.ndenumerate(SM_pol_x):
        if val == 0: SM_pol_x[index] = np.nan
    for index, val in np.ndenumerate(ULF_pol_x):
        if val == 0: ULF_pol_x[index] = np.nan
    for index, val in np.ndenumerate(SM_pol_y):
        if val == 0: SM_pol_y[index] = np.nan
    for index, val in np.ndenumerate(ULF_pol_y):
        if val == 0: ULF_pol_y[index] = np.nan
    for index, val in np.ndenumerate(SM_pol_z):
        if val == 0: SM_pol_z[index] = np.nan
    for index, val in np.ndenumerate(ULF_pol_z):
        if val == 0: ULF_pol_z[index] = np.nan

    #plotting

    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'Polarization ratios for ' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth

    ax1 = fig.add_subplot(311)
    ax1.set_ylabel('Bx (North) ratio')
    ax1.set_xticks(hour)
    #ax1.xaxis.set_minor_locator(AutoMinorLocator(4))
    plt.setp(ax1.get_xticklabels(which='both'), visible=False)
    ax1.axhline(linewidth=0.5, color='fuchsia')

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.set_ylabel('By (East) ratio')
    plt.setp(ax2.get_xticklabels(which='both'), visible=False)
    ax2.axhline(linewidth=0.5, color='fuchsia')

    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.set_ylabel('Bz (Verticle) ratio')
    plt.setp(ax3.get_xticklabels(which='major'), visible=True)
    plt.setp(ax3.get_xticklabels(which='minor'), visible=False)
    ax3.set_xlabel('Time [UTC]')
    ax3.axhline(linewidth=0.5, color='fuchsia')

    for i, item in enumerate(stations,start=0):
        plt_color = next(colors)
        if smoothed == True:
            if choose == 'both' or choose == 'SuperMag':
                SM_x_smooth = savgol_filter(SM_pol_x[:,(i+1)], 11, 0) #smoothing out the periodogram (window size 1, polynomial order 0)
                SM_y_smooth = savgol_filter(SM_pol_y[:,(i+1)], 11, 0)
                SM_z_smooth = savgol_filter(SM_pol_z[:,(i+1)], 11, 0)
                ax1.plot(SM_time, SM_x_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                ax2.plot(SM_time, SM_y_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                ax3.plot(SM_time, SM_z_smooth, label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
            if choose == 'both' or choose == 'ULF':
                ULF_x_smooth = savgol_filter(ULF_pol_x[:,(i+1)], 11, 0)
                ULF_y_smooth = savgol_filter(ULF_pol_y[:,(i+1)], 11, 0)
                ax1.plot(ULF_time, ULF_x_smooth, label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6, 2])
                ax2.plot(ULF_time, ULF_y_smooth, label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6, 2])
                ax3.plot(ULF_time, ULF_pol_z[:,(i+1)], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6, 2])
        else:
            if choose == 'both' or choose == 'SuperMag':
                ax1.plot(SM_time, SM_pol_x[:,(i+1)], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                ax2.plot(SM_time, SM_pol_y[:,(i+1)], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
                ax3.plot(SM_time, SM_pol_z[:,(i+1)], label=str(item)+' SuperMag', color=plt_color, linewidth=ln_wth)
            if choose == 'both' or choose == 'ULF':
                ax1.plot(ULF_time, ULF_pol_x[:,(i+1)], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6, 2])
                ax2.plot(ULF_time, ULF_pol_y[:,(i+1)], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6, 2])
                ax3.plot(ULF_time, ULF_pol_z[:,(i+1)], label=str(item)+' ULF', color=plt_color, linewidth=ln_wth, dashes=[6, 2])

    ax1.legend(loc="upper right", fontsize=6)
    ax2.legend(loc="upper right", fontsize=6)
    ax3.legend(loc="upper right", fontsize=6)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    if showplot == True:
        plt.show()

    if saveplot == True:
        png_dir = 'C:/Users/Tyler/Desktop/Polarization/'

        if mm >= 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_mag_pol.png'
        elif mm < 10 and dd >= 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_mag_pol.png'
        elif mm < 10 and dd < 10:
            png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_mag_pol.png'
        else:
            png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_mag_pol.png'

        fig.savefig(png_file, bbox_inches='tight')


def Run_Wrapper(yyyy_start, mm_start, dd_start, yyyy_end, mm_end, dd_end, function, *args, **kwargs):

    #DEFINITION: A wrapper function to run a range of any function input (daily cycle, not hourly)

    for year in range(yyyy_start,(yyyy_end+1)):
        if year == yyyy_start and year == yyyy_end:
            for month in range(mm_start,(mm_end+1)):
                if month == mm_end:
                    for day in range(dd_start,(dd_end+1)):
                        function(year, month, day, *args, **kwargs)
                        plt.close('all')
                else:
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        function(year, month, day, *args, **kwargs)
                        plt.close('all')
        if year == yyyy_start:
            for month in range(mm_start,(12+1)):
                if month == mm_start:
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(dd_start,(day_range+1)):
                        function(year, month, day, *args, **kwargs)
                        plt.close('all')
                else:
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        function(year, month, day, *args, **kwargs)
                        plt.close('all')
        if year == yyyy_end:
            for month in range(1,(mm_end+1)):
                if month == mm_end:
                    for day in range(1,(dd_end+1)):
                        function(year, month, day, *args, **kwargs)
                        plt.close('all')
                else:
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        function(year, month, day, *args, **kwargs)
                        plt.close('all')
        else:
            for month in range(1,(12+1)):
                start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                for day in range(1,(day_range+1)):
                    function(year, month, day, *args, **kwargs)
                    plt.close('all')


#------------------------------------------------------------------------------#
#Ideas

#Idea_1: Triangulate with station axis correlation
#Idea_2: UI to plot next 1 hr. and store 0/1 if there is event, faster searching
#Idea_3: plot FFT frequency max intensity over time, looking for single-axis anomalies
##Furthermore, look at evens that show x and y axis signature and plot intensity ratio as an angle (5/9/08 as an example event)
#Idea_4: Convolute x and y axis of strong events and see if both signatures are very similar (could indicate an electrical commonality)
#Idea_5: Semi-interactive event database through Python
