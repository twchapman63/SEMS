#!/usr/bin/env python37-32
#-*- coding: utf-8 -*-

"""
Created on Mon Nov 12 06:34:03 2018
Last modified on 11/12/2018

@author (orignial): tyler_chapman

This program analyizes nonlinear dynamic parameters in ULF data.
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
from pathlib import Path

#Specific to this script
from UNH_analysis import ULF_txt2npy
import nolds

#------------------------------------------------------------------------------#
#Functions

def ULF_entropy(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: Measures the complexity of a time-series, based on approximate entropy.

    #------------------------------------------------------------------------------#
    #Variables and settings

    np_file = ULF_txt2npy(yyyy, mm, dd, station=station)
    dat_arr = np.load(np_file)

    d1 = dat_arr[:,0] #time of day in seconds (0.1 sec increments, 10 Hz sample)
    x_mag = dat_arr[:,1] #x-axis signal ###SWAPPED FROM WHAT DAVID HAD###
    y_mag = dat_arr[:,2] #y-axis signal
    d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

    hour = np.arange(0,25,1)

    dt = 0.1 #timestep
    fs = 1./dt #sampling [Hz]
    nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192 (number of fft's)
    nov = nfft/2 #overlap in fft segments, of length nfft

    my_dpi=120
    plt_clr = '#000000' #blue
    ln_wth= 0.5 #plot linewidth
    #------------------------------------------------------------------------------#
    #Main

    #calculations
    temp_x = []
    temp_y = []
    x_sampen = np.array([0], dtype='float32')
    y_sampen = np.array([0], dtype='float32')
    n_pts = 864 #number of iteration, equal to resolution of 100 sec
    idx = np.linspace(0, (len(x_mag)-1), n_pts, dtype=int)

    e_b = 10000
    for val in range(0,len(idx)):
        if val == 0:
            temp_x = nolds.sampen(x_mag[0:idx[val+1]], emb_dim=e_b)
        elif val == (len(idx)-1):
            temp_x = nolds.sampen(x_mag[idx[val-1]:idx[val]], emb_dim=e_b)
        else:
            temp_x = nolds.sampen(x_mag[idx[val-1]:idx[val+1]], emb_dim=e_b)
        x_sampen = np.c_[x_sampen, temp_x] if x_sampen.size else temp_x
    x_sampen = np.transpose(x_sampen)

    for val in range(0,len(idx)):
        if val == 0:
            temp_y = nolds.sampen(y_mag[0:idx[val+1]], emb_dim=e_b)
        elif val == (len(idx)-1):
            temp_y = nolds.sampen(y_mag[idx[val-1]:idx[val]], emb_dim=e_b)
        else:
            temp_y = nolds.sampen(y_mag[idx[val-1]:idx[val+1]], emb_dim=e_b)
        y_sampen = np.c_[y_sampen, temp_y] if y_sampen.size else temp_y
    y_sampen = np.transpose(y_sampen)

    print("hello")
    print(y_sampen.shape)
    #np.savetxt('texttttt.txt', np.c_[x_sampen, y_sampen])
    np.savetxt('texttttt.txt', x_sampen)

    #plotting
    fig = plt.figure(figsize=(1200/my_dpi, 800/my_dpi), dpi=my_dpi) #initalize the figure
    title = 'Sampled Entropy from ' + station + ' on' + str(mm) + '/' + str(dd) + '/' + str(yyyy)
    plt.suptitle(title)

    x1 = fig.add_subplot(211)
    x1.set_ylabel('Bx [nT]')
    x1.set_xticks(np.arange(0,864001,(144000/2)/2))
    x1.set_xticklabels(hour)
    #plt.setp(x1.get_xticklabels(), visible=False)
    x1.plot(idx, x_sampen, plt_clr, linewidth=ln_wth)

    x2 = fig.add_subplot(212, sharex=x1)
    #plt.setp(x2.get_xticklabels(), visible=False)
    x2.set_ylabel('By [nt]')
    x2.plot(idx, y_sampen, plt_clr, linewidth=ln_wth)

    #------------------------------------------------------------------------------#
    #saving and displaying plots

    png_dir = 'C:/Users/Tyler/Desktop/nonlinear/'

    if mm >= 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_0' + str(dd) + '_entropy.png'
    elif mm < 10 and dd >= 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_' + str(dd) + '_entropy.png'
    elif mm < 10 and dd < 10:
        png_file = png_dir + str(yyyy) + '_0' + str(mm) + '_0' + str(dd) + '_entropy.png'
    else:
        png_file = png_dir + str(yyyy) + '_' + str(mm) + '_' + str(dd) + '_entropy.png'

    if showplot == True:
        plt.show()

    if saveplot == True:
        fig2.savefig(png_file, bbox_inches='tight')

def ULF_corr_dim(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):
    #DESCRIPTION: A measure of the correlation fractal dimension of a time series which is also related to complexity.

    #x_corr_dim = nolds.corr_dim(x_mag, emb_dim=2)
    print('corr_dim size is:')
    #print(x_corr_dim)

def ULF_L_exp(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):

    #DESCRIPTION: Positive Lyapunov exponents indicate chaos and unpredictability.

    #Nolds provides the algorithm of Rosenstein et al. (lyap_r) to estimate the largest Lyapunov exponent
    #and the algorithm of Eckmann et al. (lyap_e) to estimate the whole spectrum of Lyapunov exponents.

    #x_lyap_r = nolds.lyap_r(x_mag)
    #x_lyap_e = nolds.lyap_e(x_mag)
    print('lyap_r size is:')
    #print(x_lyap_r.size)
    print('lyap_e size is:')
    #print(x_lyap_e.size)

def ULF_hurst_exp(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):
    #The hurst exponent is a measure of the “long-term memory” of a time series. It can be used to determine
    #whether the time series is more, less, or equally likely to increase if it has increased in previous steps.
    #This property makes the Hurst exponent especially interesting for the analysis of stock data.

    #x_hurst_rs = nolds.hurst_rs(x_mag)
    print('hurst_rs size is:')
    #print(x_hurst_rs)

def ULF_dfa(yyyy, mm, dd, station='LYR', showplot=False, saveplot=False):
    #Detrended Fluctuation Analysis (DFA) measures the Hurst parameter H, which is very similar to the Hurst exponent. The main difference is
    #that DFA can be used for non-stationary processes (whose mean and/or variance change over time).

    #x_dfa = nolds.dfa(x_mag)
    print('dfa size is:')
    #print(x_dfa)

#------------------------------------------------------------------------------#
#Executions

ULF_entropy(2008, 2, 21)
