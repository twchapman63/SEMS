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

#Files to import
from UNH_analysis import *
from ULF_batch import *

#------------------------------------------------------------------------------#
#EXECUTIONS
'''
save_place = 'C:/Users/Tyler/Desktop/current_work/'
for day in range(3,(31+1)):
    ULF_subday(2008, 1, day, 0, 24, station='LYR', showplot=False, smoothed=True, saveplot=True, f_max=0.2, save_dir=save_place)
    plt.close('all')
'''
#ULF_subday(2008, 1, 28, 0, 24, station='LYR', showplot=True, smoothed=True, saveplot=False, f_max=0.2)


#OMNI_plots(2008, 5, 8, showplot=True)

save_place = 'C:/Users/Tyler/Desktop/current_work/'
for day in range(1,(30+1)):
    #ACES_plots(2008, 2, day, showplot=False, saveplot=True)
    #SuperMag_plots(2008, 6, day, showplot=False, smoothed=False, saveplot=True)
    #SuperMag_grad(2008, 6, day, showplot=False, smoothed=False, saveplot=True)
    #ULF_combo_plot(2008, 2, day, choose='SuperMag', showplot=False, saveplot=True, smoothed=True)
    #OMNI_save_dir(2008,4,day)
    ULF_subday(2008, 4, day, 0, 24, station='LYR', showplot=False, smoothed=True, saveplot=True, f_max=0.2, save_dir=save_place)
    plt.close('all')

#ULF_combo_plots(2008, 2, 21, choose='both', limit_same=True, showplot=True, saveplot=False, smoothed=True, sample_sec=60)
#SuperMag_plots(2008, 4, 21, showplot=True, smoothed=False, saveplot=False)
#SuperMag_spec(2008, 4, 19, showplot=True, saveplot=False)
#ULF_B_field_plots(2008, 4, 19, showplot=True)

#OMNI_ULF_plots(2008, 5, 12, showplot=True)
#NEED TO ADD INTERPOLATION....also error currently???

#OMNI_all_plots(2008, 2, 21, saveplot=True)

#------------------------------------------------------------------------------#
#Executions

#TEST_CASE_2(2008, 5, 9)
#ULF_subday(2008, 5, 8, 0, 24, station='LYR', showplot=True, smoothed=True, saveplot=False, f_max=0.2)
#ULF_subday_multistation_grad(2016, 3, 29, 7, 11, showplot=True, smoothed=True, saveplot=False)
'''
for day in range(23,(31+1)):
    for hour in range(1,25):
        ULF_subday_multistation(2016, 3, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True)
        #ULF_subday(2008, 1, day, (hour-1), hour, showplot=False, smoothed=True, saveplot=True)
        plt.close('all')
'''
#run_range(2008, 1, 1, 2009, 1, 1, station='HOR', third=True) #about 1.5 hrs. to run a full year
'''
st = 'LYR'
year = 2009
for month in range(3,(12+1)):
    for day in range(1,(31+1)):
        ULF_subday(year, month, day, 0, 8, showplot=False, saveplot=True, station=st)
        ULF_subday(year, month, day, 8, 16, showplot=False, saveplot=True, station=st)
        ULF_subday(year, month, day, 16, 24, showplot=False, saveplot=True, station=st)
        plt.close('all')

day = 21
ULF_subday(2008, month, day, 0, 8, showplot=True, saveplot=False, station=st)
ULF_subday(2008, month, day, 8, 16, showplot=True, saveplot=False, station=st)
ULF_subday(2008, month, day, 16, 24, showplot=True, saveplot=False, station=st)
'''

#run_range(2008, 1, 1, 2009, 1, 1, station='HOR', third=True)
