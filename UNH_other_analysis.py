#!/usr/bin/env python37-32
#-*- coding: utf-8 -*-

"""
Created on Wed Nov 20 23:03:30 2018
Last modified on 11/20/2018

@author (orignial): tyler_chapman

This program is a development area for various non-general ULF data analysis tools and other experimental methods.
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
from UNH_analysis import ULF_txt2npy

'''
from pathlib import Path
import pandas as pd
import nolds
#import spacepy as sp #currently not installed (doesn't support pip install???)
#from spacepy import seapy
#import astropy #failed to install (needs Visual Studio update), but not currently used...has useful physical constants
from scipy.signal import savgol_filter
from itertools import cycle
import calendar
from calendar import monthrange
from datetime import date
from pprint import pprint
'''

#------------------------------------------------------------------------------#
#Functions

def ULF_SD(yyyy, mm, dd):

    #DESCRIPTION: This function computes Standard Deviation (SD)

    x_std = np.std(x_mag)
    print('the standard deviation of x_mag is: %s' %x_std)

def ULF_FD(yyyy, mm, dd):

    #DESCRIPTION: This function computes Fractal Dimension (FD)

    x_std = blank

def ULF_correlate(yyyy, mm, dd):

    #DESCRIPTION: This function preforms Correlation Analysis

    x_std = blank

def ULF_PCA(yyyy, mm, dd):

    #DESCRIPTION: This function performs Principal Component Analysis (PCA)

    x_std = blank

def ULF_WT(yyyy, mm, dd):

    #DESCRIPTION: This function preforms Wavelet Transform (WT)

    x_std = blank

def ULF_SEA(yyyy, mm, dd):

    #DESCRIPTION: This function performs Superpose Epoch Analysis (SEA)

    x_std = blank

#------------------------------------------------------------------------------#
#Executions
