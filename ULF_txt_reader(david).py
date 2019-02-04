### #!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Wed Mar 14 14:35:47 2018
Last modified on 10/11/2018

@author (orignial): david_kenward
@modified: tyler_chapman

This program to read in ULF data saved as a .txt, then export as numpy save

NOTES: [] comments are mine, added to David's
"""

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import pylab
import datetime as datetime
import sys
import time

#-------------------------------------------------------------#

datapath = open('C:/Users/Tyler/Desktop/Project Lisbon/Datasets/ULF/LYR/TXT/2008_02_21_LYR_dbdt_v1.txt', 'r')
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

np.save('C:/Users/Tyler/Desktop/Project Lisbon/Datasets/ULF/LYR/NPY/2008_02_21_LYR.npy',ulf_dat)

#-------------------------------------------------------------#

dat_arr = np.load('C:/Users/Tyler/Desktop/Project Lisbon/Datasets/ULF/LYR/NPY/2008_02_21_LYR.npy')

d1 = dat_arr[:,0]
y = dat_arr[:,1] #y signal
x = dat_arr[:,2] #x signal
d4 = dat_arr[:,3] #no idea what this is, it just is all 0.0 [seems to be configured for a 3rd axis input]

hour = np.arange(0,25,1)
time_list = np.arange(0,86400,.1)

dt = .1 #timestep
fs = 1./dt #sampling [Hz]
nfft = 2**10 #8=256, 9=512, 10=1024, 11=2048, 12=4096, 13=8192
nov = nfft/2

#-------------------------------------------------------------#
'''
#attemps to close all currently open windows before plotting new ones...yet unsuccessful
plt.show(block=False)
time.sleep(5)
plt.pause(3)
plt.close("all")
'''

fig = plt.figure()
plt.suptitle('Hornsund ULF Data')

ax1 = fig.add_subplot(412)
ax1.set_ylabel('dBx/dT')
ax1.set_xticks(np.arange(0,86401,(14400/2)/2))
ax1.set_xticklabels(np.arange(0,25,1))
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.plot(time_list, x)

ax2 = fig.add_subplot(411, sharex=ax1)
pxx1, freq1, t1, cax1 = pylab.specgram(x, nfft, fs, noverlap = nov, cmap=cm.jet,
                                       vmin=-50, vmax=5)
ax2.set_ylabel('Freq')
ax2.set_ylim(0,5)
#cb1 = plt.colorbar(cax1, orientation='vertical',pad=.005)
plt.setp(ax2.get_xticklabels(), visible=False)

ax3 = fig.add_subplot(413, sharex=ax1)
pxx2, freq2, t2, cax2 = pylab.specgram(x, nfft, fs, noverlap = nov, cmap=cm.jet,
                                       vmin=-50, vmax=5)
ax3.set_ylabel('Freq')
ax3.set_ylim(0,5)
plt.setp(ax3.get_xticklabels(), visible=False)


ax4 = fig.add_subplot(414, sharex=ax1)
ax4.set_ylabel('dBy/dT')
ax4.plot(time_list, y)

#ax4.set_xticks(np.arange(0,86401,14400))
#ax4.set_xticklabels(np.arange(0,25,4))

#plt.subplots_adjust(top=.9,bottom=.1)
#plt.tight_layout(h_pad=.0)

fx, pxx = signal.periodogram(x[252000:288001],fs)
fy, pxy = signal.periodogram(y[252000:288001],fs)

fig,ax = plt.subplots()
plt.title('ULF HND periodogram 7 - 8 UT')
plt.axvline(x=5e-3, color='b') #[adds a verticle line across the axis]
plt.axvline(x=16e-3, color='pink')
ax.semilogx(fx,pxx)
ax.semilogx(fy,pxy)
ax.set_xlabel('Frequency')
ax.set_ylabel('Power')

#Attempt to fit a sine wave to time series
fig,ukn = plt.subplots()


xx = x[252000:288001]
yy = y[252000:288001]

sample = len(xx)
amp = .4
freq = .5 * 5e-3 #Hz
phase = 0.

time_list = np.arange(0, sample, 1)

genx = amp * np.sin(2 * np.pi * freq * time_list)

plt.plot(time_list,genx)
plt.plot(time_list,xx)

plt.show()
