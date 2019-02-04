#Snippets of code that may or not be of use

import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import signal
import pylab
import datetime as datetime
import sys
import time


from pprint import pprint

################################################################################
'''
for val in np.nditer(x_mag): #iterates through the x_mag array
    #computes the instantaneous integral via Simpsons method
    B = scipy.integrate.simps(x_mag[idx:], x=None, dx=dt, even='avg')
    B_field.append(B) #appends latest integral value to list
    idx += 1 #increase index value
x_int = np.asarray(B_field) #turns the complete list into a numpy array



        ulf_dat = [] #creates an empty list
        for year in range(yyyy_start,(yyyy_end+1)):
            for month in range(mm_start,(mm_end+1)):
                if month == mm_end:
                    for day in range(dd_start,(dd_end+1)):
                        np_file = ULF_txt2npy(year, month, day, station=station)
                        dat_arr = np.load(np_file)
                        dat_arr = dat_arr.tolist()
                        if len(ulf_dat) == 0:
                            ulf_dat = dat_arr
                        else:
                            ulf_dat.extend(dat_arr)
                        dat_arr = None
                else:
                    start_DOTW, day_range = monthrange(year,month) #calculates number of days in specified month, taking leap years into account
                    for day in range(1,(day_range+1)):
                        np_file = ULF_txt2npy(year, month, day, station=station)
                        dat_arr = np.load(np_file)
                        dat_arr = dat_arr.tolist()
                        if len(ulf_dat) == 0:
                            ulf_dat = dat_arr
                        else:
                            ulf_dat.extend(dat_arr)
                        dat_arr = None



a = np.arange(800)
b = a.reshape((4,200))
c = np.transpose(b)
#gamma = np.transpose(np.arange(400).reshape((4,100)))
#pprint(c)
d = np.transpose(c.reshape((20,10,4)), axes=(1, 0, 2))
e = np.mean(d, axis=0)
pprint(e)




Year = omni_dat[:,0]         #(1995 to 2006)
Day	= omni_dat[:,1] 	     #(365 or 366)
Hour = omni_dat[:,2] 	     #(0 to 23)
Minute = omni_dat[:,3] 	     #(0 to 59)
ID_IMF = omni_dat[:,4]       #ID for IMF spacecraft
ID_SW = omni_dat[:,5]        #ID for SW Plasma spacecraft
N_IMF = omni_dat[:,6]        #Number of points in IMF averages
N_SW = omni_dat[:,7]         #Number of points in Plasma averages
Per_Int = omni_dat[:,8]      #Percent interp
Timeshift = omni_dat[:,9]    #[sec]
RMS_Time = omni_dat[:,10]    #Timeshift
RMS_Phase = omni_dat[:,11]   #Phase front normal
Delta_Time = omni_dat[:,12]  #Time btwn observations
F_Mag_Avg = omni_dat[:,13]   #Field magnitude average [nT]
Bx_GSEM = omni_dat[:,14]     #[nT] (GSE and GSM)
By_GSE = omni_dat[:,15]      #[nT] (GSE)
Bz_GSE = omni_dat[:,16]      #[nT] (GSE)
By_GSM = omni_dat[:,17]      #[nT] (GSE)
Bz_GSM = omni_dat[:,18]      #[nT] (GSE)
RMS_Scalar = omni_dat[:,19]  #RMS SD B Scalar [nT]
RMS_Vector = omni_dat[:,20]  #RMS SD field vector [nT]
Flow_Speed = omni_dat[:,21]  #[km/s]
Vx_GSE = omni_dat[:,22]      #Velocity [km/s]
Vy_GSE = omni_dat[:,23]      #Velocity [km/s]
Vz_GSE = omni_dat[:,24] 	 #Velocity [km/s]
P_Density = omni_dat[:,25]   #Proton Density [n/cc]
Temp_K = omni_dat[:,26]      #Temperature [K]
Flow_Pres = omni_dat[:,27]   #Flow pressure [nPa]
E_Field = omni_dat[:,28]     #Electric Field [mV/m]
Plasma_Beta = omni_dat[:,29]
Alfven_Mach = omni_dat[:,30] #Mach Number
X_SC = omni_dat[:,31]        #(s/c), GSE, Re
Y_SC = omni_dat[:,32]        #(s/c), GSE, Re
Z_SC = omni_dat[:,33]        #(s/c), GSE, Re
BSN_X_GSE = omni_dat[:,34]   #BSN location, Re
BSN_Y_GSE = omni_dat[:,35]   #BSN location, Re
BSN_Z_GSE = omni_dat[:,36]   #BSN location, Re
AE_IDX = omni_dat[:,37]      #Index [nT]
AL_IDX = omni_dat[:,38]      #Index [nT]
AU_IDX = omni_dat[:,39]      #Index [nT]
SYM_D_IDX = omni_dat[:,40]   #Index [nT]
SYM_H_IDX = omni_dat[:,41]   #Index [nT]
ASY_D_IDX = omni_dat[:,42]   #Index [nT]
ASY_H_IDX = omni_dat[:,43]   #Index [nT]
PC_N_IDX = omni_dat[:,44]    #Index [nT]
Mag_Mach = omni_dat[:,45]    #Magnetosonic mach number


#------------------------------------------------------------------------------#

#Plot empirical cumulative distribution using Matplotlib and Numpy
import numpy as np
import matplotlib as plt

num_bins = 20
counts, bin_edges = np.histogram (data, bins=num_bins, normed=True)
cdf = np.cumsum (counts)
plt.plot (bin_edges[1:], cdf/cdf[-1])


#Print median and tail statistics using Numpy
import numpy

for q in [50, 90, 95, 100]:
  print ("{}%% percentile: {}".format (q, np.percentile(data, q)))
'''
#------------------------------------------------------------------------------#
#checking for correct multi-dim array reshaping and manipulation

a1 = np.arange(11,18)
b1 = np.arange(21,28)
c1 = np.arange(31,38)
d1 = np.arange(41,48)
e1 = np.arange(51,58)
f1 = np.arange(61,68)
a2 = np.arange(111,118)
b2 = np.arange(121,128)
c2 = np.arange(131,138)
d2 = np.arange(141,148)
e2 = np.arange(151,158)
f2 = np.arange(161,168)

array = np.stack((a1,b1,c1,d1,e1,f1,a2,b2,c2,d2,e2,f2))
print(array.shape)

row, col = np.shape(array)
narray = np.transpose(array.reshape(int(row/6), 6, col), axes=(0,1,2))
pprint(array)
pprint(narray.shape)
print(narray[0,0,0]) #1
print(narray[1,0,0]) #101
print(narray[0,1,0]) #6
print(narray[0,2,0]) #11
print(narray[0,0,1]) #2
print(narray[0,0,2]) #3
print(narray[0,5,0]) #20
print(narray[1,5,6]) #200
