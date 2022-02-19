#########################
##### CROWDMAG DATA #####
#########################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from datetime import datetime

###################
### Read in csv ###
###################

def ReadCSVCrowdMag(filename,start=3,end=-1):
    data = pd.read_csv(filename)
    
    # Selecting all rows
    rows = np.array(data.loc[:])
    
    # Defining all relevant columns, define start and end points for easier splitting
    date = rows[:,0][start:end]
    magX = rows[:,3][start:end]
    magY = rows[:,4][start:end]
    magZ = rows[:,5][start:end]
    
    return date,magX,magY,magZ

######################
### Split UTC date ###
######################

def SplitTime(date):
    
    # Create list for each value
    year,month,day,hour,minute,second,ymd,hms = [],[],[],[],[],[],[],[]
    
    # Strip time
    for i in range(len(date)):
        dt = datetime.fromisoformat(date[i])
        # year
        yr = float(dt.strftime('%Y'))
        year.append(yr)    
        # month
        mo = float(dt.strftime('%m'))
        month.append(mo)
        # day
        d = float(dt.strftime('%d'))
        day.append(d)
        # hour
        hr = float(dt.strftime('%H'))
        hour.append(hr)
        # minute
        mi = float(dt.strftime('%M'))
        minute.append(mi)
        # second
        s = float(dt.strftime('%S'))
        second.append(s)
        # year:month:day
        yearmonthday = dt.strftime('%Y-%m-%d')
        ymd.append(yearmonthday)
        # hour:minute:second
        hourminutesecond = dt.strftime('%H:%M:%S')
        hms.append(hourminutesecond)       
    
    # Change lists to numpy arrays
    year = np.array(year)
    month = np.array(month)
    day = np.array(day)
    hour = np.array(hour)
    minute = np.array(minute)
    second = np.array(second)
    ymd = np.array(ymd)
    hms = np.array(hms)
    
    # Total seconds 
    timeinseconds = []
    for t in range(len(year)):
        dt = datetime(int(year[t]),int(month[t]),int(day[t]),int(hour[t]),int(minute[t]),int(second[t]))
        dtseconds = (dt-datetime(1970,1,1)).total_seconds()
        timeinseconds.append(dtseconds)
    timeinseconds = np.array(timeinseconds)
    
    return year,month,day,hour,minute,second,ymd,hms,timeinseconds

##################################################
### Horizontal component of the magnetic field ###
##################################################

def HorizontalMag(x,y):
    x = x.astype(float)
    y = y.astype(float)
    H = np.sqrt(x**2 + y**2)
    return H

############################
### Total magnetic field ###
############################

def TotalMag(x,y,z):
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    total = np.sqrt(x**2 + y**2 + z**2)
    return total

#######################################
### Plot magnetic field versus time ###
#######################################

def PlotBCrowdMag(filename,fieldtype='T',start=3,end=-1):
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
        
    # Date and Magnetic field data (x,y,z)
    date, magX, magY, magZ = ReadCSVCrowdMag(filename,start,end)
    
    # Time frame
    starttime = date[0]
    endtime = date[-1]
        
    # Plot
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    plt.title("CrowdMag : {} - {}".format(starttime,endtime), fontsize=16)
    plt.xlabel("UTC time", fontsize=12)
    
    if fieldtype == 'T':
        # Total magnetic field
        totalmag = TotalMag(magX,magY,magZ) 
        ax.scatter(date,totalmag)
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
        plt.ylim(np.min(totalmag[:-3])-500,np.max(totalmag[:-3])+500)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field
        horizontalmag = HorizontalMag(magX,magY)          
        ax.scatter(date,horizontalmag)
        plt.ylabel("Horizontal Magnetic Field (nT)", fontsize=12)
        plt.ylim(np.min(horizontalmag[:-3])-500,np.max(horizontalmag[:-3])+500)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction        
        ax.scatter(date,magX)
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
        plt.ylim(np.min(magX[:-3])-500,np.max(magX[:-3])+500)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        ax.scatter(date,magY)
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        plt.ylim(np.min(magY[:-3])-500,np.max(magY[:-3])+500)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        ax.scatter(date,magZ)
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        plt.ylim(np.min(magZ[:-3])-500,np.max(magZ[:-3])+500)
 
    # Reduce the number of ticks for the x-axis
    xticks = ticker.MaxNLocator(10)
    ax.xaxis.set_major_locator(xticks)
    plt.xticks(fontsize = 8)
    plt.show()