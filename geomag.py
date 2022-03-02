#######################
##### GEOMAG DATA #####
#######################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from datetime import datetime
import requests
import pathlib
import os
import crowdmag as cm

#####################################
### Download relevant Geomag data ###
#####################################

def DownloadGeoMag(filenameCM,component='H',observatory='BRW',startCM=3,endCM=-1):
    
    # Extract start and end times
    date = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM)[0]
    
    # Time frame
    starttimeYMD = cm.SplitTime(date)[6][0]      # year-month-day
    starttimeHMS = cm.SplitTime(date)[7][0]      # hour-minute-second
    endtimeYMD = cm.SplitTime(date)[6][-1]       # year-month-day
    endtimeHMS = cm.SplitTime(date)[7][-1]       # hour-minute-second
    
    # Define url where the original file is located
    url = 'https://geomag.usgs.gov/ws/data/?elements={}&endtime={}T{}.000Z&format=iaga2002&id={}&sampling_period=60&starttime={}T{}.000Z&type=adjusted'.format(component,endtimeYMD,endtimeHMS,observatory,starttimeYMD,starttimeHMS)
    
    # Download the file
    myfile = requests.get(url, allow_redirects=True)
    
    # Define file location and name
    open('data/geomag/geomag{}{}_{}_{}.csv'.format(observatory,component,starttimeYMD,endtimeYMD), 'wb').write(myfile.content) 
    print("Download URL: {}".format(url))
    print("Downloaded file successfully. Observatory: {}, B-field component: {}, Start date: {}, End date: {}."
          .format(observatory,component,starttimeYMD,endtimeYMD))
    
    # Find the path of the file
    location = '\data\geomag'
    path = str(pathlib.Path(__file__).parent.resolve())                    
    print("Geomag data file location: '{}'.".format(path + location))

###################
### Read in csv ###
###################

def ReadCSVGeoMag(filenameGM,startGM=3,endGM=-1):
    data = pd.read_csv(filenameGM, skiprows=18, sep="\s+")
    
    # Selecting all rows
    rows = np.array(data.loc[:])
    
    # Defining all relevant columns, define start and end points for easier splitting
    date = rows[:,0][startGM:endGM]
    time = rows[:,1][startGM:endGM]
    doy = rows[:,2][startGM:endGM]         # day of year
    magfield = rows[:,3][startGM:endGM]
   
    # Fuse date and time to match CrowdMag date
    datetime = []
    for t in range(len(date)):
        dt = date[t] + ' ' + time[t][0:8]
        datetime.append(dt)
    datetime = np.array(datetime) 
    
    # Total time in seconds
    timeinseconds = cm.SplitTime(datetime)[8] 
    
    # Define location and component of magnetic field
    for lines in open(filenameGM, 'r'):
        if lines.startswith('DATE'):
            location = lines.split(" ")[21]

    return date,time,doy,magfield,timeinseconds,location

##########################################
### Download and define all components ###
##########################################

def DefineAllComponents(filenameCM,observatory='BRW',startCM=3,endCM=-1,startGM=0,endGM=-1,download=True):
    
    # Extract start and end times
    dateCM = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM)[0]
    starttimeYMD = cm.SplitTime(dateCM)[6][0]      # year-month-day
    endtimeYMD = cm.SplitTime(dateCM)[6][-1]       # year-month-day
    
    if download == True:    # Only download the files if needed
        # Download files from GeoMag
        DownloadGeoMag(filenameCM,component='X',observatory=observatory,startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='Y',observatory=observatory,startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='Z',observatory=observatory,startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='H',observatory=observatory,startCM=startCM,endCM=endCM)
    
    # Read in files    
    date,time,doy,magX,timeinseconds,location = ReadCSVGeoMag('data/geomag/geomag{}X_{}_{}.csv'
                                                              .format(observatory,starttimeYMD,endtimeYMD),
                                                              startGM=startGM,endGM=endGM)
    magY = ReadCSVGeoMag('data/geomag/geomag{}Y_{}_{}.csv'
                         .format(observatory,starttimeYMD,endtimeYMD),
                         startGM=startGM,endGM=endGM)[3]
    magZ = ReadCSVGeoMag('data/geomag/geomag{}Z_{}_{}.csv'
                         .format(observatory,starttimeYMD,endtimeYMD),
                         startGM=startGM,endGM=endGM)[3]
    magH = ReadCSVGeoMag('data/geomag/geomag{}H_{}_{}.csv'
                         .format(observatory,starttimeYMD,endtimeYMD),
                         startGM=startGM,endGM=endGM)[3]
    
    return date,time,doy,magX,magY,magZ,magH,timeinseconds,location
    

#######################################
### Plot magnetic field versus time ###
#######################################

def PlotBGeoMag(filenameCM,observatory='BRW',fieldtype='T',startCM=3,endCM=-1,startGM=0,endGM=-1,download=True):
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
        
    # Date, time, day of year, magnetic field data (x,y,z,h), location
    date,time,doy,magX,magY,magZ,magH,timeinseconds,location = DefineAllComponents(filenameCM,observatory,startCM,endCM,startGM,endGM,download)
    
    # Time frame
    startdate = date[0]
    enddate = date[-1]    
    starttime = time[0]
    endtime = time[-1]
    
    # Fuse date and time to match CrowdMag date
    datetime = []
    for t in range(len(date)):
        dt = date[t] + ' ' + time[t][0:8]
        datetime.append(dt)
    datetime = np.array(datetime)
        
    # Plot
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    plt.title("GeoMag : {} {} - {} {}".format(startdate,starttime,enddate,endtime), fontsize=16)
    plt.xlabel("UTC time", fontsize=12)
    
    if fieldtype == 'T':
        # Total magnetic field
        totalmag = cm.TotalMag(magX,magY,magZ) 
        ax.plot(datetime,totalmag, label="Total Magnetic Field")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
        plt.ylim(np.min(totalmag)-100,np.max(totalmag)+100)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field        
        ax.plot(datetime,magH, label="Horizontal Magnetic Field")
        plt.ylabel("Horizontal Magnetic Field (nT)", fontsize=12)
        plt.ylim(np.min(magH)-100,np.max(magH)+100)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction        
        ax.plot(datetime,magX, label="Magnetic Field - X component")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
        plt.ylim(np.min(magX)-100,np.max(magX)+100)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        ax.plot(datetime,magY, label="Magnetic Field - Y component")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        plt.ylim(np.min(magY)-100,np.max(magY)+100)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        ax.plot(datetime,magZ, label="Magnetic Field - Z component")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        plt.ylim(np.min(magZ)-100,np.max(magZ)+100)
 
    # Reduce the number of ticks for the x-axis
    xticks = ticker.MaxNLocator(10)
    ax.xaxis.set_major_locator(xticks)
    plt.xticks(fontsize = 8)
    plt.legend()
    plt.show()