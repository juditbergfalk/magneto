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
import filterdata as filt

#####################################
### Download relevant GeoMag data ###
#####################################

def DownloadGeoMag(filenameCM,
                   component = 'X',
                   observatory = 'BRW',
                   startCM = 3, endCM = -1):
    """
    Using the CrowdMag .csv file, download the relevant GeoMag data for the same timeframe.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    component : string, default=horizontal, component of the magnetic field
    observatory : string, default=Barrow Observatory, code for the observatory
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming

    Returns
    ----------
    Downloads the GeoMag datafile (.csv format) to the user's computer. 
    """
    
    # Extract start and end times
    date = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM,rollingave=False,dc_shift=False,bkps=None)[0]
    
    # Time frame
    starttimeYMD = cm.SplitTime(date)[6][0]      # year-month-day
    starttimeHMS = cm.SplitTime(date)[7][0]      # hour-minute-second
    endtimeYMD = cm.SplitTime(date)[6][-1]       # year-month-day
    endtimeHMS = cm.SplitTime(date)[7][-1]       # hour-minute-second
    
    # Define url where the original file is located
    url = 'https://geomag.usgs.gov/ws/data/?elements={}&endtime={}T{}.000Z&format=iaga2002&id={}&sampling_period=60&starttime={}T{}.000Z&type=variation'.format(component,endtimeYMD,endtimeHMS,observatory,starttimeYMD,starttimeHMS)
    
    # Download the file
    myfile = requests.get(url, allow_redirects = True)
    
    # Define file location and name
    open(os.getcwd()+'/data/geomag/geomag{}{}_{}_{}.csv'.format(observatory,component,starttimeYMD,endtimeYMD), 'wb').write(myfile.content) 
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

def ReadCSVGeoMag(filenameGM,
                  startGM = 0, endGM = -1):
    """
    Read the GeoMag .csv file and return the date, magnetic field, etc.
    
    Parameters
    ----------
    filenameGM : string, GeoMag .csv filename
    startGM : int, default=0, starting row for trimming
    endGM : int, default=-1 (last element), ending row for trimming

    Returns
    ----------
    date : numpy array, year-month-day
    time : numpy array, hour-minute-second
    doy : numpy array, day of year
    magfield : numpy array, strength of the magnetic field
    timeinseconds : numpy array, time in seconds
    location : numpy array, observatory code
    """
    
    # Read in .csv file, skip first 19 rows
    data = pd.read_csv(filenameGM, skiprows=20, sep="\s+")
    
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

def DefineAllComponents(filenameCM,
                        observatory = 'BRW',
                        startCM = 3, endCM = -1, startGM = 0, endGM = -1,
                        download = True,
                        dc_shift = False,
                        filter_signal = 'raw'):
    """
    GeoMag data: Download all components of the magnetic field for the given timeframe and define all variables. 
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    observatory : string, default=Barrow Observatory, code for the observatory
    startCM : int, default=3, starting row for trimming CrowdMag data
    endCM : int, default=-1 (last element), ending row for trimming CrowdMag data
    startGM : int, default=0, starting row for triming GeoMag data
    endGM : int, default=-1 (last element), ending row for trimming GeoMag data
    download : boolean, if True: download the .csv file from GeoMag
    dc_shift : boolean, if True: data will be DC shifted to zero
    filter_signal : string, can be 'raw' (no filter, 
                                   'filtfilt' (scipy.filtfilt), 
                                   'savgol' (scipy.savgol_filter), 
                                   'ffthighfreq', 
                                   'fftbandpass', 
                                   'combo' (filtfilt and fftbandbpass)

    Returns
    ----------
    date : numpy array, year-month-day
    time : numpy array, hour-minute-second
    doy : numpy array, day of year
    magX,magY,magZ,magH,totalmag : numpy array, strength of the X,Y,Z,H and total component of the magnetic field
    timeinseconds : numpy array, time in seconds
    location : numpy array, observatory code
    """
    
    # Extract start and end times
    dateCM = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM,rollingave=False,dc_shift=False,bkps=None)[0]
    starttimeYMD = cm.SplitTime(dateCM)[6][0]      # year-month-day
    endtimeYMD = cm.SplitTime(dateCM)[6][-1]       # year-month-day
    
    if download:                 # Only download the files if needed
        # Download files from GeoMag
        DownloadGeoMag(filenameCM,component='X',observatory=observatory,
                       startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='Y',observatory=observatory,
                       startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='Z',observatory=observatory,
                       startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='H',observatory=observatory,
                       startCM=startCM,endCM=endCM)
        DownloadGeoMag(filenameCM,component='F',observatory=observatory,
                       startCM=startCM,endCM=endCM)
    
    # Read in GeoMag .csv file for each component    
    date,time,doy,magX,timeinseconds,location = ReadCSVGeoMag(os.getcwd()+'/data/geomag/geomag{}X_{}_{}.csv'.format(observatory,starttimeYMD,endtimeYMD),startGM=startGM,endGM=endGM)
    magY = ReadCSVGeoMag(os.getcwd()+'/data/geomag/geomag{}Y_{}_{}.csv'
                         .format(observatory,starttimeYMD,endtimeYMD),
                         startGM=startGM,endGM=endGM)[3]
    magZ = ReadCSVGeoMag(os.getcwd()+'/data/geomag/geomag{}Z_{}_{}.csv'
                         .format(observatory,starttimeYMD,endtimeYMD),
                         startGM=startGM,endGM=endGM)[3]
    #magH = ReadCSVGeoMag(os.getcwd()+'/data/geomag/geomag{}H_{}_{}.csv'
                         #.format(observatory,starttimeYMD,endtimeYMD),
                         #startGM=startGM,endGM=endGM)[3]
    magH = cm.HorizontalMag(magX,magY)
    totalmag = ReadCSVGeoMag(os.getcwd()+'/data/geomag/geomag{}F_{}_{}.csv'
                         .format(observatory,starttimeYMD,endtimeYMD),
                         startGM=startGM,endGM=endGM)[3]
    #totalmag = cm.TotalMag(magX,magY,magZ)
    
    # DC shift
    if dc_shift:
        totalmag = filt.DC_Shift(totalmag,bkps=None)
        magH = filt.DC_Shift(magH,bkps=None)
        magX = filt.DC_Shift(magX,bkps=None)
        magY = filt.DC_Shift(magY,bkps=None)
        magZ = filt.DC_Shift(magZ,bkps=None)   
    
    # Filter signal: None
    if filter_signal == 'raw':
        pass
    
    # Filter signal: digital filter forward and backward to a signal
    if filter_signal == 'filtfilt':
        totalmag = filt.Filter_filtfilt(totalmag)
        magH = filt.Filter_filtfilt(magH)
        magX = filt.Filter_filtfilt(magX)
        magY = filt.Filter_filtfilt(magY)
        magZ = filt.Filter_filtfilt(magZ)
    
    # Filter signal: Savitzky-Golay filter
    if filter_signal == 'savgol':
        totalmag = filt.Filter_savgol(totalmag)
        magH = filt.Filter_savgol(magH)
        magX = filt.Filter_savgol(magX)
        magY = filt.Filter_savgol(magY)
        magZ = filt.Filter_savgol(magZ)
    
    # Filter signal: FFT high freq filter
    if filter_signal == 'ffthighfreq':
        totalmag = filt.Filter_ffthighfreq(totalmag)[0]
        magH = filt.Filter_ffthighfreq(magH)[0]
        magX = filt.Filter_ffthighfreq(magX)[0]
        magY = filt.Filter_ffthighfreq(magY)[0]
        magZ = filt.Filter_ffthighfreq(magZ)[0]
    
    # Filter signal: FFT bandpass filter
    if filter_signal == 'fftbandpass':
        totalmag = filt.Filter_fftbandpass(totalmag)
        magH = filt.Filter_fftbandpass(magH)
        magX = filt.Filter_fftbandpass(magX)
        magY = filt.Filter_fftbandpass(magY)
        magZ = filt.Filter_fftbandpass(magZ)
    
    # Filter signal: Combo: Digital filter forward and backward to a signal and FFT bandpass filter
    if filter_signal == 'combo':
        totalmag = filt.Filter_filtfilt(totalmag)
        magH = filt.Filter_filtfilt(magH)
        magX = filt.Filter_filtfilt(magX)
        magY = filt.Filter_filtfilt(magY)
        magZ = filt.Filter_filtfilt(magZ)
        
        totalmag = filt.Filter_fftbandpass(totalmag)
        magH = filt.Filter_fftbandpass(magH)
        magX = filt.Filter_fftbandpass(magX)
        magY = filt.Filter_fftbandpass(magY)
        magZ = filt.Filter_fftbandpass(magZ)
        
    #magH = -magH
    
    # Calculate magnitude
    totalmag = np.abs(totalmag)
    magH = np.abs(magH)
    
    return date,time,doy,magX,magY,magZ,magH,totalmag,timeinseconds,location
    

#######################################
### Plot magnetic field versus time ###
#######################################

def PlotBGeoMag(filenameCM,
                observatory = 'BRW',
                fieldtype = 'F',
                startCM = 3, endCM = -1, startGM = 0, endGM = -1,
                download = True,
                dc_shift = False,
                filter_signal = 'raw'):
    """
    Plotting the GeoMag data of the chosen component of the magnetic field. 
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    observatory : string, default=Barrow Observatory, code for the observatory
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming CrowdMag data
    endCM : int, default=-1 (last element), ending row for trimming CrowdMag data
    startGM : int, default=0, starting row for triming GeoMag data
    endGM : int, default=-1 (last element), ending row for trimming GeoMag data
    download : boolean, if True: download the .csv file from GeoMag
    dc_shift : boolean, if True: data will be DC shifted to zero
    filter_signal : string, can be 'raw' (no filter, 
                                   'filtfilt' (scipy.filtfilt), 
                                   'savgol' (scipy.savgol_filter), 
                                   'ffthighfreq', 
                                   'fftbandpass', 
                                   'combo' (filtfilt and fftbandbpass)

    Returns
    ----------
    Plot of the magnetic field from GeoMag data.    
    """
    
    # Key:
    ##### fieldtype = 'F'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
        
    # Date, time, day of year, magnetic field data (x,y,z,h), location
    date,time,doy,magX,magY,magZ,magH,totalmag,timeinseconds,location = DefineAllComponents(filenameCM,observatory,
                                                                                           startCM,endCM,startGM,endGM,
                                                                                           download,
                                                                                           dc_shift,
                                                                                           filter_signal)
    
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
    
    if fieldtype == 'F':
        # Total magnetic field
        ax.plot(datetime,totalmag, label="Total Magnetic Field", color="orange")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
  
    if fieldtype == 'H':        
        # Horizontal magnetic field        
        ax.plot(datetime,magH, label="Horizontal Magnetic Field", color="orange")
        plt.ylabel("Horizontal Magnetic Field (nT)", fontsize=12)
   
    if fieldtype == 'X':        
        # Magnetic field - X direction        
        ax.plot(datetime,magX, label="Magnetic Field - X component", color="orange")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
   
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        ax.plot(datetime,magY, label="Magnetic Field - Y component", color="orange")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
       
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        ax.plot(datetime,magZ, label="Magnetic Field - Z component", color="orange")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)

    # Reduce the number of ticks for the x-axis
    xticks = ticker.MaxNLocator(10)
    ax.xaxis.set_major_locator(xticks)
    plt.xticks(fontsize = 8)
    plt.legend()
    plt.show()