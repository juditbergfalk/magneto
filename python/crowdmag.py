#########################
##### CROWDMAG DATA #####
#########################

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import pandas as pd
from datetime import datetime
from ipywidgets import Dropdown, Box, Layout, Label
import warnings
import filterdata as filt

#################################
### Dropdown menu of filelist ###
#################################

# Make a list of filenames from the crowdmag folder
filelist = os.listdir(os.getcwd()+'./data/crowdmag/')                   

# Create a dropdown menu and label
form_item_layout = Layout(display='flex', flex_flow='row', justify_content='space-between')
dropdownmenu = Dropdown(options=filelist, value='crowdmag_March 14 2022_iPhone12,1_2022-03-14 135556.csv')
form_items = [Box([Label(value='CrowdMag file: '),dropdownmenu], layout=form_item_layout)]

# Call function for dropdown menu
CrowdMagFileList = Box(form_items, layout=Layout(display='flex', flex_flow='column', 
                                                 border='solid 2px', align_items='stretch', width='50%'))

############################
### Total magnetic field ###
############################

def TotalMag(x,y,z):
    """Calculating the total magnetic field.
    
    Parameters
    ----------
    x,y,z : int, X,Y,Z component of the magnetic field

    Returns
    ----------
    total : int, total magnetic field
    """
    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    total = np.sqrt(x**2 + y**2 + z**2)
    return total

##################################################
### Horizontal component of the magnetic field ###
##################################################

def HorizontalMag(x,y):    
    """
    Calculating the horizontal component of the magnetic field.
    
    Parameters
    ----------
    x,y : int, X and Y component of the magnetic field

    Returns
    ----------
    H : int, Horizontal component of the magnetic field
    """
    x = x.astype(float)
    y = y.astype(float)
    H = np.sqrt(x**2 + y**2)
    return H

###################
### Read in csv ###
###################

def ReadCSVCrowdMag(filenameCM,
                    startCM = 3, endCM = -1,
                    rollingave = False,
                    window_size = 10,
                    dc_shift = False,
                    bkps = None,
                    filter_signal = 'raw'):
    """
    Read CrowdMag .csv files and return the arrays for date and X,Y,Z component of magnetic field.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming
    rollingave : boolean, if True: calculates the rolling average
    window_size : int, default=10, window-size for the rolling average
    dc_shift : boolean, if True: data will be DC shifted to zero
    bkps : int, default=4, max number of predicted breaking points
    filter_signal : string, can be 'raw' (no filter, 
                                   'filtfilt' (scipy.filtfilt), 
                                   'savgol' (scipy.savgol_filter), 
                                   'ffthighfreq', 
                                   'fftbandpass', 
                                   'combo' (filtfilt and fftbandbpass)

    Returns
    ----------
    date,magX,magY,magZ : numpy arrays
    """
    
    # Read in .csv file
    data = pd.read_csv(filenameCM)
                                    #, parse_dates=['Time (UTC)'])
      
    # Selecting all rows
    rows = np.array(data.loc[:])
    
    # Defining all relevant columns, define start and end points for easier trimming
    date = rows[:,0][startCM:endCM]
    magX = rows[:,3][startCM:endCM]
    magY = rows[:,4][startCM:endCM]
    magZ = rows[:,5][startCM:endCM] 
    
    # Total magnetic field
    totalmag = TotalMag(magX,magY,magZ)

    # Horizontal magnetic field
    magH = HorizontalMag(magX,magY) 
    
    # DC shift
    if dc_shift:
        totalmag = filt.DC_Shift(totalmag,bkps)
        magH = filt.DC_Shift(magH,bkps)
        magX = filt.DC_Shift(magX,bkps)
        magY = filt.DC_Shift(magY,bkps)
        magZ = filt.DC_Shift(magZ,bkps)        
    
    # Rolling average
    if rollingave:
        # Calculate rolling average for each component
        totalmag = filt.RollingAverage(totalmag,window_size)
        magH = filt.RollingAverage(magH,window_size)
        magX = filt.RollingAverage(magX,window_size)
        magY = filt.RollingAverage(magY,window_size)
        magZ = filt.RollingAverage(magZ,window_size)
    
        # To match the length of date to the magnitudes, we need to parse the date
        n = (len(date)-(window_size - 1))/(window_size - 1)
        date = np.delete(date, np.arange(int(len(date)/(window_size-1)), date.size, int(n))) 
        
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
        
    # Remove outliers
    #totalmag = filt.Outliers(totalmag)
    #magH = filt.Outliers(magH)
    
    # Calculate magnitude
    totalmag = np.abs(totalmag)
    magH = np.abs(magH)
    
    return date, totalmag, magH, magX, magY, magZ

######################
### Split UTC date ###
######################

def SplitTime(date):
    """
    This function splits up the CrowdMag date that is in the format of YYYY-MM-DD hh:mm:ss into 
    years, months, days, hours, minutes, seconds, year-month-days, hours-minute-seconds, and time in seconds.
    
    Parameters
    ----------
    date : string, date in the format of YYYY-MM-DD hh:mm:ss

    Returns
    ----------
    year, month, day, hour, minute, second : numpy arrays
    year-month-day, hour-minute-second : strings in numpy arrays
    time in seconds : numpy arrays
    """
    
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

#######################################
### Plot magnetic field versus time ###
#######################################

def PlotBCrowdMag(filenameCM,
                  fieldtype = 'F',
                  startCM = 3, endCM = -1,
                  rollingave = False,
                  window_size = 10,
                  dc_shift = False,
                  bkps = None,
                  filter_signal = 'raw'):
    """
    Plotting the CrowdMag data of the chosen component of the magnetic field.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming
    rollingave : boolean, if True: calculates the rolling average
    window_size : int, default=10, window-size for the rolling average
    dc_shift : boolean, if True: data will be DC shifted to zero
    bkps : int, default=4, max number of predicted breaking points
    filter_signal : string, can be 'raw' (no filter, 
                                   'filtfilt' (scipy.filtfilt), 
                                   'savgol' (scipy.savgol_filter), 
                                   'ffthighfreq', 
                                   'fftbandpass', 
                                   'combo' (filtfilt and fftbandbpass)

    Returns
    ----------
    Plot of the magnetic field from CrowdMag data.
    """
    
    # Key:
    ##### fieldtype = 'F'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # Ignore warnings
    warnings.simplefilter(action='ignore', category=RuntimeWarning)    
    warnings.simplefilter(action='ignore', category=UserWarning)  
    
    # Date and magnetic field data (total,horizontal,vertical)
    date, totalmag, magH, magX, magY, magZ = ReadCSVCrowdMag(filenameCM,startCM,endCM,
                                                             rollingave,window_size,
                                                             dc_shift,bkps,
                                                             filter_signal)
    
    # Time frame
    starttime = date[0]
    endtime = date[-1]
    
    # Plot
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    plt.title("CrowdMag : {} - {}".format(starttime,endtime), fontsize=16)
    plt.xlabel("UTC time", fontsize=12)
    
    if fieldtype == 'F':
        # Total magnetic field 
        ax.plot(date,totalmag, label="Total Magnetic Field")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)    
    
    if fieldtype == 'H':        
        # Horizontal magnetic field       
        ax.plot(date,magH, label="Horizontal Magnetic Field")
        plt.ylabel("Horizontal Magnetic Field (nT)", fontsize=12)

    if fieldtype == 'X':        
        # Magnetic field - X component       
        ax.plot(date,magX, label="Magnetic Field - X component")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
    
    if fieldtype == 'Y':
        # Magnetic field - Y component
        ax.plot(date,magY, label="Magnetic Field - Y component")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        
    if fieldtype == 'Z':
        # Magnetic field - Z component
        ax.plot(date,magZ, label="Magnetic Field - Z component")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
 
    # Reduce the number of ticks for the x-axis
    xticks = ticker.MaxNLocator(10)
    ax.xaxis.set_major_locator(xticks)
    plt.xticks(fontsize = 8)
    plt.legend()
    plt.show()
    
################################
### Find index of given date ###
################################

def FindDate(filenameCM,startCM,endCM,finddate):
    """
    Finding the index (or row) of the given date for trimming the data.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming
    finddate : string, find the row this date starts on

    Returns
    ----------
    Prints out the row number.
    """
    
    # Change date to string
    finddate = str(finddate)        
    
    # Define all dates in the CrowdMag data
    date = ReadCSVCrowdMag(filenameCM,startCM,endCM)[0]
    
    # Define start and end date
    for d in range(len(date)):
        if date[d][:10] == finddate:
            index = d
            break
        else: 
            index = None
    if index == None:
        print("The date {} is not included in the dataset.".format(finddate))
    else: 
        print("The date {} starts on row {}.".format(finddate,index))
    