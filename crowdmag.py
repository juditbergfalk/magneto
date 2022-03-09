#########################
##### CROWDMAG DATA #####
#########################

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from datetime import datetime

#################################
### Dropdown menu of filelist ###
#################################

# Import modules
import os
from ipywidgets import Dropdown, Box, Layout, Label

# Make a list of filenames from the crowdmag folder
filelist = os.listdir('./data/crowdmag/')                   

# Create a dropdown menu and label
form_item_layout = Layout(display='flex', flex_flow='row', justify_content='space-between')
dropdownmenu = Dropdown(options=filelist, value='crowdmag_2-26-22_iPhone12,1_2022-02-15 170458.csv')
form_items = [Box([Label(value='CrowdMag file: '),dropdownmenu], layout=form_item_layout)]

# Call function for dropdown menu
CrowdMagFileList = Box(form_items, layout=Layout(display='flex', flex_flow='column', 
                                                 border='solid 2px', align_items='stretch', width='50%'))

####################################
### Calculating rolling averages ###
####################################

def RollingAverage(array, window_size=10):
    """
    Calculate the rolling/moving average of a numpy array, given the window-size.
    
    Parameters
    ----------
    array : numpy array
    window_size : int, default=10, window-size for the rolling average

    Returns
    ----------
    rollave : numpy array, rolling average of an array
    """    
    
    # Convert array of integers to pandas series
    numbers_series = pd.Series(array)

    # Get the window of series of observations of specified window size
    windows = numbers_series.rolling(window_size)

    # Create a series of moving averages of each window
    moving_averages = windows.mean()

    # Convert pandas series back to list
    moving_averages_list = moving_averages.tolist()

    # Remove null entries from the list
    rollave = moving_averages_list[window_size - 1:]
    
    # Change the series back to numpy array
    rollave = np.array(rollave)
    
    return rollave

###################
### Read in csv ###
###################

def ReadCSVCrowdMag(filenameCM,startCM=3,endCM=-1):
    """
    Read CrowdMag .csv files and return the arrays for date and X,Y,Z component of magnetic field.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming

    Returns
    ----------
    date,magX,magY,magZ : numpy arrays
    """
    
    # Read in .csv file
    data = pd.read_csv(filenameCM)
    
    # Selecting all rows
    rows = np.array(data.loc[:])
    
    # Defining all relevant columns, define start and end points for easier trimming
    date = rows[:,0][startCM:endCM]
    magX = rows[:,3][startCM:endCM]
    magY = rows[:,4][startCM:endCM]
    magZ = rows[:,5][startCM:endCM]
    
    # Change to magnitude of the magnetic field
    magX = np.abs(magX)
    magY = np.abs(magY)
    magZ = np.abs(magZ)
    
    # Calculate rolling average for each component
    window_size = 10
    magX = RollingAverage(magX, window_size)
    magY = RollingAverage(magY, window_size)
    magZ = RollingAverage(magZ, window_size)
    
    # To match the length of date to the magnitudes, we need to parse the date
    n = len(date) / (window_size - 2)                # Delete exactly window_size - 2 element
    date = np.delete(date, slice(None, None, int(n))) 
    
    return date,magX,magY,magZ

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
        #dtseconds = (dt-datetime(1970,1,1)).total_seconds()
        dtseconds = (dt-datetime(1,1,1)).total_seconds()
        timeinseconds.append(dtseconds)
    timeinseconds = np.array(timeinseconds)
    
    return year,month,day,hour,minute,second,ymd,hms,timeinseconds

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

#######################################
### Plot magnetic field versus time ###
#######################################

def PlotBCrowdMag(filenameCM,fieldtype='T',startCM=3,endCM=-1):
    """
    Plotting the CrowdMag data of the chosen component of the magnetic field.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming

    Returns
    ----------
    Plot of the magnetic field from CrowdMag data.
    """
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
        
    # Date and Magnetic field data (x,y,z)
    date, magX, magY, magZ = ReadCSVCrowdMag(filenameCM,startCM,endCM)
    
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
        ax.plot(date,totalmag, label="Total Magnetic Field")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
        plt.ylim(np.min(totalmag[:-3])-500,np.max(totalmag[:-3])+500)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field
        horizontalmag = HorizontalMag(magX,magY)          
        ax.plot(date,horizontalmag, label="Horizontal Magnetic Field")
        plt.ylabel("Horizontal Magnetic Field (nT)", fontsize=12)
        plt.ylim(np.min(horizontalmag[:-3])-500,np.max(horizontalmag[:-3])+500)
    
    if fieldtype == 'X':        
        # Magnetic field - X component       
        ax.scatter(date,magX, label="Magnetic Field - X component")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
        plt.ylim(np.min(magX[:-3])-500,np.max(magX[:-3])+500)
    
    if fieldtype == 'Y':
        # Magnetic field - Y component
        ax.plot(date,magY, label="Magnetic Field - Y component")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        plt.ylim(np.min(magY[:-3])-500,np.max(magY[:-3])+500)
        
    if fieldtype == 'Z':
        # Magnetic field - Z component
        ax.plot(date,magZ, label="Magnetic Field - Z component")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        plt.ylim(np.min(magZ[:-3])-500,np.max(magZ[:-3])+500)
 
    # Reduce the number of ticks for the x-axis
    xticks = ticker.MaxNLocator(10)
    ax.xaxis.set_major_locator(xticks)
    plt.xticks(fontsize = 8)
    plt.legend()
    plt.show()