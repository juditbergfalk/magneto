#########################
##### CROWDMAG DATA #####
#########################

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
from datetime import datetime
import ruptures as rpt
import os
from ipywidgets import Dropdown, Box, Layout, Label
from scipy.special import erfc

#################################
### Dropdown menu of filelist ###
#################################

# Make a list of filenames from the crowdmag folder
filelist = os.listdir('./data/crowdmag/')                   

# Create a dropdown menu and label
form_item_layout = Layout(display='flex', flex_flow='row', justify_content='space-between')
dropdownmenu = Dropdown(options=filelist, value='crowdmag_March 14 2022_iPhone12,1_2022-03-14 135556.csv')
form_items = [Box([Label(value='CrowdMag file: '),dropdownmenu], layout=form_item_layout)]

# Call function for dropdown menu
CrowdMagFileList = Box(form_items, layout=Layout(display='flex', flex_flow='column', 
                                                 border='solid 2px', align_items='stretch', width='50%'))

####################################
### Calculating rolling averages ###
####################################

def RollingAverage(array, 
                   window_size = 10):
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

################
### Outliers ###
################

def Outliers(signal):
    """
    Find outliers. Points need to be evaluated using Chauvenet's criterion.
    
    Parameters
    ----------
    signal : numpy array

    Returns
    ----------
    signal without outliers : numpy array    
    
    """
    
    # Find outliers
    meansignal = np.mean(signal)                      # Mean of signal
    stdsignal = np.std(signal)                        # Standard deviation of signal
    chauvenet_crit = 1.0/(2*len(signal))              # Chauvenet's criterion
    residuals = abs(signal - meansignal) / stdsignal  # Distance of a value to mean in stdv's
    prob = erfc(residuals)                            # Area normal distribution    
    outliers = prob < chauvenet_crit                  # List of boolean. True means there is an outlier
    
    # Replace outlier with mean value of the signal
    for i in range(len(signal)):
        if outliers[i]:
            signal[i] = meansignal 
            
    return signal

##########################
### Standardize signal ###
##########################

def Standardize(signal):
    """
    Standardize (Z-score normalization) time-series signal.
    
    Parameters
    ----------
    signal : numpy array

    Returns
    ----------
    standardized signal : numpy array    
    
    """
    
    meansignal = np.mean(signal)                                   # Calculate the mean of signal
    stdsignal = np.std(signal)                                     # Calculate the standard deviation of signal
    standardized_signal = (signal - meansignal) / stdsignal        # Standardize signal (Z-score normalization)
    
    return standardized_signal

################
### DC Shift ###
################

def DC_Shift(mag, 
             bkps = 10):
    """
    Calculate change points (breaking points) in a signal, break it up to segments, 
    standardize each segment and return new array of DC shifted standardized array.
    
    Parameters
    ----------
    mag : numpy array, magnetic field strength measurements
    bkps : int, default=10, max number of predicted breaking points

    Returns
    ----------
    standardized and DC shifted magnetic field strength data : numpy array    
    
    """
    
    # Change point detection
    model = "l2"  # "l1", "rbf", "linear", "normal", "ar"      # Segment model
    algo = rpt.Window(width = 40, model = model).fit(mag)      # Window sliding method to find breaking points
    my_bkps = algo.predict(n_bkps = bkps - 1)                  # Predict the location of breaking points in the signal
    print("Breaking points = {}".format(my_bkps))   

    # Define segments using the breaking points
    start = 0                                                  # Starting point of the data set
    segmented_list = []                                        # Empty list for the segmented lists
    for i in range(len(mag)):                                  # Loop through the magnetic field strength measurements
        for p in my_bkps:                                      # But also loop through the breaking point list
            if i == p:                                         # If the index equals the breaking point
                segment = mag[start:p]                         # Define a new segment from start point to the breaking point
                segmented_list.append(segment)                 # Append this new segment to the segmented list
                start = p                                      # Redefine the starting point
    last_segment = mag[start:]                                 # Define last segment
    segmented_list.append(last_segment)                        # Append last segment to the segmented list
    segmented_list = np.array(segmented_list,dtype=object)     # Convert list to numpy array
    
    # Standardize each segment
    norm_mag = []                                              # Empty list for the normalized magnitude
    for s in segmented_list:                                   # Loop through each segment
        stand_s = Standardize(s)                               # Standardize each segment (Z-score normalization)
        norm_mag.append(stand_s)                               # Append standardized signal to the normalized magnitude list
    norm_mag = np.array(norm_mag,dtype=object)                 # Convert list to numpy array
    norm_mag = np.concatenate(norm_mag,axis=0)                 # Concatenate the whole sequence of arrays
    
    return norm_mag

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
                    dc_shift = False,
                    standardize = False,
                    bkps = 10):
    """
    Read CrowdMag .csv files and return the arrays for date and X,Y,Z component of magnetic field.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming
    rollingave : boolean, do you want to calculate the rolling average?
    dc_shift : boolean, does the data need to be DC shifted? 
               Note: DC shift includes standardization so if this is True, standarize cannot be True!
    standardize : boolean, does the data need to be standardized (Z-score normalization)?
    bkps : int, default=10, max number of predicted breaking points

    Returns
    ----------
    date,magX,magY,magZ : numpy arrays
    """
    
    # Read in .csv file
    data = pd.read_csv(filenameCM)#, parse_dates=['Time (UTC)'])
    
    # Selecting all rows
    rows = np.array(data.loc[:])
    
    # Defining all relevant columns, define start and end points for easier trimming
    date = rows[:,0][startCM:endCM]
    magX = rows[:,3][startCM:endCM]
    magY = rows[:,4][startCM:endCM]
    magZ = rows[:,5][startCM:endCM]
    
    # Change to magnitude of the magnetic field
    #magX = abs(magX)
    #magY = abs(magY)
    #magZ = abs(magZ)
    
    # Total magnetic field
    totalmag = TotalMag(magX,magY,magZ)

    # Horizontal magnetic field
    magH = HorizontalMag(magX,magY) 
    
    # Rolling average
    if rollingave:
        # Calculate rolling average for each component
        window_size = 10 
        totalmag = RollingAverage(totalmag,window_size)
        magH = RollingAverage(magH,window_size)
        magX = RollingAverage(magX,window_size)
        magY = RollingAverage(magY,window_size)
        magZ = RollingAverage(magZ,window_size)
    
        # To match the length of date to the magnitudes, we need to parse the date
        n = (len(date)-(window_size - 1))/(window_size - 1)
        date = np.delete(date, np.arange(int(len(date)/(window_size-1)), date.size, int(n)))     
    
    # DC shift
    if dc_shift:
        totalmag = DC_Shift(totalmag,bkps)
        magH = DC_Shift(magH,bkps)
        magX = DC_Shift(magX,bkps)
        magY = DC_Shift(magY,bkps)
        magZ = DC_Shift(magZ,bkps)
        
    # If not DC shifted, then standardize (Z-score normalization)    
    if standardize:
        totalmag = Standardize(totalmag)
        magH = Standardize(magH)
        magX = Standardize(magX)
        magY = Standardize(magY)
        magZ = Standardize(magZ)
        
    # Remove outliers
    totalmag = Outliers(totalmag)
    magH = Outliers(magH)
    
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
        #dtseconds = (dt-datetime(1970,1,1)).total_seconds()
        dtseconds = (dt-datetime(1,1,1)).total_seconds()
        timeinseconds.append(dtseconds)
    timeinseconds = np.array(timeinseconds)
    
    return year,month,day,hour,minute,second,ymd,hms,timeinseconds

#######################################
### Plot magnetic field versus time ###
#######################################

def PlotBCrowdMag(filenameCM,
                  fieldtype = 'T',
                  startCM = 3, endCM = -1,
                  rollingave = False,
                  dc_shift = False,
                  standardize = False,
                  bkps = 10):
    """
    Plotting the CrowdMag data of the chosen component of the magnetic field.
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming
    endCM : int, default=-1 (last element), ending row for trimming
    rollingave : boolean, do you want to calculate the rolling average?
    dc_shift : boolean, does the data need to be DC shifted (note: this includes standardization!)?
    standardize : boolean, does the data need to be standardized (Z-score normalization)?
    bkps : int, default=10, max number of predicted breaking points

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
    date, totalmag, magH, magX, magY, magZ = ReadCSVCrowdMag(filenameCM,startCM,endCM,rollingave,dc_shift,standardize,bkps)
    
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
        ax.plot(date,totalmag, label="Total Magnetic Field")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)    
        limit = np.max(totalmag[:-3])/50                                   # Set limit for axes
        plt.ylim(np.min(totalmag[:-3])-limit,np.max(totalmag[:-3])+limit)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field       
        ax.plot(date,magH, label="Horizontal Magnetic Field")
        plt.ylabel("Horizontal Magnetic Field (nT)", fontsize=12)
        limit = np.max(magH[:-3])/50                                       # Set limit for axes
        plt.ylim(np.min(magH[:-3])-limit,np.max(magH[:-3])+limit)
    
    if fieldtype == 'X':        
        # Magnetic field - X component       
        ax.plot(date,magX, label="Magnetic Field - X component")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
        limit = np.max(magX[:-3])/50                                       # Set limit for axes
        plt.ylim(np.min(magX[:-3])-limit,np.max(magX[:-3])+limit)
    
    if fieldtype == 'Y':
        # Magnetic field - Y component
        ax.plot(date,magY, label="Magnetic Field - Y component")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        limit = np.max(magY[:-3])/50                                       # Set limit for axes
        plt.ylim(np.min(magY[:-3])-limit,np.max(magY[:-3])+limit)
        
    if fieldtype == 'Z':
        # Magnetic field - Z component
        ax.plot(date,magZ, label="Magnetic Field - Z component")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        limit = np.max(magZ[:-3])/50                                       # Set limit for axes
        plt.ylim(np.min(magZ[:-3])-limit,np.max(magZ[:-3])+limit)
 
    # Reduce the number of ticks for the x-axis
    xticks = ticker.MaxNLocator(10)
    ax.xaxis.set_major_locator(xticks)
    plt.xticks(fontsize = 8)
    plt.legend()
    plt.show()