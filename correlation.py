import crowdmag as cm
import geomag as gm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline      # Interpolation (spline) function
import lmfit as lm                                              # Fitting
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import warnings

#######################################
### Overlay Plot CrowdMag vs GeoMag ###
#######################################

def PlotOverlay2Data(filenameCM,
                     observatory = 'BRW',
                     fieldtype = 'T',
                     startCM = 3, endCM = -1, startGM = 0, endGM = -1,
                     download = True,
                     timeshift = 0,
                     rollingave = False,
                     dc_shift = False,
                     standardizeCM = False, standardizeGM = False,
                     bkps = 10):
    """
    Overlaying the two CrowdMag and GeoMag data sets of the chosen component of the magnetic field with an adjustable timeshift. 
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    observatory : string, default=Barrow Observatory, code for the observatory
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming CrowdMag data
    endCM : int, default=-1 (last element), ending row for trimming CrowdMag data
    startGM : int, default=0, starting row for triming GeoMag data
    endGM : int, default=-1 (last element), ending row for trimming GeoMag data
    download : boolean, default=True, do you want to download the .csv file from GeoMag (True) 
                or is it already downloaded (False)
    timeshift : int, default=0, shifting the CrowdMag dataset to the right, timeshift in seconds
    rollingave : boolean, do you want to calculate the rolling average for the CrowdMag data?
    dc_shift : boolean, does the CrowdMag data need to be DC shifted (note: this includes standardization!)?
    standardizeCM : boolean, does the CrowdMag data need to be standardized (Z-score normalization)?
    standardizeGM : boolean, does the GeoMag data need to be standardized (Z-score normalization)?
    bkps : int, default=10, max number of predicted breaking points

    Returns
    ----------
    Overlay plot of the CrowdMag and GeoMag data sets.   
    """
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # Make sure that standardization doesn't happen twice
    if dc_shift:
        standardizeCM = False    
    
    # Change timeshift to seconds
    timeshift = int(timeshift * 60)
    
    # CrowdMag data
    CMdate,Cmtotalmag,CMmagH,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM,
                                                                       rollingave,dc_shift,standardizeCM,bkps)
    
    # Time in seconds 
    CMtimesec = cm.SplitTime(CMdate)[8] - cm.SplitTime(CMdate)[8][0]
    
    # Start time in seconds
    CMstarttime = cm.SplitTime(CMdate)[8][0]
    
    # GeoMag data
    GMdate,GMtime,GMdoy,GMmagX,GMmagY,GMmagZ,GMmagH,GMtimesec,GMlocation = gm.DefineAllComponents(filenameCM,observatory,
                                                                                   startCM,endCM,startGM,endGM,
                                                                                   download,standardizeGM)
    GMtotalmag = cm.TotalMag(GMmagX,GMmagY,GMmagZ)
    
    # Fuse date and time to match CrowdMag date
    GMdatetime = []
    for t in range(len(GMdate)):
        dt = GMdate[t] + ' ' + GMtime[t][0:8]
        GMdatetime.append(dt)
    GMdatetime = np.array(GMdatetime)
    
    # Time in seconds 
    GMtimesec = cm.SplitTime(GMdatetime)[8] - cm.SplitTime(GMdatetime)[8][0]
    
    # Start time in seconds
    GMstarttime = cm.SplitTime(GMdatetime)[8][0]
    
    # Time frame
    starttime = CMdate[0]
    endtime = CMdate[-1] 
    
    # Plot
    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot()
    plt.title("CrowdMag vs GeoMag : {} - {}".format(starttime,endtime), fontsize=16)
    plt.xlabel("Time (seconds)", fontsize=12)
    
    if fieldtype == 'T':
        # Total magnetic field
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMtotalmag)/np.mean(CMtotalmag)
        ax.plot(CMtimesec,scale*CMtotalmag, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMtotalmag, label="GeoMag data, Observatory : {}".format(observatory))
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagH)/np.mean(CMmagH)
        ax.plot(CMtimesec,scale*CMmagH, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagH, label="GeoMag data, Observatory : {}".format(observatory))
        plt.ylabel("Magnetic Field - H (nT)", fontsize=12)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagX)/np.mean(CMmagX)
        ax.plot(CMtimesec,scale*CMmagX, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagX, label="GeoMag data, Observatory : {}".format(observatory))
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagY)/np.mean(CMmagY)
        ax.plot(CMtimesec,scale*CMmagY, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagY, label="GeoMag data, Observatory : {}".format(observatory))
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagZ)/np.mean(CMmagZ)
        ax.plot(CMtimesec,scale*CMmagZ, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagZ, label="GeoMag data, Observatory : {}".format(observatory))
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        
    plt.legend()
    plt.show()
    
#######################
### Spline function ###
#######################

def SplineFunction(x,y):
    """
    Spline (interpolation) function using scipy's InterpolatedUnivariateSpline module. 
    
    Parameters
    ----------
    x,y : numpy arrays

    Returns
    ----------
    Spline of degree 3.
    """
    return InterpolatedUnivariateSpline(x,y,k=3)

def SplineData(t,date,magfield):
    """
    Spline (interpolation) time vs magnetic field and return an array for new sampling times (t). 
    
    Parameters
    ----------
    t, date, magfield : numpy arrays

    Returns
    ----------
    Numpy array of magnetic field over given sampling times.
    """
    mag = SplineFunction(date,magfield)
    return mag(t)   

###############################    
### Correlation Coefficient ###
###############################

def CorrelationCoefficient(x,y):
    """Calculating correlation coefficient of two data sets.    
    
    Parameters
    ----------
    x,y : numpy arrays of two data sets with the same length.

    Returns
    ----------
    r : int, correlation coefficient of the two data sets.
    """
    
    # Mean
    meanx = np.mean(x)
    meany = np.mean(y)
    
    # Numerator
    numerator = np.sum((x - meanx)*(y - meany))
    
    # Denominator
    denominator1 = np.sum((x - meanx)**2)
    denominator2 = np.sum((y - meany)**2)
    
    # Compute r
    r = numerator / np.sqrt(denominator1 * denominator2)
    
    return r
    
###############    
### Fitting ###
###############

# Define linear fitting function
def LinearFunction(x,a,b):
    """Basic linear function.    
    
    Parameters
    ----------
    x : numpy array
    a,b : int, slope (a) and intercept (b)

    Returns
    ----------
    Numpy array of a linear function.
    """
    return a * x + b

# Polyfit fitting
def FittingPolyfit(data1,data2):
    """Fitting using numpy's polyfit.    
    
    Parameters
    ----------
    data1, data2 : numpy arrays of two data sets of the same length.

    Returns
    ----------
    A,B : int, slope and intercept of the fitted linear function.
    """
    
    fit = np.polyfit(data1,data2,1)
    B = fit[1]
    A = fit[0]
    
    return A,B

################################################
### Scatter Plot of CrowdMag and GeoMag data ###
################################################

def ScatterPlot(filenameCM,
                observatory = 'BRW',
                fieldtype = 'T',
                startCM = 3, endCM = -1, startGM = 0, endGM = -1,
                download = True,
                timeshift = 0,
                rollingave = False,
                dc_shift = False,
                standardizeCM = False, standardizeGM = False,
                bkps = 10):
    """
    Scatter plot of the CrowdMag and GeoMag datasets after interpolating and defining new sampling times, 
    then fitting a linear curve and calculating fitting parameters. 
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    observatory : string, default=Barrow Observatory, code for the observatory
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming CrowdMag data
    endCM : int, default=-1 (last element), ending row for trimming CrowdMag data
    startGM : int, default=0, starting row for triming GeoMag data
    endGM : int, default=-1 (last element), ending row for trimming GeoMag data
    download : boolean, default=True, do you want to download the .csv file from GeoMag (True) 
                or is it already downloaded (False)
    timeshift : int, default=0, shifting the CrowdMag dataset to the right
    rollingave : boolean, do you want to calculate the rolling average for the CrowdMag data?
    dc_shift : boolean, does the CrowdMag data need to be DC shifted (note: this includes standardization!)?
    standardizeCM : boolean, does the CrowdMag data need to be standardized (Z-score normalization)?
    standardizeGM : boolean, does the GeoMag data need to be standardized (Z-score normalization)?
    bkps : int, default=10, max number of predicted breaking points

    Returns
    ----------
    Scatter plot with fitted linear curve. 
    Printed correlation coefficient, slope and intercept of fitted linear function, chi-squared and reduced chi-squared value.
    """
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # Make sure that standardization doesn't happen twice
    if dc_shift:
        standardizeCM = False
    
    # Ignore warnings
    warnings.simplefilter(action='ignore', category=RuntimeWarning)      
    
    # Change timeshift to match the column from heatmap
    timeshift = int(timeshift * 60/70)     # CrowdMag is 70-sec, GeoMag is 60-sec intervals
        
    # CrowdMag data
    CMdate,Cmtotalmag,CMmagH,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM,
                                                                       rollingave,dc_shift,standardizeCM,bkps)
    
    # Trim data
    if timeshift != 0:
        CMdate = CMdate[timeshift:]
        CMtotalmag = CMmagX[timeshift:]
        CMmagH = CMmagX[timeshift:]
        CMmagX = CMmagX[timeshift:]
        CMmagY = CMmagY[timeshift:]
        CMmagZ = CMmagZ[timeshift:]    
    
    # Time in seconds 
    CMtimesec = cm.SplitTime(CMdate)[8] - cm.SplitTime(CMdate)[8][0]
    
    # Spline CrowdMag data
    CMtotalmagSpline = lambda t: SplineData(t,CMtimesec,CMtotalmag)
    CMmagHSpline = lambda t: SplineData(t,CMtimesec,CMmagH)
    CMmagXSpline = lambda t: SplineData(t,CMtimesec,CMmagX)
    CMmagYSpline = lambda t: SplineData(t,CMtimesec,CMmagY)
    CMmagZSpline = lambda t: SplineData(t,CMtimesec,CMmagZ)
    
    # GeoMag data
    GMdate,GMtime,GMdoy,GMmagX,GMmagY,GMmagZ,GMmagH,GMtimesec,GMlocation = gm.DefineAllComponents(filenameCM,observatory,
                                                                                   startCM,endCM,startGM,endGM,
                                                                                   download,standardizeGM)
    
    # Trim data
    if timeshift != 0:
        GMdate = GMdate[0:-timeshift]
        GMtime = GMtime[0:-timeshift]
        GMdoy = GMdoy[0:-timeshift]
        GMmagX = GMmagX[0:-timeshift]
        GMmagY = GMmagY[0:-timeshift]
        GMmagZ = GMmagZ[0:-timeshift]
        GMmagH = GMmagH[0:-timeshift]
        GMtimesec = GMtimesec[0:-timeshift]
        GMlocation = GMlocation[0:-timeshift]    
    
    # Calculate total magnetic field
    GMtotalmag = cm.TotalMag(GMmagX,GMmagY,GMmagZ)
    
    # Fuse date and time to match CrowdMag date
    GMdatetime = []
    for t in range(len(GMdate)):
        dt = GMdate[t] + ' ' + GMtime[t][0:8]
        GMdatetime.append(dt)
    GMdatetime = np.array(GMdatetime)
    
    # Time in seconds 
    GMtimesec = cm.SplitTime(GMdatetime)[8] - cm.SplitTime(GMdatetime)[8][0]
    
    # Spline GeoMag data
    GMtotalmagSpline = lambda t: SplineData(t,GMtimesec,GMtotalmag)
    GMmagHSpline = lambda t: SplineData(t,GMtimesec,GMmagH)
    GMmagXSpline = lambda t: SplineData(t,GMtimesec,GMmagX)
    GMmagYSpline = lambda t: SplineData(t,GMtimesec,GMmagY)
    GMmagZSpline = lambda t: SplineData(t,GMtimesec,GMmagZ) 
    
    # Define time interval
    time = np.linspace(0,np.max(CMtimesec),len(CMdate))
        
    # Time frame
    starttime = CMdate[0]
    endtime = CMdate[-1]
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.suptitle("CrowdMag vs GeoMag Scatter Plot", fontsize=12)
    plt.title("{} - {}".format(starttime,endtime), fontsize=12)
    
    if fieldtype == 'T':
        # Total magnetic field
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMtotalmag)/np.mean(CMtotalmag)
        plt.scatter(scale*CMtotalmagSpline(time),GMtotalmagSpline(time))
        plt.xlabel("CrowdMag - Total Magnetic Field (nT), scaled by {:.3f}".format(scale), fontsize=12)
        plt.ylabel("GeoMag - Total Magnetic Field (nT)", fontsize=12)
        cmdata = scale*CMtotalmagSpline(time)
        gmdata = GMtotalmagSpline(time)
        
    if fieldtype == 'H':        
        # Horizontal magnetic field
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagH)/np.mean(CMmagH) 
        plt.scatter(scale*CMmagHSpline(time),GMmagHSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - Horizontal (nT), scaled by {:.3f}".format(scale), fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Horizontal (nT)", fontsize=12)
        cmdata = scale*CMmagHSpline(time)
        gmdata = GMmagHSpline(time) 
    
    if fieldtype == 'X':        
        # Magnetic field - X direction
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagX)/np.mean(CMmagX) 
        plt.scatter(scale*CMmagXSpline(time),GMmagXSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - X (nT), scaled by {:.3f}".format(scale), fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - X (nT)", fontsize=12)
        cmdata = scale*CMmagXSpline(time)
        gmdata = GMmagXSpline(time)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagY)/np.mean(CMmagY) 
        plt.scatter(scale*CMmagYSpline(time),GMmagYSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - Y (nT), scaled by {:.3f}".format(scale), fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Y (nT)", fontsize=12)
        cmdata = scale*CMmagYSpline(time)
        gmdata = GMmagYSpline(time)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        if standardizeGM:
            scale = 1
        else:
            scale = np.mean(GMmagZ)/np.mean(CMmagZ) 
        plt.scatter(scale*CMmagZSpline(time),GMmagZSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - Z (nT), scaled by {:.3f}".format(scale), fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Z (nT)", fontsize=12)
        cmdata = scale*CMmagZSpline(time)
        gmdata = GMmagZSpline(time)
    
    # Calclate correlation coefficient
    r = CorrelationCoefficient(cmdata,gmdata)
    # Fitting: polyfit
    x = np.linspace(np.min(cmdata),np.max(cmdata),len(CMdate))
    slope_polyfit,intercept_polyfit = FittingPolyfit(cmdata,gmdata)
    # Residuals
    residuals = gmdata - LinearFunction(x,slope_polyfit,intercept_polyfit)
    # Chi squared and reduced chi squared
    std = np.std(gmdata)
    chisquared = np.sum(residuals**2/std**2)   # not sure about the errors!!
    # Degrees of freedom = number of data points - number of fitting parameters
    dof = len(CMdate) - 2    
    reducedchisquared = chisquared / dof
    
    plt.plot(x,LinearFunction(x,slope_polyfit,intercept_polyfit), label="Linear Fit", color='r')
    plt.legend()
    plt.show()
    print("Correlation coefficient of the two datasets is {:.4f}.".format(r))
    print("Slope = {}".format(slope_polyfit))
    print("Intercept = {}".format(intercept_polyfit))  
    print("Chi-squared = {}".format(chisquared))
    print("Reduced chi-squared = {}".format(reducedchisquared))
    
    # Pearson correlation testing
    stack = np.vstack((cmdata,gmdata)).T    
    stackpd = pd.DataFrame(stack, columns = ['CrowdMag','GeoMag'])
    overall_pearson_r = stackpd.corr().iloc[0,1]
    print(f"Pandas computed Pearson r: {overall_pearson_r:.3g}")
    
    r, p = stats.pearsonr(stackpd.dropna()['CrowdMag'], stackpd.dropna()['GeoMag'])
    print(f"Scipy computed Pearson r: {r:.3g} and p-value: {p:.3g}")

    f, ax = plt.subplots(figsize=(14,3))
    stackpd.rolling(window = 10, center = True).median().plot(ax=ax)
    ax.set(xlabel='Time (min)',ylabel='Magnetic Field - H (nT)',
           title=f"Overall Pearson r = {overall_pearson_r:.2g}");
    
#####################################
### Time-lagged Cross Correlation ###
#####################################

def TLCC(x,y, lag=0):
    
    """ Time-lagged Cross Correlation calculation. 
    Shifting data and calculating correlation coefficient.
    
    Parameters
    ----------
    lag : int, default=0
    x,y : pandas.Series objects of equal length

    Returns
    ----------
    correlation coefficient : float
    """
    return x.corr(y.shift(lag))

####################################################    
### Rolling window time-lagged cross correlation ###
####################################################

def RWTLCC(filenameCM,
           observatory = 'BRW',
           fieldtype = 'T',
           startCM = 3, endCM = -1, startGM = 0, endGM = -1,
           download = True,
           n = 500, windowsize = 300, step = 100,
           rollingave = False,
           dc_shift = False,
           standardizeCM = False, standardizeGM = False,
           bkps = 10): 
    """
    A heat-map of the rolling window time-lagged cross correlation (RWTLCC) for the CrowdMag and GeoMag data sets. 
    
    Parameters
    ----------
    filenameCM : string, CrowdMag .csv filename
    observatory : string, default=Barrow Observatory, code for the observatory
    fieldtype : string, default=total, component of the magnetic field
    startCM : int, default=3, starting row for trimming CrowdMag data
    endCM : int, default=-1 (last element), ending row for trimming CrowdMag data
    startGM : int, default=0, starting row for triming GeoMag data
    endGM : int, default=-1 (last element), ending row for trimming GeoMag data
    download : boolean, default=True, do you want to download the .csv file from GeoMag (True) 
                or is it already downloaded (False)
    n : int, default=500, time-lag range is -n to n+1
    windowsize : int, default=300, chunks of the data set that is calculated at a time, number of samples
    step : int, default=100, stepsize of the loop
    rollingave : boolean, do you want to calculate the rolling average for the CrowdMag data?
    dc_shift : boolean, does the CrowdMag data need to be DC shifted (note: this includes standardization!)?
    standardizeCM : boolean, does the CrowdMag data need to be standardized (Z-score normalization)?
    standardizeGM : boolean, does the GeoMag data need to be standardized (Z-score normalization)?
    bkps : int, default=10, max number of predicted breaking points

    Returns
    ----------
    Heat-map of the RWTLCC.
    """
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # Make sure that standardization doesn't happen twice
    if dc_shift:
        standardizeCM = False
    
    # Ignore warnings
    warnings.simplefilter(action='ignore', category=RuntimeWarning)      
    
    # CrowdMag data
    CMdate,Cmtotalmag,CMmagH,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM,
                                                                       rollingave,dc_shift,standardizeCM,bkps)
    
    # Time in seconds 
    CMtimesec = cm.SplitTime(CMdate)[8] - cm.SplitTime(CMdate)[8][0]
    
    # Spline Crowdmag data
    CMtotalmagSpline = lambda t: SplineData(t,CMtimesec,CMtotalmag)
    CMmagHSpline = lambda t: SplineData(t,CMtimesec,CMmagH)
    CMmagXSpline = lambda t: SplineData(t,CMtimesec,CMmagX)
    CMmagYSpline = lambda t: SplineData(t,CMtimesec,CMmagY)
    CMmagZSpline = lambda t: SplineData(t,CMtimesec,CMmagZ)
    
    # GeoMag data
    GMdate,GMtime,GMdoy,GMmagX,GMmagY,GMmagZ,GMmagH,GMtimesec,GMlocation = gm.DefineAllComponents(filenameCM,observatory,
                                                                                   startCM,endCM,startGM,endGM,
                                                                                   download,standardizeGM)
    GMtotalmag = cm.TotalMag(GMmagX,GMmagY,GMmagZ)
    
    # Fuse date and time to match CrowdMag date
    GMdatetime = []
    for t in range(len(GMdate)):
        dt = GMdate[t] + ' ' + GMtime[t][0:8]
        GMdatetime.append(dt)
    GMdatetime = np.array(GMdatetime)
    
    # Time in seconds 
    GMtimesec = cm.SplitTime(GMdatetime)[8] - cm.SplitTime(GMdatetime)[8][0]
    
    # Spline Crowdmag data
    GMtotalmagSpline = lambda t: SplineData(t,GMtimesec,GMtotalmag)
    GMmagHSpline = lambda t: SplineData(t,GMtimesec,GMmagH)
    GMmagXSpline = lambda t: SplineData(t,GMtimesec,GMmagX)
    GMmagYSpline = lambda t: SplineData(t,GMtimesec,GMmagY)
    GMmagZSpline = lambda t: SplineData(t,GMtimesec,GMmagZ)
    
    # Time frame
    starttime = CMdate[0]
    endtime = CMdate[-1] 
    
    # Define time interval
    time = np.linspace(0,np.max(GMtimesec),len(GMtotalmag))
    
    # Plot    
    if fieldtype == 'T':
        # Total magnetic field
        if standardizeCM:
            scale = 1
        else:
            scale = np.mean(GMtotalmag)/np.mean(CMtotalmag)
        stack = (np.vstack((scale*CMtotalmagSpline(time), GMtotalmagSpline(time))).T)
        
    if fieldtype == 'H':        
        # Horizontal magnetic field 
        if standardizeCM:
            scale = 1
        else:
            scale = np.mean(GMmagH)/np.mean(CMmagH)
        stack = (np.vstack((scale*CMmagHSpline(time), GMmagHSpline(time))).T)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        if standardizeCM:
            scale = 1
        else:
            scale = np.mean(GMmagX)/np.mean(CMmagX)
        stack = (np.vstack((scale*CMmagXSpline(time), GMmagXSpline(time))).T)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        if standardizeCM:
            scale = 1
        else:
            scale = np.mean(GMmagY)/np.mean(CMmagY)
        stack = (np.vstack((scale*CMmagYSpline(time), GMmagYSpline(time))).T)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        if standardizeCM:
            scale = 1
        else:
            scale = np.mean(GMmagZ)/np.mean(CMmagZ)
        stack = np.vstack((scale*CMmagZSpline(time), GMmagZSpline(time))).T
    
    # Calculate correlation for each window
    stackdatasets = pd.DataFrame(stack, columns = ['CrowdMag','GeoMag'])
    t_start = 0                   # Start at the beginning of the data set
    t_end = t_start + windowsize  # Define the end of the chunksize
    rss = []
    while t_end < len(stackdatasets):
        dataCM = stackdatasets['CrowdMag'].iloc[t_start:t_end]
        dataGM = stackdatasets['GeoMag'].iloc[t_start:t_end]
        rs = [TLCC(dataCM,dataGM, lag) for lag in range(-int(n),int(n+1))]
        rss.append(rs)
        t_start = t_start + step
        t_end = t_end + step
    rss = pd.DataFrame(rss)
    
    # Find max value of all columns
    max_value = rss.max(numeric_only = True).max()
    index_max = rss.max().idxmax()     # Index of max correlation value
    max_column = rss[index_max]        # Column where the max correlation value is found
    index_max = index_max-n
    print("Max correlation value = {}".format(max_value))
    print("Column of max correlation value, or time shift in minutes = {}".format(index_max))
    
    # Plot heat-map of the correlation coefficients
    f, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(rss, cmap = 'RdBu_r', ax = ax)
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs')
    ax.set_xticklabels([int(i-n) for i in ax.get_xticks()]);