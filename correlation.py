import crowdmag as cm
import geomag as gm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline      # Interpolation (spline) function
import lmfit as lm                                              # Fitting
import pandas as pd
import seaborn as sns
import scipy.stats as stats

#######################################
### Overlay Plot CrowdMag vs GeoMag ###
#######################################

def PlotOverlay2Data(filenameCM,
                     observatory='BRW',
                     fieldtype='T',
                     startCM=3,endCM=-1,startGM=0,endGM=-1,
                     download=True,
                     timeshift=0):
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # CrowdMag data
    CMdate,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM)
    CMmagH = cm.HorizontalMag(CMmagX,CMmagY)
    CMtotalmag = cm.TotalMag(CMmagX,CMmagY,CMmagZ)
    
    # Time in seconds 
    CMtimesec = cm.SplitTime(CMdate)[8] - cm.SplitTime(CMdate)[8][0]
    
    # Start time in seconds
    CMstarttime = cm.SplitTime(CMdate)[8][0]
    
    # GeoMag data
    GMdate,GMtime,GMdoy,GMmagX,GMmagY,GMmagZ,GMmagH,GMtimesec,GMlocation = gm.DefineAllComponents(filenameCM,observatory,
                                                                                                  startCM,endCM,startGM,endGM,
                                                                                                  download)
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
        scale = np.mean(GMtotalmag)/np.mean(CMtotalmag)
        ax.plot(CMtimesec,scale*CMtotalmag, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMtotalmag, label="GeoMag data")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field 
        scale = np.mean(GMmagH)/np.mean(CMmagH)
        ax.plot(CMtimesec,scale*CMmagH, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagH, label="GeoMag data")
        plt.ylabel("Magnetic Field - H (nT)", fontsize=12)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        scale = np.mean(GMmagX)/np.mean(CMmagX)
        ax.plot(CMtimesec,scale*CMmagX, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagX, label="GeoMag data")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        scale = np.mean(GMmagY)/np.mean(CMmagY)
        ax.plot(CMtimesec,scale*CMmagY, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagY, label="GeoMag data")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        scale = np.mean(GMmagZ)/np.mean(CMmagZ)
        ax.plot(CMtimesec,scale*CMmagZ, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.plot(GMtimesec+timeshift,GMmagZ, label="GeoMag data")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        
    plt.legend()
    plt.show()
    

#######################
### Spline function ###
#######################

def SplineFunction(x,y):
    return InterpolatedUnivariateSpline(x,y,k=3)

def SplineData(t,date,magfield):
    mag = SplineFunction(date,magfield)   
    return mag(t)   

###############################    
### Correlation Coefficient ###
###############################

def CorrelationCoefficient(x,y):
    """Calculating correlation coefficient of two datasets."""
    
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
    return a * x + b

def FittingPolyfit(data1,data2):
    
    fit = np.polyfit(data1,data2,1)
    B = fit[1]
    A = fit[0]
    
    return A,B

################################################
### Scatter Plot of CrowdMag and GeoMag data ###
################################################

def ScatterPlot(filenameCM,
                observatory='BRW',
                fieldtype='T',
                startCM=3,endCM=-1,startGM=0,endGM=0,
                download=True,
                timeshift=0):
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # CrowdMag data
    CMdate,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM)
    CMmagH = cm.HorizontalMag(CMmagX,CMmagY)
    CMtotalmag = cm.TotalMag(CMmagX,CMmagY,CMmagZ)
    
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
                                                                                                  download)
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
    time = np.linspace(0,np.max(CMtimesec),5000)
    
    # Plot
    plt.figure(figsize=(6,6))
    plt.suptitle("CrowdMag vs GeoMag Scatter Plot", fontsize=12)
    plt.title("{} - {}".format(starttime,endtime), fontsize=12)
    
    if fieldtype == 'T':
        # Total magnetic field
        plt.scatter(CMtotalmagSpline(time),GMtotalmagSpline(time+timeshift))
        plt.xlabel("CrowdMag - Total Magnetic Field (nT)", fontsize=12)
        plt.ylabel("GeoMag - Total Magnetic Field (nT)", fontsize=12)
        cmdata = CMtotalmagSpline(time)
        gmdata = GMtotalmagSpline(time+timeshift)
        
    if fieldtype == 'H':        
        # Horizontal magnetic field   
        plt.scatter(CMmagHSpline(time),GMmagHSpline(time+timeshift))
        plt.xlabel("CrowdMag - Magnetic Field - Horizontal (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Horizontal (nT)", fontsize=12)
        cmdata = CMmagHSpline(time)
        gmdata = GMmagHSpline(time+timeshift)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        plt.scatter(CMmagXSpline(time),GMmagXSpline(time+timeshift))
        plt.xlabel("CrowdMag - Magnetic Field - X (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - X (nT)", fontsize=12)
        cmdata = CMmagXSpline(time)
        gmdata = GMmagXSpline(time+timeshift)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        plt.scatter(CMmagYSpline(time),GMmagYSpline(time+timeshift))
        plt.xlabel("CrowdMag - Magnetic Field - Y (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Y (nT)", fontsize=12)
        cmdata = CMmagYSpline(time)
        gmdata = GMmagYSpline(time+timeshift)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        plt.scatter(CMmagZSpline(time),GMmagZSpline(time+timeshift))
        plt.xlabel("CrowdMag - Magnetic Field - Z (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Z (nT)", fontsize=12)
        cmdata = CMmagZSpline(time)
        gmdata = GMmagZSpline(time+timeshift)
    
    r = CorrelationCoefficient(cmdata,gmdata)
    # Fitting: polyfit
    x = np.linspace(np.min(cmdata),np.max(cmdata),5000)
    slope_polyfit,intercept_polyfit = FittingPolyfit(cmdata,gmdata)
    # Residuals
    residuals = gmdata - LinearFunction(x,slope_polyfit,intercept_polyfit)
    # Chi squared and reduced chi squared
    chisquared = np.sum(residuals**2/np.sqrt(gmdata)**2)   # not sure about the errors!!
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
    
    # Pearson correlation testing for H-component
    scale = np.mean(GMmagH)/np.mean(CMmagH)
    stack = (np.vstack((scale*CMmagHSpline(time), GMmagHSpline(time))).T)
    stackpd = pd.DataFrame(stack, columns = ['CrowdMag','GeoMag'])
    overall_pearson_r = stackpd.corr().iloc[0,1]
    print(f"Pandas computed Pearson r: {overall_pearson_r:.3g}")
    
    r, p = stats.pearsonr(stackpd.dropna()['CrowdMag'], stackpd.dropna()['GeoMag'])
    print(f"Scipy computed Pearson r: {r:.3g} and p-value: {p:.3g}")

    f, ax = plt.subplots(figsize=(14,3))
    stackpd.rolling(window=30,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time (s)',ylabel='Magnetic Field - H (nT)',
           title=f"Overall Pearson r = {overall_pearson_r:.2g}");
    
#####################################
### Time-lagged Cross Correlation ###
#####################################

def TLCC(x,y, lag=0, wrap=False):
    
    """ Time-lagged Cross Correlation calculation. 
    Shifting data and calculating correlation coefficient
    
    Parameters
    ----------
    lag : int, default 0
    x,y : pandas.Series objects of equal length

    Returns
    ----------
    correlation coefficient : float
    """
    
    if wrap:   # wrap to deal with the edges of the data
        shiftedy = y.shift(lag)
        shiftedy.iloc[:lag] = y.iloc[-lag:].values
        return x.corr(shiftedy)
    else: 
        return x.corr(y.shift(lag))

####################################################    
### Rolling window time-lagged cross correlation ###
####################################################

def RWTLCC(filenameCM,
           observatory='BRW',
           fieldtype='T',
           startCM=3,endCM=-1,startGM=0,endGM=-1,
           download=True,
           n=500,windowsize=300,step=100): 
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # CrowdMag data
    CMdate,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filenameCM,startCM,endCM)
    CMmagH = cm.HorizontalMag(CMmagX,CMmagY)
    CMtotalmag = cm.TotalMag(CMmagX,CMmagY,CMmagZ)
    
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
                                                                                                  download)
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
    time = np.linspace(0,np.max(CMtimesec),5000)
    
    # Plot    
    if fieldtype == 'T':
        # Total magnetic field
        scale = np.mean(GMtotalmag)/np.mean(CMtotalmag)
        stack = (np.vstack((scale*CMtotalmagSpline(time), GMtotalmagSpline(time))).T)
        
    if fieldtype == 'H':        
        # Horizontal magnetic field   
        scale = np.mean(GMmagH)/np.mean(CMmagH)
        stack = (np.vstack((scale*CMmagHSpline(time), GMmagHSpline(time))).T)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        scale = np.mean(GMmagX)/np.mean(CMmagX)
        stack = (np.vstack((scale*CMmagXSpline(time), GMmagXSpline(time))).T)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        scale = np.mean(GMmagY)/np.mean(CMmagY)
        stack = (np.vstack((scale*CMmagYSpline(time), GMmagYSpline(time))).T)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        scale = np.mean(GMmagZ)/np.mean(CMmagZ)
        stack = (np.vstack((scale*CMmagZSpline(time), GMmagZSpline(time))).T)
    
    stackdatasets = pd.DataFrame(stack, columns = ['CrowdMag','GeoMag'])
    
    t_start = 0
    t_end = t_start + windowsize
    rss = []
    while t_end < len(stackdatasets):
        dataCM = stackdatasets['CrowdMag'].iloc[t_start:t_end]
        dataGM = stackdatasets['GeoMag'].iloc[t_start:t_end]
        rs = [TLCC(dataCM,dataGM, lag, wrap=False) for lag in range(-int(n-1),int(n))]
        rss.append(rs)
        t_start = t_start + step
        t_end = t_end + step
    rss = pd.DataFrame(rss)

    f, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(rss,cmap='RdBu_r',ax=ax)
    ax.set(title=f'Rolling Windowed Time Lagged Cross Correlation', xlabel='Offset',ylabel='Epochs');
    plt.legend()