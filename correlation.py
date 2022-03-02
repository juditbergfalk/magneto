import crowdmag as cm
import geomag as gm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline      # Interpolation (spline) function
import lmfit as lm                                              # Fitting

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
        ax.scatter(CMtimesec+timeshift,scale*CMtotalmag, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.scatter(GMtimesec,GMtotalmag, label="GeoMag data")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field 
        scale = np.mean(GMmagH)/np.mean(CMmagH)
        ax.scatter(CMtimesec+timeshift,scale*CMmagH, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.scatter(GMtimesec,GMmagH, label="GeoMag data")
        plt.ylabel("Magnetic Field - H (nT)", fontsize=12)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        scale = np.mean(GMmagX)/np.mean(CMmagX)
        ax.scatter(CMtimesec,scale*CMmagX, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.scatter(GMtimesec+timeshift,GMmagX, label="GeoMag data")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        scale = np.mean(GMmagY)/np.mean(CMmagY)
        ax.scatter(CMtimesec+timeshift,scale*CMmagY, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.scatter(GMtimesec,GMmagY, label="GeoMag data")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        scale = np.mean(GMmagZ)/np.mean(CMmagZ)
        ax.scatter(CMtimesec+timeshift,scale*CMmagZ, label="CrowdMag data, scaled by {:.3f}".format(scale))
        ax.scatter(GMtimesec,GMmagZ, label="GeoMag data")
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
    plt.figure(figsize=(9,9))
    plt.suptitle("CrowdMag vs GeoMag Scatter Plot", fontsize=16)
    plt.title("{} - {}".format(starttime,endtime), fontsize=16)
    
    if fieldtype == 'T':
        # Total magnetic field
        plt.scatter(CMtotalmagSpline(time),GMtotalmagSpline(time))
        plt.xlabel("CrowdMag - Total Magnetic Field (nT)", fontsize=12)
        plt.ylabel("GeoMag - Total Magnetic Field (nT)", fontsize=12)
        r = CorrelationCoefficient(CMtotalmagSpline(time),GMtotalmagSpline(time))
        # Fitting: polyfit
        x = np.linspace(np.min(CMtotalmagSpline(time)),np.max(CMtotalmagSpline(time)),5000)
        slope_polyfit,intercept_polyfit = FittingPolyfit(CMtotalmagSpline(time),GMtotalmagSpline(time))
        # Residuals
        residuals = GMtotalmagSpline(time) - LinearFunction(x,slope_polyfit,intercept_polyfit)
        # Chi squared and reduced chi squared
        chisquared = np.sum(residuals**2/np.sqrt(GMtotalmagSpline(time))**2)   # not sure about the errors!!
        # Degrees of freedom = number of data points - number of fitting parameters
        dof = len(CMdate) - 2    
        reducedchisquared = chisquared / dof
        
    if fieldtype == 'H':        
        # Horizontal magnetic field   
        plt.scatter(CMmagHSpline(time),GMmagHSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - Horizontal (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Horizontal (nT)", fontsize=12)
        r = CorrelationCoefficient(CMmagHSpline(time),GMmagHSpline(time))
        # Fitting: polyfit
        x = np.linspace(np.min(CMmagHSpline(time)),np.max(CMmagHSpline(time)),5000)
        slope_polyfit,intercept_polyfit = FittingPolyfit(CMmagHSpline(time),GMmagHSpline(time))
        # Residuals
        residuals = GMmagHSpline(time) - LinearFunction(x,slope_polyfit,intercept_polyfit)
        # Chi squared and reduced chi squared
        chisquared = np.sum(residuals**2/np.sqrt(GMmagHSpline(time))**2)   # not sure about the errors!!
        # Degrees of freedom = number of data points - number of fitting parameters
        dof = len(CMdate) - 2    
        reducedchisquared = chisquared / dof
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        plt.scatter(CMmagXSpline(time),GMmagXSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - X (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - X (nT)", fontsize=12)
        r = CorrelationCoefficient(CMmagXSpline(time),GMmagXSpline(time))
        # Fitting: polyfit
        x = np.linspace(np.min(CMmagXSpline(time)),np.max(CMmagXSpline(time)),5000)
        slope_polyfit,intercept_polyfit = FittingPolyfit(CMmagXSpline(time),GMmagXSpline(time))
        # Residuals
        residuals = GMmagXSpline(time) - LinearFunction(x,slope_polyfit,intercept_polyfit)
        # Chi squared and reduced chi square
        chisquared = np.sum(residuals**2/np.sqrt(GMmagXSpline(time))**2)   # not sure about the errors!!
        # Degrees of freedom = number of data points - number of fitting parameters
        dof = len(CMdate) - 2    
        reducedchisquared = chisquared / dof
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        plt.scatter(CMmagYSpline(time),GMmagYSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - Y (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Y (nT)", fontsize=12)
        r = CorrelationCoefficient(CMmagYSpline(time),GMmagYSpline(time))
        # Fitting: polyfit
        x = np.linspace(np.min(CMmagYSpline(time)),np.max(CMmagYSpline(time)),5000)
        slope_polyfit,intercept_polyfit = FittingPolyfit(CMmagYSpline(time),GMmagYSpline(time))
        # Residuals
        residuals = GMmagYSpline(time) - LinearFunction(x,slope_polyfit,intercept_polyfit)
        # Chi squared and reduced chi squared
        chisquared = np.sum(residuals**2/np.sqrt(GMmagYSpline(time))**2)   # not sure about the errors!!
        # Degrees of freedom = number of data points - number of fitting parameters
        dof = len(CMdate) - 2    
        reducedchisquared = chisquared / dof
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        plt.scatter(CMmagZSpline(time),GMmagZSpline(time))
        plt.xlabel("CrowdMag - Magnetic Field - Z (nT)", fontsize=12)
        plt.ylabel("GeoMag - Magnetic Field - Z (nT)", fontsize=12)
        r = CorrelationCoefficient(CMmagZSpline(time),GMmagZSpline(time))
        # Fitting: polyfit
        x = np.linspace(np.min(CMmagZSpline(time)),np.max(CMmagZSpline(time)),5000)
        slope_polyfit,intercept_polyfit = FittingPolyfit(CMmagZSpline(time),GMmagZSpline(time))
        # Residuals
        residuals = GMmagZSpline(time) - LinearFunction(x,slope_polyfit,intercept_polyfit)
        # Chi squared and reduced chi squared
        chisquared = np.sum(residuals**2/np.sqrt(GMmagZSpline(time))**2)   # not sure about the errors!!
        # Degrees of freedom = number of data points - number of fitting parameters
        dof = len(CMdate) - 2    
        reducedchisquared = chisquared / dof
        
    plt.plot(x,LinearFunction(x,slope_polyfit,intercept_polyfit), label="Linear Fit", color='r')
    plt.legend()
    plt.show()
    print("Correlation coefficient of the two datasets is {:.2f}.".format(r))
    print("Slope = {}".format(slope_polyfit))
    print("Intercept = {}".format(intercept_polyfit))  
    print("Chi-squared = {}".format(chisquared))
    print("Reduced chi-squared = {}".format(reducedchisquared))
    
    # Pearson
    import pandas as pd
    import seaborn as sns
    import scipy.stats as stats

    df = pd.read_csv('synchrony_sample.csv')
    overall_pearson_r = df.corr().iloc[0,1]
    print(f"Pandas computed Pearson r: {overall_pearson_r}")
    # out: Pandas computed Pearson r: 0.2058774513561943

    r, p = stats.pearsonr(df.dropna()['S1_Joy'], df.dropna()['S2_Joy'])
    print(f"Scipy computed Pearson r: {r} and p-value: {p}")
    # out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

    # Compute rolling window synchrony
    f,ax=plt.subplots(figsize=(7,3))
    df.rolling(window=30,center=True).median().plot(ax=ax)
    ax.set(xlabel='Time',ylabel='Pearson r')
    ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}");