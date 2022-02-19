import crowdmag as cm
import geomag as gm
import numpy as np
import matplotlib.pyplot as plt

###############################
### Plot CrowdMag vs GeoMag ###
###############################

def PlotOverlay2Data(filename,observatory='BRW',fieldtype='T',start=3,end=-1,download=True):
    
    # Key:
    ##### fieldtype = 'T'  - total magnetic field
    ##### fieldtype = 'H'  - horizontal component of field
    ##### fieldtype = 'X'  - x-component of magnetic field
    ##### fieldtype = 'Y'  - y-component of magnetic field
    ##### fieldtype = 'Z'  - z-component of magnetic field
    
    # CrowdMag data
    CMdate,CMmagX,CMmagY,CMmagZ = cm.ReadCSVCrowdMag(filename,start,end)
    CMmagH = cm.HorizontalMag(CMmagX,CMmagY)
    CMtotalmag = cm.TotalMag(CMmagX,CMmagY,CMmagZ)
    
    # Time in seconds 
    CMtimesec = cm.SplitTime(CMdate)[8] - cm.SplitTime(CMdate)[8][0]
    
    # GeoMag data
    GMdate,GMtime,GMdoy,GMmagX,GMmagY,GMmagZ,GMmagH,GMtimesec,GMlocation = gm.DefineAllComponents(filename,observatory,
                                                                                                  start,end,download)
    GMtotalmag = cm.TotalMag(GMmagX,GMmagY,GMmagZ)
    
    # Fuse date and time to match CrowdMag date
    GMdatetime = []
    for t in range(len(GMdate)):
        dt = GMdate[t] + ' ' + GMtime[t][0:8]
        GMdatetime.append(dt)
    GMdatetime = np.array(GMdatetime)
    
    # Time in seconds 
    GMtimesec = cm.SplitTime(GMdatetime)[8] - cm.SplitTime(GMdatetime)[8][0]
    
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
        ax.scatter(CMtimesec,CMtotalmag, label="CrowdMag data")
        ax.scatter(GMtimesec,GMtotalmag, label="GeoMag data")
        plt.ylabel("Total Magnetic Field (nT)", fontsize=12)
    
    if fieldtype == 'H':        
        # Horizontal magnetic field        
        ax.scatter(CMtimesec,0.23*CMmagH, label="CrowdMag data")
        ax.scatter(GMtimesec,GMmagH, label="GeoMag data")
        plt.ylabel("Magnetic Field - H (nT)", fontsize=12)
    
    if fieldtype == 'X':        
        # Magnetic field - X direction 
        ax.scatter(CMtimesec,CMmagX, label="CrowdMag data")
        ax.scatter(GMtimesec,GMmagX, label="GeoMag data")
        plt.ylabel("Magnetic Field - X (nT)", fontsize=12)
    
    if fieldtype == 'Y':
        # Magnetic field - Y direction
        ax.scatter(CMtimesec,CMmagY, label="CrowdMag data")
        ax.scatter(GMtimesec,GMmagY, label="GeoMag data")
        plt.ylabel("Magnetic Field - Y (nT)", fontsize=12)
        
    if fieldtype == 'Z':
        # Magnetic field - Z direction
        ax.scatter(CMtimesec,CMmagZ, label="CrowdMag data")
        ax.scatter(GMtimesec,GMmagZ, label="GeoMag data")
        plt.ylabel("Magnetic Field - Z (nT)", fontsize=12)
        
    plt.legend()
    plt.show()
    

    