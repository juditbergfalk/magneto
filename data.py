import numpy as np
import os
import pandas as pd
from IPython.display import clear_output, display
from tkinter import Tk, filedialog

######################
# file select button #
######################

def selectFiles(b):
    clear_output()                                         # Button is deleted after it is clicked.
    root = Tk()
    root.withdraw()                                        # Hide the main window.
    root.call('wm', 'attributes', '.', '-topmost', True)   # Raise the root to the top of all windows.
    b.files = filedialog.askopenfilename(multiple=True)    # List of selected files will be set button's file attribute.
    print('List of file(s) selected:\n',b.files)
    
    
####################
# sorting csv data #
####################

def dataParse(files):
    '''
    Takes one or more .csv files and creates a new data frame for each file consisting of x,y,z,h, and t
    filetypes, creates a list of file names for each file selected, and returns start and end times that
    will be used to pull corresponding GeoMag data.
    '''
    
    filePaths = files
    fileNames = []
    
    # create a list containing file names
    for i in range(len(filePaths)):
        filePath = filePaths[i]  # file path
        fileName = os.path.basename(filePath)
        fileNames.append(fileName)

    # read in files and create a new data frame for each file
    dataFrames = []
    for i in range(len(filePaths)):
        file = filePaths[i]
        data = pd.read_csv(file, parse_dates=['Time (UTC)'],index_col=0)

        # find start/end times 
        #(ultimately will find best window over multiple data sets of roughly same time period for stacking)
        # for now, this is for a single data set
        startTime = data.index.min()
        endTime = data.index.max()

        # assigning columns
        xMag = data.iloc[:,2]
        yMag = data.iloc[:,3]
        zMag = data.iloc[:,4]

        # calculating H (horizontal magnitude) and T (total magnitude)
        hMag = np.sqrt(xMag**2 + yMag**2)
        tMag = np.sqrt(xMag**2 + yMag**2 + zMag**2)

        # create a new data frame
        df = pd.DataFrame({'xMag':xMag,'yMag':yMag,'zMag':zMag,'hMag':hMag,'tMag':tMag})
        dataFrames.append(df)
    
    return startTime,endTime,fileNames,dataFrames

    