import os
import codecs

import numpy as np
import math
import matplotlib.pyplot as plt

# Import curve fitting package from scipy
from scipy.optimize import curve_fit


class Spectrum():
    DELIMITER = '  '
    EOF = '****'
    HEADER_LENGTH = 11  # Including EOF
    POS_INDEX = 8  # Where the position data is in the header

    def __init__(self, filepath):
        self.filepath = filepath

        self.raw_data = self.extract_data()
        self.header = self.remove_header()
        self.data = self.convert_raw_data()

        self.num_channels = len(self.data[0])

    def extract_data(self):
        with codecs.open(self.filepath, encoding='ASCII') as f:
            data = np.array([i.strip('\n').strip('\r').strip(self.DELIMITER)
                             for i in f])
        return data

    def remove_header(self):
        '''
        Note that this also alters self.raw_data by removing the header
        '''
        header = self.raw_data[:self.HEADER_LENGTH]
        self.raw_data = self.raw_data[self.HEADER_LENGTH:-1]  # Drop EOF
        return header

    def convert_raw_data(self):
        data = [[float(i) for i in row.split(self.DELIMITER)]
                for row in self.raw_data]
        return np.array(data)

# Function to calculate the Gaussian with constants a, b, and c
def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

# Create variables for filepaths
TARGET = 'W51'
path = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/split_data/'
print('Data path set to '+path)

# If the plot directory does not exist, create the appropriate folders
plotpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/split_plots/'
print('Plot path set to '+plotpath)
if not os.path.exists(plotpath):
    os.makedirs(plotpath)
    print('Directory path created for '+plotpath)
   
# If the calibrated plot directory does not exist, create the appropriate folders
bandpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/bandpass/'
print('Plot path set to '+bandpath)
if not os.path.exists(bandpath):
    os.makedirs(bandpath)
    print('Directory path created for '+bandpath) 
   
# If the calibrated plot directory does not exist, create the appropriate folders
calpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/calibrated_plots/'
print('Plot path set to '+calpath)
if not os.path.exists(calpath):
    os.makedirs(calpath)
    print('Directory path created for '+calpath)

# If the stacked plot directory does not exist, create the appropriate folders
stackpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/stacked_calibrated_plots/averaged_over_all/'
print('Plot path set to '+stackpath)
if not os.path.exists(stackpath):
    os.makedirs(stackpath)
    print('Directory path created for '+stackpath)
    
# If the directory does not exist, create the appropriate folders
enstackpath1 = r'/Users/owen/Desktop/Analysis/Python/' + \
    TARGET+'/enlarged_stacked_calibrated_plots/125-175/'
print('Plot path set to '+enstackpath1)
if not os.path.exists(enstackpath1):
    os.makedirs(enstackpath1)
    print('Directory path created for '+enstackpath1)
    
# If the stacked plot directory does not exist, create the appropriate folders
nonavestackpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/16_stacked_calibrated_plots/'
print('Plot path set to '+nonavestackpath)
if not os.path.exists(nonavestackpath):
    os.makedirs(nonavestackpath)
    print('Directory path created for '+nonavestackpath)
    
# If the stacked plot directory does not exist, create the appropriate folders
nonaveenstackpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/16_enlarged_stacked_calibrated_plots/'
print('Plot path set to '+nonaveenstackpath)
if not os.path.exists(nonaveenstackpath):
    os.makedirs(nonaveenstackpath)
    print('Directory path created for '+nonaveenstackpath)
    
# If the stacked plot directory does not exist, create the appropriate folders
nonavepath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/8_stacked_calibrated_plots/'
print('Plot path set to '+nonavepath)
if not os.path.exists(nonavepath):
    os.makedirs(nonavepath)
    print('Directory path created for '+nonavepath)
    
# If the save data directory does not exist, create the appropriate folders
savepath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/save_data/'
print('Plot path set to '+savepath)
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print('Directory path created for '+savepath)

# Create a list containing all raw split data files
files = []
print(sorted(os.listdir(path)))
for filenames in sorted(os.listdir(path)):
    if filenames.startswith('.'):
        # If the file starts with a '.', such as .DS_Store, it will be ignored
        print('    File ignored: ')
        print(filenames)
    else:
        files.extend([filenames])
        print('    File added: ')
        print(filenames)

# Print the list of filtered files
print('    Filtered files: ')
print(files)

# averaged bandpass (uses 4 OFF-source scans from all 8 dataset)
do_1plotbandpass = 1 

# non-averaged bandpass (8 individual plots that use 4 OFF-source scans from each respective dataset)
do_8plotbandpass = 1


# bandpass-smoothed non-stacked and stacked graphs for a user-specified dataset (using averaged bandpass over all 8 datasets)
do_1plot1dataset = 0
do_2plot1dataset = 0
dataset = 6

# bandpass-smoothed non-stacked and stacked graphs for average of all 8 datasets (using averaged bandpass over all 8 datasets)
do_1plotcaldata = 0
do_2plotcaldata = 0
do_2plotencaldata = 0 # same as above function, but enlarged to specific LSR velocity range
do_single2plotencaldata = 0 # Do not use
do_2gaussplot = 0 # although attempts were made to create a self-analysing Gaussian function, these were unsuccessful - use Origin

# produces bandpass-smoothed non-stacked and stacked graphs for each individual dataset (each dataset uses its own respective bandpass)
do_8plotcaldata = 0 
do_16plotcaldata = 1 # analyses all 8 datasets, using 8 bandpasses - one for each dataset
do_16plotencaldata = 0 # same as above function, but enlarged to specific LSR velocity range

# Save data = 1. Do not save data = 0
do_savegraphs = 0

ylab1 = "Intensity (Volts)"

xlab1 = "LSR velocity (km/s)"
xscale1 = np.arange(122.88, -122.88, -0.48) # Axis with size 512
v_lower = 27.36
v_upper = 75.36
v_step = -0.48
xscale2 = np.arange(v_upper, v_lower, v_step) # Axis with size 100

xlab2 = "Channel number"

dataPointLower1 = int((122.88 - v_lower)/(-v_step))
dataPointUpper1 = int((122.88 - v_upper)/(-v_step)) 
print(dataPointLower1)
print(dataPointUpper1)
dataPointLower2 = int(dataPointLower1 + 512)
dataPointUpper2 = int(dataPointUpper1 + 512)

targetAzimuth = 274
targetAltitude = 14.67
observatoryLatitude = 51.4584
earthRotationVelocityMax = float(0.46388889 * math.cos(observatoryLatitude))
earthRotationVelocityTowardsTarget = float(earthRotationVelocityMax * math.cos(math.radians(targetAzimuth-90)) * math.cos(math.radians(targetAltitude)))
galacticLongitude = 49.5
galacticLatitude = -0.4
solarVelocityOne = 9 * math.cos(math.radians(galacticLongitude)) * math.cos(math.radians(galacticLatitude))
solarVelocityTwo = 12 * math.sin(math.radians(galacticLongitude)) * math.cos(math.radians(galacticLatitude))
solarVelocityThree = 7 * math.sin(math.radians(galacticLatitude))
solarPeculiarMotion = solarVelocityOne + solarVelocityTwo + solarVelocityThree
print('\nvelocity shift is: ' + str(earthRotationVelocityTowardsTarget + solarPeculiarMotion))
xscale1 = xscale1 + earthRotationVelocityTowardsTarget + solarPeculiarMotion
xscale2 = xscale2 + earthRotationVelocityTowardsTarget + solarPeculiarMotion


if(do_1plotbandpass == 1):
    # Create an array for the bandpass using the 4 corner scans from the 5x5 grid
    bandpass = np.zeros((1024))
    print(bandpass)
    count = 1
    for x in files:
        print('\tScan number '+str(count))
        # Process only for the 4 corner scans (Furthest from centre)
        if (count == 1 or count == 5 or count == 21 or count == 25):
            plot = Spectrum(TARGET+'/split_data/'+x)
            for i in range(plot.num_channels):
                print('\t\tData set '+str(i))
                # Loop through all 8 data sets in each scan, and add to the bandpass
                # Prints current bandpass, next data set and sum of the two
                print(bandpass)
                print(plot.data[:, i])
                bandpass += plot.data[:, i]
                print(bandpass)
        count += 1
    
    # Create a plot of the bandpass from the bandpass array data
    bandpass /= 32 # To retrieve the average, divide by number of scans
    print('Bandpass: ')
    print(bandpass)
    print('\n')
    for i in range(plot.num_channels):
        plt.plot(bandpass)
        plt.xlabel(xlab2)
        plt.title("4-scan averaged, 8-plot averaged bandpass")
    # Save figures as high resolution .png files ~ 0.5 MB each file
    if (do_savegraphs == 1):
        plt.savefig(bandpath+'bandpass averaged.png', dpi=1000)
    plt.show()


if(do_8plotbandpass == 1):
    # Create an array for the bandpass using the 4 corner scans from the 5x5 grid
    plotbandpass = np.zeros((1024, 8))
    print(plotbandpass)
    count = 1
    for x in files:
        print('\tScan number '+str(count))
        # Process only for the 4 corner scans (Furthest from centre)
        if (count == 1 or count == 5 or count == 21 or count == 25):
            plot = Spectrum(TARGET+'/split_data/'+x)
            for i in range(plot.num_channels):
                print('\t\tData set '+str(i))
                # Loop through all 8 data sets in each scan, and add to the bandpass
                # Prints current bandpass, next data set and sum of the two
                print(plotbandpass[:,i])
                print(plot.data[:, i])
                plotbandpass[:,i] += plot.data[:, i]
                print(plotbandpass[:,i])
        count += 1
    
    # Create an 8-plot of the bandpass from the bandpass array data
    plotbandpass /= 4 # To retrieve the average, divide by number of scans
    print('Bandpass: ')
    print(plotbandpass)
    print('\n')
    for i in range(plot.num_channels):
        plt.plot(plotbandpass)
    plt.xlabel(xlab2)
    plt.ylabel(ylab1)
    # Save figures as high resolution .png files ~ 0.5 MB each file
    if (do_savegraphs == 1):
        plt.savefig(bandpath+'full dataset bandpass.png', dpi=1000)
    plt.show()
    

if(do_1plot1dataset):
    # If the stacked plot directory does not exist, create the appropriate folders
    calpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/calibrated_plots/'+str(dataset)+'/'
    print('Plot path set to '+calpath)
    # If the directory does not exist, create the appropriate folders
    if not os.path.exists(calpath):
        os.makedirs(calpath)
        print('Directory path created for '+calpath)
    # Plot 1-line graph using only 1 data set
    caldata = np.zeros(1024)
    for x in files:
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        count = 0
        for i in range(plot.num_channels):
            count += 1
            if (count == dataset):
                print(i)
                print(plot.data[:,i])
                plt.plot(plot.data[0:1024,i]/bandpass[0:1024] - 1)
                plt.xlabel(xlab2)
                plt.ylabel(ylab1)
                # Save figures as high resolution .png files ~ 0.5 MB each file
                if (do_savegraphs == 1):
                    plt.savefig(calpath+x.replace('POSITION:       ','stack ').replace('.txt',' ')+str(dataset+1)+'.png', dpi = 1000)
                plt.show()


if(do_2plot1dataset):
    # If the stacked plot directory does not exist, create the appropriate folders
    stackpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/stacked_calibrated_plots/'+str(dataset)+'/'
    print('Plot path set to '+stackpath)
    # If the directory does not exist, create the appropriate folders
    if not os.path.exists(stackpath):
        os.makedirs(stackpath)
        print('Directory path created for '+stackpath)
    # Plot 2-line graph using only 1 data set
    caldata = np.zeros(1024)
    for x in files:
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        count = 0
        for i in range(plot.num_channels):
            count += 1
            if (count == dataset):
                print(i)
                print(plot.data[:,i])
                plt.plot(xscale1, plot.data[0:512,i]/bandpass[0:512] - 1)
                plt.plot(xscale1, plot.data[512:1024,i]/bandpass[512:1024] - 1)
                plt.xlabel(xlab1)
                plt.ylabel(ylab1)
                # Save figures as high resolution .png files ~ 0.5 MB each file
                if (do_savegraphs == 1):
                    plt.savefig(stackpath+x.replace('POSITION:       ','stack ').replace('.txt',' ')+str(dataset)+'.png', dpi = 1000)
                plt.show()


if(do_1plotcaldata == 1):
        # If the stacked plot directory does not exist, create the appropriate folders
    calpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/calibrated_plots/averaged_over_all/'
    print('Plot path set to '+calpath)
    # If the directory does not exist, create the appropriate folders
    if not os.path.exists(calpath):
        os.makedirs(calpath)
        print('Directory path created for '+calpath)
    caldata = np.zeros(1024)
    for x in files:
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        for i in range(plot.num_channels):
            if i == dataset:
                print(i)
                print(caldata)
                print(plot.data[:,i])
                caldata += plot.data[:,i]
                print(caldata)
                print('\n')
        caldata /= 8
        plt.plot(caldata[0:1024]/bandpass[0:1024] - 1)
        plt.xlabel(xlab2)
        plt.ylabel(ylab1)
        plt.title(x.replace('POSITION:       ','RA: ').replace('      ',', DEC: ').replace('.txt','') + ', Dataset: ' + str(i+1))
        # Save figures as high resolution .png files ~ 0.5 MB each file
        if (do_savegraphs == 1):
            plt.savefig(calpath+x.replace('POSITION:       ','stack ').replace('.txt', ' all.png'), dpi=1000)
        plt.show()


if(do_2plotcaldata == 1):
    # If the stacked plot directory does not exist, create the appropriate folders
    stackpath = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/stacked_calibrated_plots/averaged_over_all/'
    print('Plot path set to '+stackpath)
    # If the directory does not exist, create the appropriate folders
    if not os.path.exists(stackpath):
        os.makedirs(stackpath)
        print('Directory path created for '+stackpath)
    # Plot 2-line graph
    caldata = np.zeros(1024)
    for x in files:
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        count = 0
        for i in range(plot.num_channels):
            print(i)
            print(caldata)
            print(plot.data[:,i])
            caldata += plot.data[:,i]
            print(caldata)
            print('\n')
        caldata /= 8
        plt.plot(xscale1, caldata[0:512]/bandpass[0:512] - 1)
        plt.plot(xscale1, caldata[512:1024]/bandpass[512:1024] - 1)
        plt.title(x.replace('POSITION:       ','RA: ').replace('      ',', DEC: ').replace('.txt','') + ', Dataset: ' + str(i+1))
        plt.xlabel(xlab1)
        plt.ylabel(ylab1)
        # Save figures as high resolution .png files ~ 0.5 MB each file
        if (do_savegraphs == 1):
            plt.savefig(stackpath+x.replace('POSITION:       ','stack ').replace('.txt', ' all.png'), dpi=1000)
        plt.show()
    

if(do_2plotencaldata == 1):
    # Plot (100 to 200) 2-line graph
    for x in files:
        print(x)
        caldata = np.zeros(1024)
        plot = Spectrum(TARGET+'/split_data/'+x)
        for i in range(plot.num_channels):
            print(i)
            print(caldata)
            print(plot.data[:,i])
            caldata += plot.data[:,i]
            print(caldata)
            print('\n')
        caldata /= 8
        plt.plot(xscale2, caldata[dataPointUpper1:dataPointLower1]/bandpass[dataPointUpper1:dataPointLower1] - 1, label = 'LCP')
        plt.plot(xscale2, caldata[dataPointUpper2:dataPointLower2]/bandpass[dataPointUpper2:dataPointLower2] - 1, label = 'RCP')# Save figures as high resolution .png files ~ 0.5 MB each file
        plt.xlabel(xlab1)
        plt.ylabel(ylab1)
        plt.legend()
        plt.yticks(np.arange(-0.01, 0.045, 0.005))
        title = x.replace('POSITION:       ','Position ').replace('.txt', '')
        plt.title(title)
        plt.grid(True)
        # Save figures as high resolution .png files ~ 0.5 MB each file
        if (do_savegraphs == 1):
            plt.savefig(enstackpath1+x.replace('POSITION:       ','stack ').replace('.txt', ' 6 125-175.png'), dpi=1000)
        plt.show()
        
        
if(do_single2plotencaldata == 1):
    # Plot (100 to 200) 2-line graph
    scanno = 1
    for x in files:
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        for i in range(plot.num_channels):
            caldata = np.zeros(1024)
            enstackpath1 = r'/Users/owen/Desktop/Analysis/Python/'+TARGET+'/enlarged_stacked_calibrated_plots/125-175/'+str(i+1)+'/'
            print(i)
            print(caldata)
            print(plot.data[:,i])
            caldata += plot.data[:,i]
            print(caldata)
            print('\n')        
            plt.plot(xscale2, caldata[dataPointUpper1:dataPointLower1]/bandpass[dataPointUpper1:dataPointLower1] - 1, label = 'LCP')
            plt.plot(xscale2, caldata[dataPointUpper2:dataPointLower2]/bandpass[dataPointUpper2:dataPointLower2] - 1, label = 'RCP')# Save figures as high resolution .png files ~ 0.5 MB each file
            plt.xlabel(xlab1)
            plt.ylabel(ylab1)
            plt.title(x.replace('POSITION:       ','RA: ').replace('      ',', DEC: ').replace('.txt','') + ', Dataset: ' + str(i+1))
            plt.legend()
            plt.grid(True)
            # Save figures as high resolution .png files ~ 0.5 MB each file
            if (do_savegraphs == 1):
                plt.savefig(enstackpath1+x.replace('POSITION:       ','stack ').replace('.txt', ' 6 125-175.png'), dpi=1000)
                savedata = np.array([caldata[dataPointUpper1:dataPointLower1]/bandpass[dataPointUpper1:dataPointLower1] - 1, caldata[dataPointUpper2:dataPointLower2]/bandpass[dataPointUpper2:dataPointLower2] - 1, xscale2])
                savedata = savedata.T
                np.savetxt(savepath+'125-175/'+str(i+1)+'/'+x.replace('POSITION:       ','Position '), savedata, '%s')
            plt.show()
        scanno += 1

        
if(do_2gaussplot == 1):
    # Plot (100 to 200) 2-line graph
    for x in files:
        print(x)
        caldata = np.zeros(1024)
        plot = Spectrum(TARGET+'/split_data/'+x)
        for i in range(plot.num_channels):
            print(i)
            print(caldata)
            print(plot.data[:,i])
            caldata += plot.data[:,i]
            print(caldata)
            print('\n')
        caldata /= 8
        plt.plot(xscale2, caldata[dataPointUpper1:dataPointLower1]/bandpass[dataPointUpper1:dataPointLower1] - 1, label = 'LCP')
        plt.plot(xscale2, caldata[dataPointUpper2:dataPointLower2]/bandpass[dataPointUpper2:dataPointLower2] - 1, label = 'RCP')# Save figures as high resolution .png files ~ 0.5 MB each file
        plt.xlabel(xlab1)
        plt.ylabel(ylab1)
        plt.legend()
        plt.yticks(np.arange(-0.01, 0.045, 0.005))
        title = x.replace('POSITION:       ','Position ').replace('.txt', '')
        plt.title(title)
        plt.grid(True)
        std = np.std((caldata[dataPointUpper1:dataPointLower1]+caldata[dataPointUpper2:dataPointLower2])/2, ddof = 1)
        mean = np.mean((caldata[dataPointUpper1:dataPointLower1]+caldata[dataPointUpper2:dataPointLower2])/2)
        print(caldata[dataPointUpper1:dataPointLower1])
        print(caldata[dataPointUpper2:dataPointLower2])
        # Save figures as high resolution .png files ~ 0.5 MB each file
        if (do_savegraphs == 1):
            plt.savefig(enstackpath1+x.replace('POSITION:       ','stack ').replace('.txt', ' 6 125-175.png'), dpi=1000)
            savedata = np.array([caldata[dataPointUpper1:dataPointLower1]/bandpass[dataPointUpper1:dataPointLower1] - 1, caldata[dataPointUpper2:dataPointLower2]/bandpass[dataPointUpper2:dataPointLower2] - 1, xscale2])
            savedata = savedata.T
            np.savetxt(savepath+x.replace('POSITION:       ','Position ')+'.txt', savedata, '%s')
        plt.show()
    
    
if(do_8plotcaldata == 1):
    # Plot 8-line graph
    for x in files:
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        count = 0
        for i in range(plot.num_channels):
            count += 1
            print(i)
            plt.plot(plot.data[0:1024,i]/plotbandpass[0:1024,i] - 1)
            plt.title(x.replace('POSITION:       ','RA: ').replace('      ',', DEC: ').replace('.txt','') + ', Dataset: ' + str(i+1))
            plt.xlabel(xlab2)
            plt.ylabel(ylab1)
            # Save figures as high resolution .png files ~ 0.5 MB each file
            if (do_savegraphs == 1):
                plt.savefig(nonavepath+x.replace('POSITION:       ','cal ').replace('.txt',' all.png'), dpi = 1000)
            plt.show()
        

if(do_16plotcaldata == 1):
    # Plot 16-line graph
    count = 0
    for x in files:
        count += 1
        print(x)
        if count == 13:
            plot = Spectrum(TARGET+'/split_data/'+x)
            for i in range(plot.num_channels):
                plt.plot(xscale1, plot.data[0:512,i]/plotbandpass[0:512,i] - 1, label = 'LCP')
                plt.plot(xscale1, plot.data[512:1024,i]/plotbandpass[512:1024,i] - 1, label = 'RCP')
                plt.tick_params(direction ='in')
                font = 'arial' # default sans-serif
                plt.xlabel(xlab1, family=font)
                plt.ylabel(ylab1, family=font)
                plt.title(x.replace('POSITION:       ','RA: ').replace('      ','°, DEC: ').replace('.txt','') + '°, Dataset: ' + str(i+1), family=font)
                plt.rc('font', size=9) # default 10
                plt.rc('axes', labelsize=11) # default 10
                plt.rc('axes', titlesize=11) # default 12
                plt.rcParams["figure.dpi"] = 1000
                plt.tick_params(direction ='in')
                plt.grid(False)
                plt.legend()
                # Save figures as high resolution .png files ~ 0.5 MB each file
                if (do_savegraphs == 1):
                    plt.savefig(nonavestackpath+str(i+1)+'/'+x.replace('POSITION:       ','stack ').replace('.txt',' all.png'), dpi = 1000)
                    savedata = np.array([plot.data[dataPointUpper1:dataPointLower1,i]/plotbandpass[dataPointUpper1:dataPointLower1,i] - 1, plot.data[dataPointUpper2:dataPointLower2,i]/plotbandpass[dataPointUpper2:dataPointLower2,i] - 1, xscale2])
                    savedata = savedata.T
                    np.savetxt(savepath+'125-175/'+str(i+1)+'/'+x.replace('POSITION:       ','Position '), savedata, '%s')
                plt.show()
            

if(do_16plotencaldata == 1):
    # Plot 16-line graph
    count = 0
    for x in files:
        count += 1
        print(x)
        plot = Spectrum(TARGET+'/split_data/'+x)
        if count == 13:
            for i in range(plot.num_channels):
                print(i)
                print(plot.data[:,i])
                plt.plot(xscale2, plot.data[dataPointUpper1:dataPointLower1,i]/plotbandpass[dataPointUpper1:dataPointLower1,i] - 1, label = 'LCP')
                plt.plot(xscale2, plot.data[dataPointUpper2:dataPointLower2,i]/plotbandpass[dataPointUpper2:dataPointLower2,i] - 1, label = 'RCP')# Save figures as high resolution .png files ~ 0.5 MB each file
                font = 'arial' # default sans-serif
                plt.xlabel(xlab1, family=font)
                plt.ylabel(ylab1, family=font)
                plt.title(x.replace('POSITION:       ','RA: ').replace('      ','°, DEC: ').replace('.txt','') + '°, Dataset: ' + str(i+1), family=font)
                plt.rc('font', size=9) # default 10
                plt.rc('axes', labelsize=11) # default 10
                plt.rc('axes', titlesize=11) # default 12
                plt.rcParams["figure.dpi"] = 1000
                plt.tick_params(direction ='in')
                plt.grid(False)
                plt.legend()
                # Save figures as high resolution .png files ~ 0.5 MB each file
                if (do_savegraphs == 1):
                    plt.savefig(nonaveenstackpath+str(i+1)+'/'+x.replace('POSITION:       ','stack ').replace('.txt',' all.png'), dpi = 1000)
                plt.show()
