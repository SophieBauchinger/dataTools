#!/usr/bin/env python
# Time-stamp: <2023-01-20 11:52:02 dkunkel>

import numpy as np
import pandas as pd
import argparse 
import sys
import os
import matplotlib.pyplot as plt

def read_sonde(fname, verbose=False):
    '''
    This routine is very hard coded and should be handled with 
    care. If time allows a more flexible routine will be created in 
    the future.
    '''

    with open(fname) as f:
        # line 8 has the date
        for i in range(7):
            f.readline()
        # DATE of sonde launch
        fdate=f.readline().strip().split()[0:3] 
        # fdate is a list [YYYY,MM,DD] with entries as strings
        if verbose:
            print(f'Date (YYYYMMDD): {"".join(fdate)}')
        # next we want the information about the data
        f.readline()
        f.readline()
        cnames=[]
        cunits=[]
        col1=f.readline()
        cnames.append(' '.join(col1.strip().split()[0:3]))
        cunits.append(col1.strip().split()[3].replace('(','').replace(')',''))
        # add the other column names and units
        ccnames=['Time after launch', \
                 'Geopotential height', \
                 'Temperature',\
                 'Relative humidity',\
                 'Ozone partial pressure',\
                 'Horizontal wind direction',\
                 'Horizontal wind speed']
 
        ccunits=['s', 'gpm', 'degC', '%', 'mPa','degrees','m/s']
        cnames.extend(ccnames)
        cunits.extend(ccunits)

        cheader=[]
        for m in range(0, len(cnames)):
            cheader.append(cnames[m]+' ('+str(cunits[m])+')')
        if verbose: 
            print (f' HEADER \n {cheader}')
        
        #print (cnames,cunits)
        f.readline()
        numvar=f.readline()
        numvar=int(numvar)
        if verbose:
            print (f' Number of data columns: {numvar}')
        #scale factors of data columns
        vscale=[float(m) for m in f.readline().strip().split()]
        print(vscale)
        # missing values
        misvals=[m for m in f.readline().strip().split()]
        misvalsn=[float(x) if '.' in x else int(x) for x in misvals]
        if verbose:
            print(f' Missing values: {misvalsn}')
        del misvals
        # some more info about the station and the release point
        for i in range(0,42):
            f.readline()
        info=f.readline().strip().split()[:6]
        infon=[float(x) if '.' in x else int(x) for x in info]
        infond=np.asarray(infon, dtype=float).T

        nameinfo=['Number of pressure levels',\
                  'Launch time (UT hours from 0 hours on day given by DATE)',\
                  'East Longitude of station (degrees)',\
                  'Latitude of station (degrees)',\
                  'Wind speed at ground at launch (m/s)',\
                  'Temperature at ground at launch (C)']   
        
        # convert the additional information into a pandas dataframe
        add_info=pd.DataFrame([infon], columns = list(nameinfo))
        if verbose:
            print (f' Information about start conditions: \n {add_info}')
        # read on ....
        for i in range(0,9):
            f.readline()
        # DATA section
        l=f.readlines()
        d=[x.strip().split() for x in l]
        nd=np.asarray(d, dtype=float)
        df = pd.DataFrame(nd, columns = cheader)
        if verbose:
            print(f' Data \n {df}')
        
        '''
        RETURN:
        df          : data (pandas dataframe)
        add_info    : info about start conditions (pandas dataframe)
        numvar      : number of data columns (without pressure, better to use df.shape[0] to get the correct number)
        fdate       : starting date as list (=> should be added to add_info)
        misvalsn    : list with missing values for df (not pressure)
        '''
        return df, add_info, numvar, fdate, misvalsn

if __name__=='__main__':
    '''
    START Get input
    It is possbile to run the script within a loop and 
    provide the filename and path. To do this, execute the script 
    as follows:
    python read_hp_sonde.py -f <filename> -p <path_to_filename>

    You can provide the path and filename also within this script. This is
    then executed, as follows
    python read_hp_sonde.py

    For more help:
    python read_hp_sonde.py -h
    '''
    # get variable names as input
    parser=argparse.ArgumentParser(description=\
                                   'Read NDACC Sonde Data from HP \n'\
                                   )
    #  
    parser.add_argument('-f','--fn', required=False,  \
                    default="None", type=str,\
                    help='filename')
    parser.add_argument('-p','--pn', required=False,\
                        default="/home/dkunkel/tmp_monsun/2023/output/2023_daniel_ndacc/raw_data/HP/", type=str, \
                        help="path to sonde data")
    args=parser.parse_args()
    print ( " \n Input :\n ", args,              '\n' \
            '                                  \n' \
            ' ##################################')
    
    if args.fn != 'None':
        filename=args.pn+args.fn
    else:
        #path="/uni-mainz.de/homes/dkunkel/output/2023_daniel_ndacc/raw_data/HP/"
        path="/home/dkunkel/tmp_monsun/2023/output/2023_daniel_ndacc/raw_data/HP/"
        #path='/home/dkunkel/2023/output/2023_daniel_ndacc/raw_data/HP/'
        filename=path+'ho161031.b05'
    '''
    END Get input
    '''
    print (f' Read file {filename}')
    '''
    return values from read_sonde:
    data: pandas dataframe with observational data
    info: pandas dataframe with additional information about start time, location and start conditions
    ncol: number of columns in data (without pressure)  -> obsolete
    sdate: data of sonde as list: ['YYYY','MM','DD']
    '''
    data, info, ncol, sdate, misvals= read_sonde(filename)
    print (ncol, info, data)
    print(data['Temperature (degC)'])
    # get the initial time as seconds of day
    initime=int(info['Launch time (UT hours from 0 hours on day given by DATE)'].values[0]*3600)

    # rename the columns to make them a bit more handy and safe it in new pd.dataframes
    data2=data.rename({'Pressure at observation (hPa)':'PRES',\
                        'Time after launch (s)':'TIME', \
                        'Temperature (degC)':'TEMP',\
                        'Relative humidity (%)':'RELHUM',\
                        'Ozone partial pressure (mPa)':'OZONE',\
                        'Horizontal wind direction (degrees)':'WINDDIR',\
                        'Horizontal wind speed (m/s)':'WINDSPEED'},\
                        axis=1).copy()
    info2=info.rename({'East Longitude of station (degrees)':'LON','Latitude of station (degrees)':'LAT'}, axis=1).copy()
    # #############################################################
    # this is an example output
    # pressure (hPa) longitude (degE) latitude(degN) time(s of day)
    # #############################################################
    print (f'Date of flight: {sdate}')
    #for i in range(0,data.shape[0]):
    #    print (data2['PRES'][i], info2['LON'].values[0], info2['LAT'].values[0], initime+int(data2['TIME'][i]))
        #print (data['Pressure at observation (hPa)'][i], \
        #    info['East Longitude of station (degrees)'].values[0],\
        #    info['Latitude of station (degrees)'].values[0],\
        #    initime+int(data['Time after launch (s)'][i]))
    # create new dataframe (very complicated)
    # pressure
    vcol1=data2['PRES'].values
    ncol1=data2['PRES'].name
    # time
    vcol2=initime+data2['TIME'].values
    ncol2=data2['TIME'].name
    # lat
    vcol3=np.zeros(data2['PRES'].shape[0])
    vcol3[:]=info2['LAT'].values[0]
    ncol3=info2['LAT'].name
    # lon
    vcol4=np.zeros(data2['PRES'].shape[0])
    vcol4[:]=info2['LON'].values[0]
    ncol4=info2['LON'].name
    # theta in Kelvin
    vcol5=(data2['TEMP'].values+273.15)*(1.e3/data2['PRES'].values)**0.286
    vcol5[data2['TEMP'].values==misvals[2]]=-1.*misvals[2]
    ncol5='THETA'
    # temp in Kelvin
    vcol6=data2['TEMP'].values + 273.15
    ncol6=data2['TEMP'].name
    # relhum
    vcol7=data2['RELHUM'].values
    ncol7=data2['RELHUM'].name
    # ozone
    vcol8=data2['OZONE'].values
    ncol8=data2['OZONE'].name
    # winddir
    vcol9=data2['WINDDIR'].values
    ncol9=data2['WINDDIR'].name
    # windspeed
    vcol0=data2['WINDSPEED'].values
    ncol0=data2['WINDSPEED'].name
    # new data and new column names
    ndata=[vcol1,vcol2,vcol3,vcol4,vcol5,vcol6,vcol7,vcol8,vcol9,vcol0]
    ndata=np.asarray(ndata).T
    ncolumn=[ncol1,ncol2,ncol3,ncol4,ncol5,ncol6,ncol7,ncol8,ncol9,ncol0]
    fd=pd.DataFrame(ndata,columns=ncolumn)
    fd['TIME']=fd['TIME'].astype(int)
    print(fd)
    # save to file
    fd.to_csv('hp'+str(''.join(sdate))+'_converted.txt', index=False, sep='\t', float_format='%.2f')

    plt.plot(fd['TEMP'],fd['PRES'])
    plt.show()

    plt.plot(fd['THETA'],fd['PRES'])
    plt.show()

    plt.plot(fd['OZONE'],fd['PRES'])
    plt.show()


