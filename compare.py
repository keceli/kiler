#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
from math import ceil

#font = {'family' : 'times',
#        'weight' : 'normal',
#        'size'   : 16}
font = {'weight' : 'normal',
        'size'   : 16}
matplotlib.rc('font', **font)
filename=[]

mymfc=itertools.cycle(['k','"none"'])

def normalize(series):
    return [series[0]/float(x) for x in series]

def getIntegral(p,interval):
    pint=np.polyint(p)
    return pint(interval[1])-pint(interval[0])

def getMaxMin(v):
    print 'length:',len(v)
    print 'max:', max(v)
    print 'min:', min(v)
    return
def loadData(logFile):
    return  np.loadtxt(logFile,unpack=True)

def initializeLog(debug):
    import sys
    if debug: logLevel = logging.DEBUG
    else: logLevel = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(logLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logLevel)

    logging.debug("Start in debug mode:") 

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description=
     """
     Given two files with numbers filled in a column, prints the max and min absolute difference between numbers in the same row.
     """
     )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')

    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)

    file1 = input("First file name: ")
    v1=loadData(file1)
    file2 = input("Second file name: ")
    v2=loadData(file2)
    getMaxMin(v1-v2)
    
        

if __name__ == "__main__":
    main()
