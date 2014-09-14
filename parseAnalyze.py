#!/usr/bin/env python
import numpy as np
import logging
import itertools

keys=[
      "Length:",
      "Trace:",
      "Norm_1",
      "Norm_Frobenius",
      "Norm_Infinity",
      "Memory:",
      "Memory_Relaxation_icntl14",
      "Nonzeros:",
      "NonzerosAfterAnalysis_20",
      "NonzerosAfterFactorization_29",
      "TotalMemoryAfterAnalysis_17",
      "TotalMemoryAfterAnalysis_19",
      "MaxMemoryAfterAnalysis_16",
      "MaxMemoryAfterAnalysis_18",
      "MatCholeskyFactorSymbolic",
      "MatCholeskyFactorNumeric",
      ]

def list2Str(list1):
    return str(list1).replace('[','').replace(']','').replace(',','').replace("'","").replace(":","")



def readLogDirectory():
    import glob
    global filename
    for myfile in glob.glob("log.*"):
        readLogFile(myfile)
    return 0

def readLogFile(logfile):
    errorCode="OK"
    values=["NA"]*len(keys)
    a=logfile.split('.')
    order=a[2][-1]
    a=a[2].split('c')
    p=a[0].strip('p')
    c=a[1][0:2].strip('o')
    logging.debug("Reading file {0}".format(logfile))
    with open(logfile) as f:
        while True:          
            line = f.readline()
            if not line: 
                break
            else:
                for i in range(len(keys)):
                    if keys[i] in line:
                        a=line.split()
                        values[i]=a[-1]
                if "Error" in line or "ERROR" in line:
                    errorCode="ER"
                if "Performance may be degraded" in line:
                    errorCode="SL"            
        print logfile,errorCode,p,c,order,list2Str(values)               
        return 0     
    

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
     Without any input: reads Analyze log files, log.*. 
     """
     )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-n','--nCoresPerSlice', type=int, default=0,nargs='?', 
                        help='Speedup, efficiency and profile plots for specific number of cores per slice')
    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)
    print "file error p c o",list2Str(keys)
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
    else:
        readLogDirectory()
        

if __name__ == "__main__":
    main()
