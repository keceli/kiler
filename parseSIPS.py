#!/usr/bin/env python
import numpy as np
import logging
import itertools

keys=[
      "A:",
      "2:       SIPSSetUp:",
      "3:      SIPSSolve0:",
      "4:      SIPSSolve1:",
      "5:      SIPSSolve2:",
      "Memory:",
      "Flops/sec",
      "-mat_mumps_icntl_23",
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
    #log.nanotube2-r_P6.nprocEps16p256.c16.n16.03s30m23h
    a=logfile.split('.')
    s=a[2].split('p')[0].remove('nprocEps')
    p=a[2].split('p')[1].remove('p')
    c=a[3].remove('c')
    n=a[4].remove('n')
    logging.debug("Reading file {0}".format(logfile))
    with open(logfile) as f:
        while True:          
            line = f.readline()
            if not line: 
                break
            else:
                for i in range(len(keys)):
                    if keys[i] in line:
                        a=line.remove(keys[i]).split()
                        values[i]=a[1]
                if "Error" in line or "ERROR" in line:
                    errorCode="ER"
                if "Performance may be degraded" in line:
                    errorCode="SL"            
        print logfile,errorCode,s,p,c,n,list2Str(values)               
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
     Without any input: parse SIPs log files, log.*. 
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
    print "file error s p c n",list2Str(keys)
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
    else:
        readLogDirectory()
        

if __name__ == "__main__":
    main()
