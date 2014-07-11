#!/usr/bin/env python
import numpy as np
import logging
import itertools
filename=[]
matrixSize=[]
nSlices=[]
nCores=[]
totalTime=[]
totalFlops=[]
mainTime=[]
loadTime=[]
setupTime=[]
solveTime=[]
finalTime=[]
nevalList=[]
neigList=[]
EPSSolve=[] 

 
def readInterval(s):
    #sample txt "[2112] nconv 15 in [-0.671094 -0.67085], time 147.456"
    a=s.split()
    nconv=int(a[2])       
    left=float(a[4].split("[")[1])   
    right=float(a[5].split("]")[0])
    time=float(a[-1])
    return [nconv,left, right,time]

def readNeig(s):
    #sample txt "[2112] nconv 15 in [-0.671094 -0.67085], time 147.456"
    a=s.split()
    nconv=int(a[2])       

    return nconv

def parsenevalList():
    sortedList=sorted(nevalList, key=lambda x:x[2])
    timeList=[sortedList[i][3] for i in range(len(sortedList))]
    histList=[sortedList[i][0] for i in range(len(sortedList))]
    return

def readLogDirectory():
    import glob
    global filename
    for file in glob.glob("log.*"):
        readLogFile(file)
    return 0
def readLogFile(file):
    global filename,matrixSize,nSlices,nCores,totalTime,totalFlops,loadTime,setupTime,solveTime,nevalList,EPSSolve
    nevalList=[]
    errorCode=1
    filename.append(file)
    logging.debug("Reading file {0}".format(file))
    with open(file) as f:
#A: 12000 12000
#Global interval [-0.8, 0.2], nprocEps 32, subinterval 0.03125
        a = f.readline().split()
        matrixSize.append( int(a[1]))
        a = f.readline().split()
        nSlices.append(int(str(a[5]).split(',')[0]))
        while True:          
            line = f.readline()
            if not line: 
                break
#            """# interesting problem with multiline comments.
#            /home/keceli/lib/dftb-eig14/dftb on a arch-bgq-ibm-opt named Q02-I0-J07.vesta.itd with 256 processors, by Unknown Fri Feb 21 00:38:53 2014
#            Using Petsc Release Version 3.4.3, Oct, 15, 2013
#            """
            elif "processor" in line and "with" in line:
                a=line.split()
                nCores.append(int(a[7]))
#             """
#             [248] nconv 180 in [0.1375 0.16875], time 60.2293
#             [0] nconv 191 in [-0.8 -0.76875], time 61.0265
#             [184] nconv 178 in [-0.14375 -0.1125], time 61.9381
#             """
            elif line.startswith("["):
              #  a=readInterval(line)
                a=readNeig(line)
                nevalList.append(a)
#             """
#                          Max       Max/Min        Avg      Total 
# Time (sec):           6.140e+01      1.00000   6.140e+01
# Objects:              3.210e+02      1.24903   2.734e+02
# Flops:                7.478e+09      2.16392   3.723e+09  4.766e+11
# Flops/sec:            1.218e+08      2.16392   6.064e+07  7.762e+09
# MPI Messages:         0.000e+00      0.00000   0.000e+00  0.000e+00
# MPI Message Lengths:  0.000e+00      0.00000   0.000e+00  0.000e+00
# MPI Reductions:       4.300e+02      1.00000               
#             """
            elif line.startswith("Time (sec):"):
                errorCode=0
                a=line.split()
                totalTime.append(float(a[2]))
                line = f.readline()
                line = f.readline()
                line = f.readline()
                a=line.split()
                totalFlops.append(float(a[4]))
#             """
#                                Summary of Stages:   ----- Time ------  ----- Flops -----  --- Messages ---  -- Message Lengths --  -- Reductions --
#                                    Avg     %Total     Avg     %Total   counts   %Total     Avg         %Total   counts   %Total 
#             0:      Main Stage: 4.4960e+01  34.6%  0.0000e+00   0.0%  0.000e+00   0.0%  0.000e+00        0.0%  0.000e+00   0.0% 
#             1:        MatLoads: 1.3023e+00   1.0%  0.0000e+00   0.0%  1.203e+04   0.3%  1.205e+02        2.3%  5.800e+01   2.0% 
#             2:        EPSSetUp: 3.2319e+00   2.5%  0.0000e+00   0.0%  0.000e+00   0.0%  0.000e+00        0.0%  1.900e+01   0.6% 
#             3:        EPSSolve: 8.0590e+01  61.9%  6.1054e+11 100.0%  4.674e+06  99.7%  5.157e+03       97.7%  3.428e+03 115.9% 
#             4:        EPSFinal: 1.1422e-02   0.0%  0.0000e+00   0.0%  0.000e+00   0.0%  0.000e+00        0.0%  0.000e+00   0.0% 
#             """
            elif line.startswith(" 0:      Main Stage:"):
                a=line.split()
                stage0=float(a[3])
                a=f.readline().split()
                stage1=float(a[2])
                a=f.readline().split()
                stage2=float(a[2])
                a=f.readline().split()
                stage3=float(a[2])
                a=f.readline().split()
                stage4=float(a[2])
                mainTime.append(stage0)
                loadTime.append(stage1)
                setupTime.append(stage2)
                solveTime.append(stage3)
                finalTime.append(stage4)
            elif line.startswith("EPSSolve"):
                a=line.split()
                a=a[3]
                EPSSolve.append(float(a[0:10]))
                break
    neig=sum([nevalList[i] for i in range(len(nevalList)) ])
    neig=sum(nevalList)

    neigList.append(neig)
    if errorCode == 1:
        logging.debug("Failed to read file:{0}".format(file))
        nCores.append(0)
        totalTime.append(0)
        totalFlops.append(0)
        mainTime.append(0)    
        loadTime.append(0)   
        setupTime.append(0)        
        solveTime.append(0)
        finalTime.append(0)
        return 0      
    else:
        logging.debug("Read file:{0} successfully".format(file))
        logging.debug(nCores)
        logging.debug(nSlices)
        print neig,"eigenvalues found in",totalTime[-1], "s with", nSlices[-1],"slices and", nCores[-1], "cores."
                
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
     Without any input: reads SIPs (eigenvalue solver @ https://bitbucket.org/hzhangIIT/dftb-eig14) log files, log.*. 
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
    nCoresPerSlice=args.nCoresPerSlice
    if args.input is not None:
        logFile=args.input
        readLogFile(logFile)
    else:
        readLogDirectory()
  #      DataOut = np.column_stack((matrixSize,neigList,nCores,nSlices,totalTime,mainTime,setupTime,solveTime,finalTime,EPSSolve)) 
        DataOut = np.column_stack((matrixSize,neigList,nCores,nSlices,totalTime,EPSSolve)) 

        np.savetxt('timings.dat', DataOut) 

if __name__ == "__main__":
    main()
