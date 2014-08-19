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
nEigenvalues=[]
option_keys=["eps_nev","eps_tol","sips_interval","interval_redist","mat_mumps_icntl_7","mat_mumps_icntl_13","mat_mumps_icntl_23","mat_mumps_icntl_24","mat_mumps_icntl_28","mat_mumps_icntl_29"]
option_values=["100","1.e-8","-0.8,0.2","0","5","1","0","1","0","0"]
profile=[   
#     "UpdateVectors",
#     "VecMAXPBY",
#     "EPSSetUp",
#     "EPSSolve",
     "MatMult",
     "MatSolve",
     "MatCholFctrSym",
     "MatCholFctrNum",
#     "MatCopy",
#     "MatConvert",
#     "MatAssemblyBegin",
#     "MatAssemblyEnd",
#     "MatGetRow",
#     "MatZeroEntries",
#     "MatAXPY",
#     "VecScale",
#     "VecSet",
#     "VecAssemblyBegin",
#     "VecAssemblyEnd",
#     "VecScatterBegin",
#     "VecScatterEnd",
#     "VecReduceArith",
#     "VecReduceComm",
#     "STSetUp",
#     "STApply",
#     "KSPSetUp",
#     "KSPSolve",
#     "PCSetUp",
#     "PCApply",
#     "IPOrthogonalize",
     "BVOrthogonalize",
#     "IPInnerProduct",
#     "IPApplyMatrix",
#     "DSSolve",
#     "DSVectors",
#     "DSOther"
]
profile_count=[0]*(len(profile))
profile_time=[0.0]*(len(profile))
def list2Str(list1):
    return str(list1).replace('[','').replace(']','').replace(',','')
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
    global filename,matrixSize,nSlices,nCores,totalTime,totalFlops,loadTime,setupTime,solveTime,nevalList,option_keys,option_values,profile,profile_count,profile_time
    nevalList=[]
    errorCode=1
    filename.append(file)
    logging.debug("Reading file {0}".format(file))
    
    with open(file) as f:
        while True:          
            line = f.readline()
            if not line: 
                break
#            """# interesting problem with multiline comments.
#            /home/keceli/lib/dftb-eig14/dftb on a arch-bgq-ibm-opt named Q02-I0-J07.vesta.itd with 256 processors, by Unknown Fri Feb 21 00:38:53 2014
#            Using Petsc Release Version 3.4.3, Oct, 15, 2013
#            """
            else:
                a=line.split()
                if "A:" in line:
                    matrixSize.append( int(a[1]))
                elif "#PETSc Option Table entries:" in line:
                    while True:          
                        line = f.readline()
                        if "#End of PETSc Option Table entries" in line: break
                        else:
                            a=line.split()
                            if "-nprocEPS" in line:
                                nSlices.append(int(a[1]))
                            for i in range(len(option_keys)):
                                if option_keys[i] in line:
                                    if a[1]!=option_values[i]:
                                        option_values[i]=a[1]
                                        print option_keys[i],option_values[i]
                elif "processor" in line and "with" in line:
                    nCores.append(int(a[7]))
                elif line.startswith("["):
                    if "nconv" in line and "in" in line:
                        nevalList.append(int(a[2]))
                    elif "in" in line:
                        nevalList.append(int(a[1]))    
                elif line.startswith("Time (sec):"):
                    errorCode=0
                    totalTime.append(float(a[2]))
                elif line.startswith("Flops/sec:"):                 
                    totalFlops.append(float(a[4]))
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
                for i in range(len(profile)):
                    if line.startswith(profile[i]):
#                        profile_count[i]=int((a[1]))
                        profile_count[i]=int(line[18:24])
                        profile_time[i]=float(line[29:39])
                    
    neig=sum(nevalList)

    nEigenvalues.append(neig)
    if errorCode == 1:
        logging.debug("Failed to read file:{0}".format(file))
        nCores.append(0)
        nSlices.append(0)
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
     #   print neig,"eigenvalues found in",totalTime[-1], "s with", nSlices[-1],"slices and", nCores[-1], "cores."
        print file,matrixSize[-1],nCores[-1],nSlices[-1],nEigenvalues[-1],solveTime[-1],list2Str(profile_count),list2Str(profile_time)
                
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
  #      DataOut = np.column_stack((matrixSize,nEigenvalues,nCores,nSlices,totalTime,mainTime,setupTime,solveTime,finalTime,timeStage3)) 
        DataOut = np.column_stack((matrixSize,nEigenvalues,nCores,nSlices,totalTime,solveTime)) 
        print "File matrixSize nCores nSlices nEigenvalues solveTime MatMult MatSolve MatCholFctrSym MatCholFctrNum VecScatterEnd MatMult MatSolve MatCholFctrSym MatCholFctrNum VecScatterEnd"
        np.savetxt('timings.dat', DataOut) 

if __name__ == "__main__":
    main()
