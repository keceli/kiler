#!/usr/bin/env python2
import matplotlib.pyplot as plt
import numpy as np
import logging
import itertools
filename=[]
matrixSize=[]
nSlices=[]
nCores=[]
totalTime=[]
mainTime=[]
loadTime=[]
setupTime=[]
solveTime=[]
finalTime=[]
nevalList=[] 
plotDensity=False
mymarker = itertools.cycle(list('osv^<>*D'))

def normalize(series):
    return [series[0]/float(x) for x in series]
def getIntegral(p,interval):
    pint=np.polyint(p)
    return pint(interval[1])-pint(interval[0])
def getBound(p,rank,nproc,interval):
    binsize=1.0/nproc
    thresh=binsize/10.0
    step=thresh/10.0
    width=interval[1]-interval[0]
    interval[1]=(rank+1)*binsize+interval[0]
    while True:
        currentbinsize=getIntegral(p,interval)
        diff=currentbinsize-(rank+1)*binsize
        if abs(diff)<thresh:
            return interval[1]
            break
        elif diff < 0:
            interval[1]+=step
        elif diff > 0:
            interval[1]-=step
    return interval[1]    

def getIndex(n):
    #finds indices 
    return [i for i in range(len(nCores)) if nCores[i]/nSlices[i]==n]   
 
def readInterval(s):
    #sample txt "[2112] nconv 15 in [-0.671094 -0.67085], time 147.456"
    a=s.split()
    nconv=int(a[2])       
    left=float(a[4].split("[")[1])   
    right=float(a[5].split("]")[0])
    time=float(a[-1])
    return [nconv,left, right,time]
def readLogDirectory():
    import glob
    global filename
    for file in glob.glob("log.*"):
        readLogFile(file)
    return 0
def readLogFile(file):
    global filename,matrixSize,nSlices,nCores,totalTime,loadTime,setupTime,solveTime,nevalList
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
            elif "processors" in line and "with" in line:
                a=line.split()
                nCores.append(int(a[7]))
#             """
#             [248] nconv 180 in [0.1375 0.16875], time 60.2293
#             [0] nconv 191 in [-0.8 -0.76875], time 61.0265
#             [184] nconv 178 in [-0.14375 -0.1125], time 61.9381
#             """
            elif line.startswith("["):
                a=readInterval(line)
                nevalList.append(a)
#             """
#                                       Max       Max/Min        Avg      Total 
#              Time (sec):           1.301e+02      1.00000   1.301e+02                   
#             """
            elif line.startswith("Time (sec):"):
                errorCode=0
                a=line.split()
                totalTime.append(float(a[2]))
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
                break
    neig=sum([nevalList[i][0] for i in range(len(nevalList)) ])
    if errorCode == 1:
        logging.debug("Failed to read file:{0}".format(file))
        nCores.append(0)
        totalTime.append(0)
        mainTime.append(0)    
        loadTime.append(0)   
        setupTime.append(0)        
        solveTime.append(0)
        finalTime.append(0)
        return 0      
    else:
        logging.debug("Read file:{0} successfully".format(file))
        print neig,"found in",totalTime[-1], "sec with", nSlices[-1],"slices and", nCores[-1], "cores."
                
        return 0     
    
def plotTimings(nCoresPerSlice):
    index=getIndex(nCoresPerSlice)
    if len(index)< 2: return 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[totalTime[i] for i in index]
    plt.plot(x,y,linestyle='None',label=mylabel,marker=next(mymarker),markersize=8) 
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    plt.xlim(min(x)*0.7,max(x)/0.7)
    plt.ylim(min(y)*0.7,max(y)/0.7)
    plt.legend(loc='lower left')

def plotSpeedup(nCoresPerSlice):
    index=getIndex(nCoresPerSlice) 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[totalTime[i] for i in index]
    speedup=[]
    speedup=normalize(sorted(y,reverse=True))
    plt.plot(sorted(x),speedup,linestyle='None',marker='o',color='b',label=mylabel,markersize=8)
    plt.plot(sorted(x),[speedup[0]*2**i for i in range(len(speedup))],linestyle='-',marker='.',color='k',label='ideal')
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.legend(loc='upper left')
    plt.title(mylabel)
    plt.savefig('speedup.png')    
    def plotEfficiency(ncores,timings):
        efficiency=[]
        efficiency=[speedup[i]*normalize(sorted(ncores))[i] for i in range(len(timings))]
        plt.figure()
        plt.plot(sorted(ncores),efficiency,linestyle='None',marker='o',color='b',markersize=8)
        plt.xscale('log',basex=2)
        plt.xlim(min(ncores)*0.7,max(ncores)/0.7)
        plt.ylim(0,1.05)
        plt.xlabel('Number of cores')
        plt.ylabel('Efficiency')
        plt.title(mylabel)        
        plt.savefig('efficiency.png')
        return
    plotEfficiency(x,y)

    
def plotProfile(nCoresPerSlice):
    plt.figure()
    index=getIndex(nCoresPerSlice) 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[totalTime[i] for i in index]
    plt.plot(x,[mainTime[i] for i in index],label='Main',linestyle='None', marker='.',markersize=12,color='k')    
    plt.plot(x,[loadTime[i] for i in index],label='Load',linestyle='None', marker='o',markersize=8,color='r') 
    plt.plot(x,[setupTime[i] for i in index],label='Setup',linestyle='None', marker='s',markersize=8,color='b') 
    plt.plot(x,[solveTime[i] for i in index],label='Solve',linestyle='None',marker='+',markersize=8,color='g') 
    plt.plot(x,[finalTime[i] for i in index],label='Final',linestyle='None',marker='*',markersize=8,color='y') 
    plt.title(mylabel)
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Time (s)')
    plt.legend('best')
    
def plotPolyFit(x,y,n):
    #plot nth order polynomial fit   
    import numpy as np
    z=np.polyfit(x,y,n)
    p=np.poly1d(z)
    mylabel=str(n)+" order polynomial fit"
    plt.plot(x,p(x),linestyle='--',label=mylabel) 
      
def plotEigDist():
    title=filename[0]
    sortedList=sorted(nevalList, key=lambda x:x[2])
    timeList=[sortedList[i][3] for i in range(len(sortedList))]
    histList=[sortedList[i][0] for i in range(len(sortedList))]
    leftList=[sortedList[i][1] for i in range(len(sortedList))]
    interval=[sortedList[0][1],sortedList[-1][2]]
    widthList=[(sortedList[i][2]-sortedList[i][1]) for i in range(len(sortedList))]
    sumneval=sum(histList)
    densityList=[histList[i]/widthList[i]/sumneval for i in range(len(sortedList))]
    order=10
    plt.figure()
    plt.plot(leftList,densityList,linestyle='-')#, marker='o',markersize=8,color='k') 
    plt.ylabel('Probability density')
    plotPolyFit(leftList,densityList,order)  
    plt.title(title)
    plt.legend(loc='upper right')
    plt.savefig('probability.png')

    plt.figure()
    plt.plot(widthList,timeList,linestyle='None', marker='.',color='k')
    plt.xlabel('Slice width')
    plt.ylabel('Walltime (s)')
    plt.title(title)
    plt.savefig('width.png')
    
    plt.figure()
    plt.plot(histList,timeList,linestyle='None', marker='.',color='k')
    plt.xlabel('Number of eigenvalues')
    plt.ylabel('Walltime (s)')
    plt.title(title)
    plt.savefig('neval.png')
    plt.show()
    return    

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
     Reads dftb (eigenvalue solver @ https://bitbucket.org/hzhangIIT/dftb-eig14) log file and make plots. 
     """
     )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Log file to be parsed. All log files will be read if a log file is not specified.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')

    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)
    
    if args.input is not None:
        logFile=args.input
        plotDensity=True
        readLogFile(logFile)
        plotEigDist()
        exit()
    else:
        readLogDirectory()
        plt.figure()
        for i in range(max(nCores)):
            plotTimings(i+1)
        plt.savefig('timing.png')
        plt.figure()
        plotSpeedup(4)
        plotProfile(4)
        plt.savefig('profile.png')

    plt.show()                  
if __name__ == "__main__":
    main()
