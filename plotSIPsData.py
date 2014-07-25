#!/usr/bin/env python
import matplotlib.pyplot as plt
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
nEigs=[]
EPSSolve=[]

plotDensity=False
mymarker  = itertools.cycle(list('o^>v<*D'))
mymarker2 = itertools.cycle(list('o^>v<*D'))
mymarkersize = itertools.cycle([6,7,8,9,10,11])
mymarkersize2= itertools.cycle([6,7,8,9,10,11])
mymarker3  = itertools.cycle(list('o^>v<*D'))
mymarkersize3= itertools.cycle([6,7,8,9,10,11])
mymarker4  = itertools.cycle(list('o^>v<*D'))
mymarkersize4= itertools.cycle([6,7,8,9,10,11])
mymarker5  = itertools.cycle(list('o^>v<*D'))
mymarkersize5= itertools.cycle([6,7,8,9,10,11])

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
 
def plotTimings(nCoresPerSlice):
    index=getIndex(nCoresPerSlice)
    if len(index)< 2: return 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[EPSSolve[i] for i in index]
    plt.plot(x,y,linestyle='None',label=mylabel,marker=next(mymarker),markersize=8) 
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    plt.xlim(min(nCores)*0.7,max(nCores)/0.7)
    plt.ylim(min(EPSSolve)*0.7,max(EPSSolve)/0.7)
    plt.legend(loc='lower left')
    plt.title("Matrix length:"+str(max(matrixSize)))

def plotProfile(nCoresPerSlice):
    plt.figure()
    index=getIndex(nCoresPerSlice) 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[EPSSolve[i] for i in index]
    plt.plot(x,[mainTime[i] for i in index],label='Main',linestyle='None', marker='.',markersize=12,color='k')    
    plt.plot(x,[loadTime[i] for i in index],label='Load',linestyle='None', marker='o',markersize=8,color='r') 
    plt.plot(x,[setupTime[i] for i in index],label='Setup',linestyle='None', marker='s',markersize=8,color='b') 
    plt.plot(x,[solveTime[i] for i in index],label='Solve',linestyle='None',marker='+',markersize=8,color='g') 
    plt.plot(x,[finalTime[i] for i in index],label='Final',linestyle='None',marker='*',markersize=8,color='y') 
    plt.title(mylabel)
   # plt.xlim(min(x)*0.7,max(x)/0.7)
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Time (s)')
    plt.legend(loc='best')
    figname="profile"+str(nCoresPerSlice)+".png"
    plt.savefig(figname)  
      
def plotPolyFit(x,y,n):
    #plot nth order polynomial fit   
    import numpy as np
    z=np.polyfit(x,y,n)
    p=np.poly1d(z)
    mylabel="order "+ str(n) +" polynomial fit"
  #  plt.xscale('log',basex=2)
  #  plt.yscale('log',basey=2)
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

def plotFlops(nCoresPerSlice):
    index=getIndex(nCoresPerSlice)
    if len(index)< 2: return 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[totalFlops[i] for i in index]
    plt.plot(x,y,linestyle='None',label=mylabel,marker=next(mymarker3),markersize=next(mymarkersize3)) 
    plt.xscale('log',basex=10)
    plt.yscale('log',basey=10)
    plt.xlabel('Number of cores')
    plt.ylabel('Total flops/sec')
    plt.xlim(min(nCores)*0.7,max(nCores)/0.7)
    plt.ylim(min(totalFlops)*0.7,max(totalFlops)/0.7)
    plt.legend(loc='upper left')
    plt.title("Matrix length:"+str(max(matrixSize)))    

def plotSpeedup(matSize):
    x=[]
    y=[]
    mylabel="Matrix length:"+ str(matSize)
    for i in range(len(EPSSolve)):
        if matrixSize[i]==matSize:
            x.append(nCores[i])
            y.append(EPSSolve[i])
    speedup=[]
    speedup=normalize(sorted(y,reverse=True))
    coreup=normalize(sorted(x,))
    plt.figure()    
    plt.plot(sorted(x),speedup,linestyle='None',marker='o',color='b',label=mylabel,markersize=8)
    plt.plot(sorted(x),[speedup[0]/coreup[i] for i in range(len(speedup))],linestyle='-',color='k',label='ideal')
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.legend(loc='upper left')
    figname="speedup"+str(matSize)+".png"
    plt.savefig(figname) 
    return

def plotScatter(x,y,xlabel,ylabel,myLegend,myTitle):    
    plt.plot(x,y,linestyle='None',label=myLegend,marker=next(mymarker2),markersize=next(mymarkersize2)) 
 #   plt.xscale('log',basex=2)
 #   plt.yscale('log',basey=2)
#    plt.xscale('log')
#    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([5000,1E6])
    plt.ylim([5,1500])
  #  plt.xlim(min(x)*0.7,max(x)/0.7)
  #  plt.ylim(min(y)*0.7,max(y)/0.7)
    plt.legend(loc='upper left')
#    plt.title(myTitle)
    plt.grid(which='major', axis='both')

def plotLogScatter(x,y,xlabel,ylabel,myLegend,myTitle):    
    plt.plot(x,y,linestyle='None',label=myLegend,marker=next(mymarker2),markersize=next(mymarkersize2)) 
 #   plt.xscale('log',basex=2)
 #   plt.yscale('log',basey=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
 #   plt.xlim([5000,1E6])
 #   plt.ylim([5,7200])
    plt.xlim([100,1E5])
    plt.ylim([5,3600])
  #  plt.xlim(min(x)*0.7,max(x)/0.7)
  #  plt.ylim(min(y)*0.7,max(y)/0.7)
    plt.legend(loc='lower left')
    plt.legend(loc='upper left')
    plt.legend(loc='best')
#    plt.title(myTitle)
    plt.grid(which='major', axis='both')    
    

def plotStrong(matSize):
    x=[]
    y=[]
    mylabel="Matrix length: "+ str(matSize)
    for i in range(len(EPSSolve)):
        if matrixSize[i]==matSize:
            x.append(nCores[i])
            y.append(EPSSolve[i])
    plotLogScatter(x,y,"Number of cores","Walltime (s)",mylabel, "Strong Scaling")
    return

def plot121eigenvalues(n):
    x=xrange(1,n)
    y=[2-2*np.cos(np.pi*k/(n+1)) for k in x]
    plt.plot(x,y,linestyle='None',marker=next(mymarker4),markersize=next(mymarkersize))
    plt.figure()
    plt.hist(y,100)

    plt.show()                  
    return

def plotEfficiency(matSize):
    x=[]
    y=[]
    mylabel="Matrix length: "+ str(matSize)
    for i in range(len(EPSSolve)):
        if matrixSize[i]==matSize:
            x.append(nCores[i])
            y.append(EPSSolve[i])
    speedup=[]
    speedup=normalize(sorted(y,reverse=True))
    coreup=normalize(sorted(x,))    
    efficiency=[]
    efficiency=[100*speedup[i]*normalize(sorted(x))[i] for i in range(len(x))]
    plt.plot(sorted(x),efficiency,linestyle='None',marker=next(mymarker4),markersize=next(mymarkersize4),label=mylabel)

    plt.xscale('log')#,basex=2)
    plt.xlim(100,1E5)
    plt.ylim(0,110)
    plt.xlabel('Number of cores')
    plt.ylabel('Efficiency')
    plt.legend(loc='lower left')
    plt.grid(which='major', axis='both')    
    return
def plotSizeDependence(cores):
    x=[]
    y=[]
    mylegend=str(cores/16)+" slices"
    for i in range(len(EPSSolve)):
        if nCores[i]==cores:
            x.append(matrixSize[i])
           # y.append(EPSSolve[i])
            y.append(EPSSolve[i])
    x=sorted(x)       
    y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
 #   plotScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Size dependence, 16 cores per slice") 
    plotLogScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Size dependence, 16 cores per slice") 
    
def getWeakScaling(ratio):
    x=[]
    y=[]
    thresh=0.345
    mylegend="Matrix length / number of cores ~ "+str(ratio)
    for i in range(len(EPSSolve)):
        if   abs(float(matrixSize[i])/float(nCores[i]) - ratio)/ratio < thresh:
            print matrixSize[i], nCores[i], ratio
            x.append(matrixSize[i])
            y.append(EPSSolve[i])
    x=sorted(x)       
    y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
    plotLogScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Weak scaling, 16 cores per slice")    
    
def getWeakScaling2(ratio):
    x=[]
    y=[]
    thresh=0.1
    mylegend="Matrix length / square root of number of cores ~ "+str(ratio)
    for i in range(len(EPSSolve)):
        if   abs(float(matrixSize[i])/np.sqrt(float(nCores[i])) - ratio)/ratio < thresh:
            print matrixSize[i], nCores[i], ratio
            x.append(matrixSize[i])
            y.append(EPSSolve[i])
    x=sorted(x)       
    y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
    plotLogScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Weak scaling, 16 cores per slice")              

def makeWeakScalingPlots(ratios):
    plt.figure()
 #   map(getWeakScaling2,ratios)
    map(getWeakScaling,ratios)
    plt.savefig("weak_16cores.png")
    return

def makeWeakScalingPlots2(ratios):
    plt.figure()
 #   map(getWeakScaling2,ratios)
    map(getWeakScaling2,ratios)
    plt.savefig("weak_16cores.png")
    return

def makeSpeedupPlots(matSize):
 #   matSize=[24000,48000,96000]#,16384]
    map(plotSpeedup,matSize)
    plt.figure()  
  #  plt.title("Efficiency")        
    map(plotEfficiency,matSize)
    plt.axhline(y=100,color='k',linewidth=2)
    plt.savefig("efficiency.png")    
    return

def makeStrongScalingPlots(matSizeList):
    plt.figure()
    map(plotStrong,matSizeList)
    plt.savefig("strong_16cores.png")
    return

def makeSizeDependencePlots():
    cores=[1024,2048,4096,8192,16384]
    plt.figure()
    map(plotSizeDependence,cores)
    plt.savefig("size_16cores.png")
    return

def makeOverAllPlot():
    plt.figure()
 #   plotLogScatter(matrixSize,totalTime,"Matrix length","Walltime (s)","", "All data")
    plotLogScatter(matrixSize,EPSSolve,"Matrix length","Walltime (s)","", "All data")
   # plotScatter(matrixSize,totalTime,"Matrix length","Walltime (s)","", "All data")
    plt.savefig("timings.png")
    return

def makeTimingPlot():
    plt.figure()
    plotScatter(matrixSize,EPSSolve,"Matrix length","Walltime (s)","", "All data")
    plt.savefig("nanotube.png")
    return

def loadData(logFile):
    global matrixSize,nEigs,nCores,nSlices,totalTime,EPSSolve
    matrixSize,neigList,nCores,nSlices,totalTime,EPSSolve=np.loadtxt(logFile,unpack=True)
 #   matrixSize,nEigs,nCores,nSlices,totalTime=np.loadtxt(logFile,unpack=True)  
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
     Can make several plots for SIPs log, requires a data file obtained with parseSIPsLog.py. 
     """
     )
    parser.add_argument('input', metavar='FILE', type=str, nargs='?',
        help='Data file that has 6 columns: matrix length, number of eigenvalues,nCores,nSlices,totalTime,EPSSolve.')
    parser.add_argument('-d', '--debug', action='store_true', help='Print debug information.')
    parser.add_argument('-n','--nCoresPerSlice', type=int, default=0,nargs='?', 
                        help='Speedup, efficiency and profile plots for specific number of cores per slice')
    return parser.parse_args()  
          
def main():
    args=getArgs()
    initializeLog(args.debug)
    if args.input is not None:
        logFile=args.input
        loadData(logFile)
  #      makeOverAllPlot()
   #     makeSizeDependencePlots()
        ratios=[6,24,96]#,192]    
        ratios2=[8000,4000,2000,1000, 500]#,192]            
        ratios2=[2000, 1000,500, 250]#,192]            
    #    ratios=[np.sqrt(6),np.sqrt(24),np.sqrt(96)]#,192]        

     #   makeWeakScalingPlots2(ratios2)
        matlist1=[64000,128000,512000]
        matlist1=[32000,128000,512000]
        matlist2=[8000,16000,19200]
        matlist3=[8064,16128,31968]
        makeStrongScalingPlots(matlist2)

#        makeSpeedupPlots(matlist1)

        plt.show()                  
    else:
        print "Requires input data file"
        plot121eigenvalues(1000)

if __name__ == "__main__":
    main()
