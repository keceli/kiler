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
#matplotlib.rc('text', usetex=True)
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
solveTime=[]
plotDensity=False
mymarker = itertools.cycle(list('dso*>v<*'))
mymarker2 = itertools.cycle(list('dso*>v<*'))
#mymarker2 = itertools.cycle(list('o^sd*h>v<*'))
mymarker3 = itertools.cycle(list('osd*>v<*'))
mymarker4 = itertools.cycle(list('osd*>v<*'))
mymarker5 = itertools.cycle(list('osd^*>v<*'))
mymarkersize = itertools.cycle([8])
mymarkersize2 = itertools.cycle([8])
mymarkersize3= itertools.cycle([6,7,8,9,10,11])
mycolor = itertools.cycle(list('bgrkcmyk'))
mycolorRGB = itertools.cycle(list('rgbk'))
mycolorRGB2 = itertools.cycle(list('rgbk'))
mycolorRGB3 = itertools.cycle(list('rgbk'))
mycolorRGB4 = itertools.cycle(list('rgbk'))
mycolor5 = itertools.cycle(list('bgrkcmyk'))
mycolor2 = itertools.cycle(list('bgr'))
mycolor3 = itertools.cycle(list('bgr'))
mymarkersize3= itertools.cycle([6,7,8,9,10,11])
mymarkersize4= itertools.cycle([6,7,8,9,10,11])
#mymarker5  = itertools.cycle(list('o^>v<*D'))
#mymarkersize5= itertools.cycle([6,7,8,9,10,11])
#myfillstyle=itertools.cycle(['full','none'])
mymfc=itertools.cycle(['k','"none"'])

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
 
def binning(M, blockSize):
    size=M.shape[0]
    binNbr=int(ceil(size/blockSize))
    matSize=M.shape[0]
    blockSize=ceil(matSize/binNbr)
    bins=np.zeros((binNbr, binNbr), dtype=int)
    for i,j in itertools.izip(M.row, M.col):
        bins[(i//blockSize,j//blockSize)]+=1
    return bins
 
def plotTimings(nCoresPerSlice):
    index=getIndex(nCoresPerSlice)
    if len(index)< 2: return 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[solveTime[i] for i in index]
    plt.plot(x,y,linestyle='None',label=mylabel,marker=next(mymarker),markersize=8) 
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    plt.xlim(min(nCores)*0.7,max(nCores)/0.7)
    plt.ylim(min(solveTime)*0.7,max(solveTime)/0.7)
    plt.legend(loc='lower left')
    plt.title("Matrix length:"+str(max(matrixSize)))

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
    plt.xlim([10,1E6])
    plt.ylim([1,5000])
  #  plt.xlim(min(x)*0.7,max(x)/0.7)
  #  plt.ylim(min(y)*0.7,max(y)/0.7)
    plt.legend(loc='lower left')
  #  plt.legend(loc='upper left')
  #  plt.title(myTitle)
    plt.grid(which='major', axis='both')    
    
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

def oldplotSpeedup(nCoresPerSlice):
    index=getIndex(nCoresPerSlice) 
    mylabel=str(nCoresPerSlice)+" cores per slice,"+" Matrix length:"+str(max(matrixSize))   
    x=[nCores[i] for i in index]
    y=[solveTime[i] for i in index]
    speedup=[]
    speedup=normalize(sorted(y,reverse=True))
    coreup=normalize(sorted(x,))
    plt.figure()
    plt.plot(sorted(x),speedup,linestyle='None',marker='o',color='b',label=mylabel,markersize=8)
    plt.plot(sorted(x),[speedup[0]/coreup[i] for i in range(len(speedup))],linestyle='-',color='k')
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.legend(loc='upper left')
    plt.title(mylabel)
    figname="speedup"+str(nCoresPerSlice)+".png"
    plt.savefig(figname)    
 
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
        figname="efficiency"+str(nCoresPerSlice)+".png"
        plt.savefig(figname)            
        return
    
    plotEfficiency(x,y)
    return

def plotSpeedup(matSize):
    x=[]
    y=[]
    mylabel="Matrix length:"+ str(matSize)
    for i in range(len(solveTime)):
        if matrixSize[i]==matSize:
            x.append(nCores[i])
            y.append(solveTime[i])
    speedup=[]
    speedup=normalize(sorted(y,reverse=True))
    coreup=normalize(sorted(x,))
    plt.figure()    
    plt.plot(sorted(x),speedup,linestyle='None',marker='o',color='b',label=mylabel,markersize=8)
    plt.plot(sorted(x),[speedup[0]/coreup[i] for i in range(len(speedup))],linestyle='-',color='k')
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.xlabel('Number of cores')
    plt.ylabel('Speedup')
    plt.legend(loc='upper left')
    figname="speedup"+str(matSize)+".png"
    plt.savefig(figname) 
    return

def plotStrong(matSize):
    x=[]
    y=[]
    myLabel=r'$N=$' + str(matSize)
    xlabel="Number of cores"
    ylabel="Time to solution (s)"
    for i in range(len(totalTime)):
        if matrixSize[i]==matSize:
            x.append(nCores[i])
     #       y.append(totalTime[i])
            y.append(totalTime[i])
#    plotLogScatter(x,y,"Number of cores","Time to solution (s)",mylabel, "Strong Scaling")
    plt.plot(x,y,linestyle='None',label=myLabel,marker=next(mymarker2),markersize=next(mymarkersize2)) 
    plt.plot(sorted(x),[sorted(y,reverse=True)[0]/(sorted(x)[i]/sorted(x)[0]) for i in range(len(x))],linestyle='-',color=next(mycolor) )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([10,5E5])
    plt.ylim([1,5000])
    return

def plotNoFillData(x,y,myLabel):
#    plt.plot(x,y,linestyle='None',marker=next(mymarker2),markersize=next(mymarkersize2),fillstyle=next(myfillstyle)) 
    myplt,=plt.plot(x,y,label=myLabel,linestyle='None',marker=next(mymarker),markersize=8,mfc='none',color=next(mycolorRGB)) 
    plt.plot(x,[y[0]*(x[i]/x[0]) for i in range(len(x))],linestyle='--',color=next(mycolorRGB2) )
    myplt.set_mec(myplt.get_color()) 
    plt.xscale('log')
    plt.yscale('log')
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.xlim([10,5E5])
#    plt.ylim([1,5000])
    plt.xlim([5E3,1E6])
    return

def plotData(x,y,myLabel):
#    plt.plot(x,y,linestyle='None',marker=next(mymarker2),markersize=next(mymarkersize2),fillstyle=next(myfillstyle)) 
    myplt,=plt.plot(x,y,label=myLabel,linestyle='None',marker=next(mymarker2),markersize=8,color=next(mycolorRGB3)) 
    plt.plot(x,[y[0]*(x[i]/x[0]) for i in range(len(x))],linestyle='--',color=next(mycolorRGB4) )
#    myplt.set_mec(myplt.get_color()) 
    plt.xscale('log')
    plt.yscale('log')
#    plt.xlabel(xlabel)
#    plt.ylabel(ylabel)
#    plt.xlim([10,5E5])
#    plt.ylim([1,5000])

    
    return



def plotLogScatter(x,y,xlabel,ylabel,myLegend,myTitle):  
    #legendEntries=[] 
    #legendText=[]  
    #myplot=
    plt.plot(x,y,linestyle='None',label=myLegend,marker=next(mymarker2),markersize=next(mymarkersize2)) 

 #   plt.xscale('log',basex=2)
 #   plt.yscale('log',basey=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
 #   plt.xlim([5000,1E6])
 #   plt.ylim([5,7200])
    plt.xlim([10,1E6])
    plt.ylim([1,5000])
  #  plt.xlim(min(x)*0.7,max(x)/0.7)
  #  plt.ylim(min(y)*0.7,max(y)/0.7)
  #  plt.legend(loc='lower left')
  #  plt.legend(loc='lower right')
    plt.legend(loc='upper right')
  #  plt.legend(loc='upper left')
 #   plt.title(myTitle)
 #   plt.grid(which='major', axis='both')   

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
    for i in range(len(solveTime)):
        if matrixSize[i]==matSize:
            x.append(nCores[i])
            y.append(solveTime[i])
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


def makeSpeedupPlots(matSize):
 #   matSize=[24000,48000,96000]#,16384]
    map(plotSpeedup,matSize)
    plt.figure()  
  #  plt.title("Efficiency")        
    map(plotEfficiency,matSize)
    plt.axhline(y=100,color='k',linewidth=2)
    plt.savefig("efficiency.png")    
    return

def plotProfile(nCoresPerSlice):
    plt.figure()
    index=getIndex(nCoresPerSlice) 
    mylabel=str(nCoresPerSlice)+" cores per slice"   
    x=[nCores[i] for i in index]
    y=[solveTime[i] for i in index]
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

def plotSizeDependence(cores):
    x=[]
    y=[]
    mylegend=str(cores/16)+" slices"
    for i in range(len(solveTime)):
        if nCores[i]==cores:
            x.append(matrixSize[i])
           # y.append(solveTime[i])
            y.append(solveTime[i])
    x=sorted(x)       
    y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
 #   plotScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Size dependence, 16 cores per slice") 
    plotLogScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Size dependence, 16 cores per slice") 
    
def plotWeak(ratio):
    x=[]
    y=[]
    thresh=0.345
    mylegend="Matrix length / number of cores ~ "+str(ratio)
    for i in range(len(solveTime)):
        if   abs(float(matrixSize[i])/float(nCores[i]) - ratio)/ratio < thresh:
            print matrixSize[i], nCores[i], ratio
            x.append(matrixSize[i])
            y.append(solveTime[i])
    x=sorted(x)       
    y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
    plotLogScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Weak scaling, 16 cores per slice")    
    
def plotWeak2(ratio):
    x=[]
    y=[]
    thresh=0.1
    mylegend="Matrix length / square root of number of cores ~ "+str(ratio)
    for i in range(len(solveTime)):
        if   abs(float(matrixSize[i])/np.sqrt(float(nCores[i])) - ratio)/ratio < thresh:
            print matrixSize[i], nCores[i], ratio
            x.append(matrixSize[i])
            y.append(solveTime[i])
    x=sorted(x)       
    y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
    plotLogScatter(x,y,"Matrix length","Walltime (s)",mylegend, "Weak scaling, 16 cores per slice")  
    
def plotWeak3(ratio):
    x=[]
    y=[]
    thresh=0.1
    myLabel="ratio = "+str(ratio)
    xlabel="Number of cores"
    xlabel="Number of basis functions"
    ylabel="Time to solution (s)"
    for i in range(len(totalTime)):
        if   abs(float(matrixSize[i])/np.sqrt(float(nCores[i])) - ratio)/ratio < thresh:
            print matrixSize[i], nCores[i], ratio
   #         x.append(nCores[i])
            x.append(matrixSize[i])
            y.append(totalTime[i])
 #   x=sorted(x)       
 #   y=sorted(y)
   # plotPolyFit(np.log2(x),np.log2(y),1)       
 #   plotLogScatter(x,y,"Number of processes","Time to solution (s)",mylegend, "Weak scaling") 
    plt.plot(x,y,linestyle='None',label=myLabel,marker=next(mymarker),markersize=next(mymarkersize)) 
 #   plt.xscale('log',basex=2)
 #   plt.yscale('log',basey=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left',prop={'size':12})
    plt.legend(loc='lower right',prop={'size':12})
 #   plt.xlim([5000,1E6])
 #   plt.ylim([5,7200])
    plt.xlim([5E3,1E6])
    plt.ylim([1,5000])      
                

def oldmakeWeakScalingPlots(ratios):
    plt.figure()
 #   map(plotWeak2,ratios)
    map(plotWeak,ratios)
    plt.savefig("weak_16cores.png")
    return

def makeWeakScalingPlots2(ratios):
    plt.figure()
 #   map(plotWeak2,ratios)
    map(plotWeak2,ratios)
    plt.legend(loc='lower right')
    plt.savefig("weak_16cores.png")
    return

def makeWeakScalingPlots3(ratios):
    plt.figure()
 #   map(plotWeak2,ratios)
    map(plotWeak3,ratios)
    #plt.xlim(100,1E5)
    plt.grid(which='major', axis='both')   

    plt.savefig("weak.png")
   # plt.show()
    return

def makeScalingPlots():
    #strong and weak scaling plots
    font = {'weight' : 'normal',
        'size'   : 17}
    matplotlib.rc('font', **font)
    TmatrixSize,TnCores,Tratios,TsolveTime,TtotalTime,Tncoresperslice=np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt',unpack=True,usecols = (1,3,6,8,9,10))
    WmatrixSize,WnCores,Wratios,WsolveTime,WtotalTime,Wncoresperslice=np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_W.txt',unpack=True,usecols = (1,3,6,8,9,10))
    DmatrixSize,DnCores,Dratios,DsolveTime,DtotalTime,Dncoresperslice=np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt',unpack=True,usecols = (1,3,6,8,9,10))

    ratiosT=[2000,1000,500,250,125,62.5]#,192]            
    ratiosW=[2000,1000,500,250,125,62.5]#,192]            
    ratiosD=[2000,1000,500,250,125,62.5]#,192]            
    ratiosT=[2000,1000,500,250]#,192]            
    ratiosW=[2000,1000,500,250]#,192]            
    ratiosD=[2000,1000,500,250]#,192]            

    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(20,6.5))
   # fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(9.5,3.5))
#    fig, axes = plt.subplots(nrows=1, ncols=3)
    plt.setp(axes,xlim=[5E3,1E6],ylim=[5,5000],xscale='log',yscale='log')
    plt.subplot(1,3,1)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5000,1e6)
    plt.ylim(10,1e4)
    plt.grid(which='major', axis='y')
    ylabel="Time to solution (s)"
    plt.ylabel(ylabel)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
        
    
    legendsize=12
    ms=8
    for i in range(len(ratiosT)):
        x=TmatrixSize[np.logical_and(Tratios==ratiosT[i], Tncoresperslice==16)]
        y=TsolveTime[np.logical_and(Tratios==ratiosT[i],  Tncoresperslice==16)]
        x2=TmatrixSize[np.logical_and(Tratios==ratiosT[i], Tncoresperslice==32)]
        y2=TsolveTime[np.logical_and(Tratios==ratiosT[i],  Tncoresperslice==32)]
        myLabel="ratio = "+str(ratiosT[i])
        currentmarker=next(mymarker)
        currentcolor=next(mycolorRGB)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=currentmarker,markersize=ms,mfc=currentcolor,mec=currentcolor)
        plt.plot(x2,y2,linestyle='None',marker=currentmarker,markersize=ms,mfc='none',mec=currentcolor)
    plt.legend(loc='upper right',prop={'size':legendsize})
    plt.subplot(1,3,2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(5000,1e6)
    plt.ylim(10,1e4)
    plt.grid(which='major', axis='y')   
    xlabel="Number of basis functions"
    plt.xlabel(xlabel)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    for i in range(len(ratiosW)):
        x=WmatrixSize[np.logical_and(Wratios==ratiosW[i], Wncoresperslice==16)]
        y=WsolveTime[np.logical_and(Wratios==ratiosW[i],  Wncoresperslice==16)]
        x2=WmatrixSize[np.logical_and(Wratios==ratiosW[i], Wncoresperslice==32)]
        y2=WsolveTime[np.logical_and(Wratios==ratiosW[i],  Wncoresperslice==32)]
        myLabel="ratio = "+str(ratiosW[i])
        currentmarker=next(mymarker)
        currentcolor=next(mycolorRGB)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=currentmarker,markersize=ms,mfc=currentcolor,mec=currentcolor)
        plt.plot(x2,y2,linestyle='None',marker=currentmarker,markersize=ms,mfc='none',mec=currentcolor)    
    plt.legend(loc='upper right',prop={'size':legendsize})
    plt.xlim(5000,1e6)
    plt.ylim(10,1e4)
    plt.subplot(1,3,3)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    mymarker = itertools.cycle(list('o^sd*h>v<*'))   
    for i in range(len(ratiosD)):
        x=DmatrixSize[np.logical_and(Dratios==ratiosD[i], Dncoresperslice==16)]
        y=DsolveTime[np.logical_and(Dratios==ratiosD[i],  Dncoresperslice==16)]
        x2=DmatrixSize[np.logical_and(Dratios==ratiosD[i], Dncoresperslice==64)]
        y2=DsolveTime[np.logical_and(Dratios==ratiosD[i],  Dncoresperslice==64)]
        myLabel="ratio = "+str(ratiosD[i])
        currentmarker=next(mymarker)
        currentcolor=next(mycolorRGB)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=currentmarker,markersize=ms,mfc=currentcolor,mec=currentcolor)
        plt.plot(x2,y2,linestyle='None',marker=currentmarker,markersize=ms,mfc='none',mec=currentcolor)    
    plt.legend(loc='upper right',prop={'size':legendsize})
    plt.xlim(5000,1e6)
    plt.ylim(10,1e4)
    fig.tight_layout()
    plt.savefig("weak.png")
    
    matlistD=[8000,16000,32000,64000]
    matlistW=[8000,32000,128000]
    matlistT=[8000,64000,512000]
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(20,6.5))
  #  fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(9.5,3.5))

    plt.setp(axes,xlim=[10,1E6],ylim=[1,5000],xscale='log',yscale='log')
    plt.subplot(1,3,1)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which='major', axis='y')
    plt.ylabel(ylabel)
    plt.xlim(10,5e5)
    plt.ylim(1,1e4)
    xlabel="Number of cores"
    ylabel="Time to solution (s)"
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
    
    for i in range(len(matlistT)):
        x=TnCores[np.logical_and(TmatrixSize==matlistT[i],Tncoresperslice==16)]
        y=TsolveTime[np.logical_and(TmatrixSize==matlistT[i],Tncoresperslice==16)]
        x1=TnCores[(TmatrixSize==matlistT[i])]
        y1=TsolveTime[(TmatrixSize==matlistT[i])]
        x2=TnCores[np.logical_and(TmatrixSize==matlistT[i],Tncoresperslice==32)]
        y2=TsolveTime[np.logical_and(TmatrixSize==matlistT[i],Tncoresperslice==32)]
        myLabel=r'$N=$' + str(matlistT[i])
        currentmarker=next(mymarker)
        currentcolor=next(mycolorRGB)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=currentmarker,markersize=ms,mfc=currentcolor,mec=currentcolor)
        plt.plot(x2,y2,linestyle='None',marker=currentmarker,markersize=ms,mfc='none',mec=currentcolor)    
        plt.plot(sorted(x1),[sorted(y1,reverse=True)[0]/(sorted(x1)[i]/sorted(x1)[0]) for i in range(len(x1))],linestyle='-',color=currentcolor )

    plt.legend(loc='lower left',prop={'size':legendsize})
    plt.subplot(1,3,2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,5e5)
    plt.ylim(1,1e4)
    plt.grid(which='major', axis='y')   
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
    for i in range(len(matlistW)):
        x=WnCores[np.logical_and(WmatrixSize==matlistW[i],Wncoresperslice==16)]
        y=WsolveTime[np.logical_and(WmatrixSize==matlistW[i],Wncoresperslice==16)]
        x1=WnCores[(WmatrixSize==matlistW[i])]
        y1=WsolveTime[(WmatrixSize==matlistW[i])]
        x2=WnCores[np.logical_and(WmatrixSize==matlistW[i],Wncoresperslice==32)]
        y2=WsolveTime[np.logical_and(WmatrixSize==matlistW[i],Wncoresperslice==32)]
        myLabel=r'$N=$' + str(matlistW[i])
        currentmarker=next(mymarker)
        currentcolor=next(mycolorRGB)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=currentmarker,markersize=ms,mfc=currentcolor,mec=currentcolor)
        plt.plot(x2,y2,linestyle='None',marker=currentmarker,markersize=ms,mfc='none',mec=currentcolor)    
        plt.plot(sorted(x1),[sorted(y1,reverse=True)[0]/(sorted(x1)[i]/sorted(x1)[0]) for i in range(len(x1))],linestyle='-',color=currentcolor )
        
    plt.legend(loc='lower left',prop={'size':legendsize})
    plt.xlabel(xlabel)
    plt.subplot(1,3,3)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,5e5)
    plt.ylim(1,1e4)
    plt.grid(which='major', axis='y')
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolorRGB = itertools.cycle(list('rgbk'))
    for i in range(len(matlistD)):
        x=DnCores[np.logical_and(DmatrixSize==matlistD[i],Dncoresperslice==16)]
        y=DsolveTime[np.logical_and(DmatrixSize==matlistD[i],Dncoresperslice==16)]
        x1=DnCores[(DmatrixSize==matlistD[i])]
        y1=DsolveTime[(DmatrixSize==matlistD[i])]
        x2=DnCores[np.logical_and(DmatrixSize==matlistD[i],Dncoresperslice==64)]
        y2=DsolveTime[np.logical_and(DmatrixSize==matlistD[i],Dncoresperslice==64)]
        myLabel=r'$N=$' + str(matlistD[i])
        currentmarker=next(mymarker)
        currentcolor=next(mycolorRGB)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=currentmarker,markersize=ms,mfc=currentcolor,mec=currentcolor)
        plt.plot(x2,y2,linestyle='None',marker=currentmarker,markersize=ms,mfc='none',mec=currentcolor)    
        plt.plot(sorted(x1),[sorted(y1,reverse=True)[0]/(sorted(x1)[i]/sorted(x1)[0]) for i in range(len(x1))],linestyle='-',color=currentcolor )
    plt.legend(loc='lower left',prop={'size':legendsize})
    fig.tight_layout()
    plt.savefig("strong.png")
    
    # plt.show()
    return

def makeScalingPlotsforAnnualReport():
    font = {'weight' : 'normal',
        'size'   : 8}
    legendsize=6
    matplotlib.rc('font', **font)
    TmatrixSize,TnCores,Tratios,TsolveTime,TtotalTime=np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt',unpack=True,usecols = (1,3,6,8,9,))
    DmatrixSize,DnCores,Dratios,DsolveTime,DtotalTime=np.loadtxt('/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_D.txt',unpack=True,usecols = (1,3,6,8,9,))
    xlabel="Number of cores"
    ylabel="Time to solution (s)"
    matlistD=[8000,16000,32000,64000]
    matlistT=[8000,64000,512000]
    
    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(3,3))
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    plt.setp(axes,xlim=[10,1E6],ylim=[1,5000],xscale='log',yscale='log')
    plt.grid(which='major', axis='y')
    mymarker = itertools.cycle(list('o^sd*h>v<*'))
    mycolor = itertools.cycle(list('bgrkcmyk'))
    
    for i in range(len(matlistT)):
        x=TnCores[(TmatrixSize==matlistT[i])]
        y=TsolveTime[(TmatrixSize==matlistT[i])]
        myLabel=r'$N=$' + str(matlistT[i])
        mc=next(mycolor)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=next(mymarker),markersize=5,mec=mc,mfc=mc)
        plt.plot(sorted(x),[sorted(y,reverse=True)[0]/(sorted(x)[i]/sorted(x)[0]) for i in range(len(x))],linestyle='-',color=mc )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left',prop={'size':legendsize})
    plt.savefig("2014-139-N0_Fig1.png",dpi=300)

    fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(3,3))
    plt.setp(axes,xlim=[10,1E6],ylim=[1,5000],xscale='log',yscale='log')
    plt.grid(which='major', axis='y')
    plt.subplots_adjust(bottom=0.15)
    plt.subplots_adjust(left=0.15)
    mymarker = itertools.cycle(list('o^sd*h>v<*'))   
    mycolor = itertools.cycle(list('bgrkcmyk'))
    mycolor2 = itertools.cycle(list('bgrkcmyk'))
    
    for i in range(len(matlistD)):
        x=DnCores[(DmatrixSize==matlistD[i])]
        y=DsolveTime[(DmatrixSize==matlistD[i])]
        myLabel=r'$N=$' + str(matlistD[i])
        mc=next(mycolor)
        plt.plot(x,y,linestyle='None',label=myLabel,marker=next(mymarker),markersize=5,color=mc,mec=mc,mfc=mc)
        plt.plot(sorted(x),[sorted(y,reverse=True)[0]/(sorted(x)[i]/sorted(x)[0]) for i in range(len(x))],linestyle='-',color=mc )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='lower left',prop={'size':legendsize})
    plt.savefig("2014-139-N0_Fig2.png",dpi=300)
    
    # plt.show()
    return

def makeStrongScalingPlots(matSizeList):
    plt.figure()
    map(plotStrong,matSizeList)
    plt.xlim(10,1E6)
    plt.xlim(10,7E5)
    plt.legend(loc='lower left',prop={'size':12})
    plt.grid(which='major', axis='both')   
    plt.savefig("strong.png")
    #plt.show()
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
    plotLogScatter(matrixSize,solveTime,"Matrix length","Walltime (s)","", "All data")
   # plotScatter(matrixSize,totalTime,"Matrix length","Walltime (s)","", "All data")
    plt.savefig("timings.png")
    return

def makeTimingPlot():
    plt.figure()
    plotScatter(matrixSize,solveTime,"Matrix length","Walltime (s)","", "All data")
    plt.savefig("nanotube.png")
    return

def makeNonzeroPlot():
    mydata=loadData("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_fillin.txt")
    plt.figure()
    x=mydata[0]
    plt.plot(x, np.multiply(x,x),marker='*',label='dense',ms=12,linestyle='-',color='black' )
    plotData(x, mydata[6], 'diamond')
    plotData(x, mydata[5], 'wire')
    plotData(x, mydata[4], 'nanotube')
    plotNoFillData(x, mydata[3], 'diamond')
    plotNoFillData(x, mydata[2], 'nanowire')
    plotNoFillData(x, mydata[1], 'nanotube')
    plt.xlabel("Number of basis functions")
    plt.ylabel("Number of nonzeros")
    plt.xlim([5E3, 1E6])
    plt.ylim([1E5, 1E12])
    plt.grid(which='major', axis='both') 
    #       plt.legend(loc='lower right')  
    plt.legend(loc='upper left', prop={'size':12})
    plt.savefig("fillin.png") 
    return
    
def oldmakeScalingPlots(logFile):                 
    loadSIPsData(logFile)
    ratiosT=[2000,1000,500,250,125,62.5]#,192]            
    ratiosW=[2000,1000,500,250,125,62.5]#,192]            
    ratiosD=[2000,1000,500,250,125,62.5]#,192]            
    matlistD=[8000,16000,32000,64000]
    matlistW=[8000,32000,128000]
    matlistT=[8000,64000,512000]
    makeWeakScalingPlots3(ratiosW)
    makeStrongScalingPlots(matlistW)
    return

def makeoldProfilePlots(logFile):
    data=loadData(logFile)
    x=data[0]
    plt.plot(x,data[2],label='symbolic',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    plt.plot(x,[data[2][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color=next(mycolor) )
    plt.plot(x,data[3],label='numeric',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    plt.plot(x,[data[3][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color=next(mycolor) )
    plt.plot(x,data[1],label='solve',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    plt.plot(x,[data[1][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color=next(mycolor) )
    plt.plot(x,data[4],label='orthogonalization',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    plt.plot(x,[data[4][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color=next(mycolor) )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    plt.xticks(x,x)
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':12})
    plt.savefig('profile.png')
    return

def makeoneSlicePlots():
    qmymarker5 = itertools.cycle(list('osd^*>v<*'))
    mymarker6 = itertools.cycle(list('osd^*>v<*'))
    mycolor5 = itertools.cycle(list('bgrkcmyk'))
    mycolor6 = itertools.cycle(list('bgrkcmyk'))
    nSlices,nCores,totalTime=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_1slice.txt",unpack=True,usecols = (1,2,8))
    T_size,T_nSlices,T_nCores,T_totalTime=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_T.txt",unpack=True,usecols = (1,2,3,9))
 #   mydata=loadData(dataFile)
    x=nCores[(nSlices==1)]
    y=totalTime[(nSlices==1)]
    plt.figure()
    plt.plot(x,y,label='one slice',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    x2=nCores[(nSlices>1)]
    y2=totalTime[(nSlices>1)]
    plt.plot(x2,y2,label='one slice per core',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    x3=sorted(T_nCores[(T_size==8000)])
    y3=sorted(T_totalTime[(T_size==8000)],reverse=True)
    plt.plot(x3,y3,label='one slice per 16 cores',linestyle='None',marker=next(mymarker5),markersize=8,color=next(mycolor5))
    x4=[1,2,4,8,16,64,256,1024,4096,16394]
    plt.plot(x3,[y3[0]/(x3[i]/x3[0]) for i in range(len(x3))],linestyle='-',color='red')
    print [y3[0]/(x3[i]/x3[0]) for i in range(len(x3))]
    print [y3[0]/x3[i]/x3[0] for i in range(len(x3))]
    print [x3[i]/x3[0] for i in range(len(x3))]

    plt.plot(x4,[y[0]/(x4[i]/x4[0]) for i in range(len(x4))],linestyle='-',color='black')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
    plt.ylabel('Time to solution (s)')  
    plt.xticks(x4,map(lambda i : "%d" % i, x4))
    plt.legend(loc='lower left', prop={'size':12})
    plt.xlim([0.5,25000])
    plt.ylim([0.1,5000])
   # plt.show()
    plt.savefig('one_slice.png')
    return

def makeoneSlicePlots2():
    nSlices,nCores,totalTime,solTime,symTime,numTime,ortTime=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_1slice.txt",unpack=True,usecols = (1,2,8,15,16,17,18))
 #   mydata=loadData(dataFile)
    myxlim=[0.8,18000]
    myylim=[0.1,5000]
    mylegendsize=8
    myx=[1,4,16,64,256,1024,4096,16394]
    myxticks=[1,16,256,4096]


    plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=2)#,figsize=(8,6))
    plt.setp(ax,xlim=myxlim,ylim=myylim,xscale='log',yscale='log')
    x=nCores[(nSlices==1)]
    mymarker5 = itertools.cycle(list('osd^*>v<*'))
    mymarker6 = itertools.cycle(list('osd^*>v<*'))
    mycolor5 = itertools.cycle(list('rgbkcmyk'))
    mycolor6 = itertools.cycle(list('rgbkcmyk'))

    plt.subplot(2,2,1)
    y=symTime[(nSlices==1)]
    plt.plot(myx,[y[0]/myx[i]/myx[0] for i in range(len(myx))],linestyle='-',color='black')
    plt.plot(x,y,label='sym. (one slice)',linestyle='None',marker=next(mymarker5),markersize=5,mfc='none',mec='red')
    x2=nCores[(nSlices>1)]
    y2=symTime[(nSlices>1)]
    plt.plot(x2,y2,label='sym. (one slice per core)',linestyle='None',marker=next(mymarker6),markersize=5,mfc='blue',mec='blue')

    plt.ylabel('Walltime (s)')  
    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
    
    plt.subplot(2,2,2)
    y=numTime[(nSlices==1)]
    plt.plot(myx,[y[0]/myx[i]/myx[0] for i in range(len(myx))],linestyle='-',color='black')
    plt.plot(x,y,label='num. (one slice)',linestyle='None',marker=next(mymarker5),markersize=5,mfc='none',mec='red')
    x2=nCores[(nSlices>1)]
    y2=numTime[(nSlices>1)]
    plt.plot(x2,y2,label='num. (one slice per core)',linestyle='None',marker=next(mymarker6),markersize=5,mfc='blue',mec='blue')

    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
    frame1.get_yaxis().set_ticklabels([])

    
    plt.subplot(2,2,3)
    y=solTime[(nSlices==1)]
    plt.plot(myx,[y[0]/myx[i]/myx[0] for i in range(len(myx))],linestyle='-',color='black')
    plt.plot(x,y,label='sol. (one slice)',linestyle='None',marker=next(mymarker5),markersize=5,mfc='none',mec='red')
    x2=nCores[(nSlices>1)]
    y2=solTime[(nSlices>1)]
    plt.plot(x2,y2,label='sol. (one slice per core)',linestyle='None',marker=next(mymarker6),markersize=5,mfc='blue',mec='blue')
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')  
    plt.legend(loc='upper right', prop={'size':mylegendsize})
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
  #  plt.xticks(myxticks,map(lambda i : "%d" % i, myxticks))
    
    plt.subplot(2,2,4)
    y=ortTime[(nSlices==1)]
    plt.plot(myx,[y[0]/myx[i]/myx[0] for i in range(len(myx))],linestyle='-',color='black')
    plt.plot(x,y,label='ort. (one slice)',linestyle='None',marker=next(mymarker5),markersize=5,mfc='none',mec='red')
    x2=nCores[(nSlices>1)]
    y2=ortTime[(nSlices>1)]
    plt.plot(x2,y2,label='ort. (one slice per core)',linestyle='None',marker=next(mymarker6),markersize=5,mfc='blue',mec='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(myxlim)
    plt.ylim(myylim)
    plt.xlabel('Number of cores')
  #  plt.ylabel('Time to solution (s)')  
    frame1 = plt.gca()
    frame1.get_yaxis().set_ticklabels([])  
 #   plt.xticks(myx,map(lambda i : "%d" % i, myx))
    plt.legend(loc='upper right', prop={'size':mylegendsize})
#    plt.ylim([1,5000])
   # plt.show()
 #   plt.setp(ax,xlim=[0.8,18000],ylim=[0.1,5000],xscale='log',yscale='log')
    plt.savefig('one_slice_profile.png')
    return

def oldmakeFactorizationPlots():
    data=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_factorization.txt",unpack=True)
    x=data[0]
    fig, axes = plt.subplots(nrows=2, ncols=1)
    plt.subplot(2,1,1)


    plt.plot(x,data[1],label='symbolic (AMD)',linestyle='None',marker='o',markersize=8,color='black')
    plt.plot(x,[data[1][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='black' )
    plt.plot(x,data[3],label='symbolic (PM)',linestyle='None',marker='s',markersize=8,color='red')
    plt.plot(x,[data[3][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[5],label='symbolic (PS) ',linestyle='None',marker='d',markersize=8,color='blue')
    plt.plot(x,[data[5][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':12})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2,1,2)
    plt.plot(x,data[2]/2.,label='numeric (AMD)',linestyle='None',marker='o',markersize=8,color='black')
    plt.plot(x,[data[2][0]/2./x[i]/x[0] for i in range(len(x))],linestyle='-',color='black')
    plt.plot(x,data[4],label='numeric (PM)',linestyle='None',marker='s',markersize=8,color='red')
    plt.plot(x,[data[4][0]/2./x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[6]/2.,label='numeric (PS) ',linestyle='None',marker='d',markersize=8,color='blue')
    plt.plot(x,[data[6][0]/2./x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':12})
    plt.savefig('factorization.png')
    return

def makeProfilePlots():
    data=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_tube64k_profile.txt",unpack=True)
    x=data[1]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.setp(axes,xlim=[0.8,20],ylim=[0.1,100])
    plt.subplot(2,2,1)
    plt.plot(x,data[3],label='sym. (AMD)',linestyle='None',marker='.',markersize=8,mfc='black',mec='black')
    plt.plot(x,[data[3][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='black' )
    plt.plot(x,data[7],label='sym. (PM)',linestyle='None',marker='+',markersize=8,mfc='none',mec='red')
    plt.plot(x,[data[7][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[11],label='sym. (PS) ',linestyle='None',marker='*',markersize=8,mfc='none',mec='blue')
    plt.plot(x,[data[11][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2,2,2)
    plt.plot(x,data[4],label='num. (AMD)',linestyle='None',marker='.',markersize=8,mfc='black',mec='black')
    plt.plot(x,[data[4][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='black')
    plt.plot(x,data[8],label='num. (PM)',linestyle='None',marker='+',markersize=8,mfc='none',mec='red')
    plt.plot(x,[data[8][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[12],label='num. (PS) ',linestyle='None',marker='*',markersize=8,mfc='none',mec='blue')
    plt.plot(x,[data[12][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
  #  plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    frame2 = plt.gca()
    frame2.get_xaxis().set_ticklabels([])  
    frame2.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

    plt.subplot(2,2,3)
    plt.plot(x,data[2],label='sol. (AMD)',linestyle='None',marker='.',markersize=8,mfc='black',mec='black')
    plt.plot(x,[data[2][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='black' )
    plt.plot(x,data[6],label='sol. (PM)',linestyle='None',marker='+',markersize=8,mfc='none',mec='red')
    plt.plot(x,[data[6][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[10],label='sol. (PS) ',linestyle='None',marker='*',markersize=8,mfc='none',mec='blue')
    plt.plot(x,[data[10][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
  #  frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2,2,4)
    plt.plot(x,data[5],label='ort. (AMD)',linestyle='None',marker='.',markersize=8,mfc='black',mec='black')
    plt.plot(x,[data[5][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='black')
    plt.plot(x,data[9],label='ort. (PM)',linestyle='None',marker='+',markersize=8,mfc='none',mec='red')
    plt.plot(x,[data[9][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[13],label='ort. (PS) ',linestyle='None',marker='*',markersize=8,mfc='none',mec='blue')
    plt.plot(x,[data[13][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    plt.xticks(x,map(lambda i : "%d" % i, x))
    frame4 = plt.gca()
    frame4.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})
    
   # fig.tight_layout()
    plt.savefig('profile.png')
    return

def makeProfilePlots2():
    data=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_tube64k_profile.txt",unpack=True)
    x=data[1]
    fig, axes = plt.subplots(nrows=2, ncols=2)
    plt.setp(axes,xlim=[0.8,20],ylim=[1,100])
    plt.subplot(2,2,1)
    plt.plot(x,data[7],label='sym. (PM)',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    plt.plot(x,[data[7][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[11],label='sym. (PS) ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[data[11][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlim([0.8, 20])
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2,2,2)
    plt.plot(x,data[8],label='num. (PM)',linestyle='None',marker='s',markersize=5,mfc='none',mec='red')
    plt.plot(x,[data[8][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[12],label='num. (PS) ',linestyle='None',marker='s',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[data[12][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
  #  plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    frame2 = plt.gca()
    frame2.get_xaxis().set_ticklabels([])  
    frame2.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

    plt.subplot(2,2,3)
    plt.plot(x,data[6],label='sol. (PM)',linestyle='None',marker='d',markersize=5,mfc='none',mec='red')
    plt.plot(x,[data[6][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[10],label='sol. (PS) ',linestyle='None',marker='d',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[data[10][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
  #  frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10})
 #   plt.savefig('sym_factor.png')
    plt.subplot(2,2,4)
    plt.plot(x,data[9],label='ort. (PM)',linestyle='None',marker='^',markersize=5,mfc='none',mec='red')
    plt.plot(x,[data[9][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,data[13],label='ort. (PS) ',linestyle='None',marker='^',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[data[13][0]/x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of cores')
  #  plt.ylabel('Walltime (s)')
    plt.xticks(x,map(lambda i : "%d" % i, x))
    frame4 = plt.gca()
    frame4.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})
    
   # fig.tight_layout()
    plt.savefig('profile.png')
    return

def makeFactorizationPlots2():
    mytype,ncores,tsym,tnum=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_chol.txt",unpack=True)
    x_PM_D=ncores[(mytype==24)]
    x_PM_W=ncores[(mytype==16)]
    x_PM_T=ncores[(mytype==8)]
    x_PS_D=ncores[(mytype==27)]
    x_PS_W=ncores[(mytype==18)]
    x_PS_T=ncores[(mytype==9)]
    y1_PM_D=tsym[(mytype==24)]
    y1_PM_W=tsym[(mytype==16)]
    y1_PM_T=tsym[(mytype==8)]
    y1_PS_D=tsym[(mytype==27)]
    y1_PS_W=tsym[(mytype==18)]
    y1_PS_T=tsym[(mytype==9)]
    y2_PM_D=tnum[(mytype==24)]
    y2_PM_W=tnum[(mytype==16)]
    y2_PM_T=tnum[(mytype==8)]
    y2_PS_D=tnum[(mytype==27)]
    y2_PS_W=tnum[(mytype==18)]
    y2_PS_T=tnum[(mytype==9)]
    
    fig, axes = plt.subplots(nrows=2, ncols=3)
    plt.setp(axes,xlim=[0.8,80],ylim=[0.1,500])
#    axes.tick_params(axis='x',which='minor',bottom='off')
#    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)

    plt.subplot(2,3,1)
    x1=x_PM_T
    y1=y1_PM_T
    x2=x_PS_T
    y2=y1_PS_T


    plt.plot(x1,y1,label='sym. (PM)',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x2,y2,label='sym. (PS) ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
 #   plt.xscale('log')
#    plt.yscale('log')
    plt.xscale('log', subsx=[])
    plt.yscale('log')
    plt.xticks(x1,map(lambda i : "%d" % i, x1))
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10}) 
    
    plt.subplot(2,3,2)
    x1=x_PM_W
    y1=y1_PM_W
    x2=x_PS_W
    y2=y1_PS_W
    plt.plot(x1,y1,label='sym. (PM)',linestyle='None',marker='s',markersize=5,mfc='none',mec='red')
    plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x2,y2,label='sym. (PS) ',linestyle='None',marker='s',markersize=5,mfc='blue',mec='blue')
    plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
 #   plt.xscale('log')
 #   plt.yscale('log')
    frame1 = plt.gca()
    plt.xscale('log', subsx=[])
    plt.yscale('log')
    plt.xticks(x1,map(lambda i : "%d" % i, x1))
    frame1.get_xaxis().set_ticklabels([])
    frame1.get_yaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10}) 
    
    plt.subplot(2,3,3)
    x1=x_PM_D
    y1=y1_PM_D
    x2=x_PS_D
    y2=y1_PS_D
    plt.plot(x1,y1,label='sym. (PM)',linestyle='None',marker='d',markersize=5,mfc='none',mec='red')
    plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x2,y2,label='sym. (PS) ',linestyle='None',marker='d',markersize=5,mfc='blue',mec='blue')
    plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
    plt.xscale('log', subsx=[])
    plt.yscale('log')
    plt.xticks(x1,map(lambda i : "%d" % i, x1))
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
    frame1.get_yaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10})       

    plt.subplot(2,3,4)
    x1=x_PM_T
    y1=y2_PM_T
    x2=x_PS_T
    y2=y2_PS_T
    plt.plot(x1,y1,label='num. (PM)',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x2,y2,label='num. (PS) ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
    plt.xscale('log', subsx=[])
    plt.yscale('log')
    plt.xticks(x1,map(lambda i : "%d" % i, x1))
  #  plt.xlabel('Number of cores')
    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
  #  frame1.get_xaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10}) 
    
    plt.subplot(2,3,5)
    x1=x_PM_W
    y1=y2_PM_W
    x2=x_PS_W
    y2=y2_PS_W
    plt.plot(x1,y1,label='num. (PM)',linestyle='None',marker='s',markersize=5,mfc='none',mec='red')
    plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x2,y2,label='num. (PS) ',linestyle='None',marker='s',markersize=5,mfc='blue',mec='blue')
    plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
    plt.xscale('log', subsx=[])
    plt.yscale('log')
    plt.xlabel('Number of cores')
    plt.xticks(x1,map(lambda i : "%d" % i, x1))
    frame1 = plt.gca()
 #   frame1.get_xaxis().set_ticklabels([])
    frame1.get_yaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower left', prop={'size':10})
#    plt.xticks(x1,['1','2','4','8','16','32','64'])

    plt.subplot(2,3,6)
    x1=x_PM_D
    y1=y2_PM_D
    x2=x_PS_D
    y2=y2_PS_D
    plt.plot(x1,y1,label='num. (PM)',linestyle='None',marker='d',markersize=5,mfc='none',mec='red')
    plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x2,y2,label='num. (PS) ',linestyle='None',marker='d',markersize=5,mfc='blue',mec='blue')
    plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
    plt.xscale('log', subsx=[])
 #   plt.xlabel('Number of cores')
    plt.xticks(x1,map(lambda i : "%d" % i, x1))
    plt.yscale('log')
#    plt.ylabel('Walltime (s)')
    frame1 = plt.gca()
 #   frame1.get_xaxis().set_ticklabels([])
    frame1.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower left', prop={'size':10})

    fig.tight_layout()

    plt.savefig('factor2.png')
    plt.savefig('factor2.eps')
    return


def powerfit(x,y):
    from scipy import optimize,log10
    logx = log10(x)
    logy = log10(y)
    
    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y: (y - fitfunc(p, x))
    
    pinit = [1.0, -1.0]
    out = optimize.leastsq(errfunc, pinit,
                           args=(logx, logy), full_output=1)
    
    pfinal = out[0]
    covar = out[1]
    
    index = pfinal[1]
    amp = 10.0**pfinal[0]
    
    print 'amp:',amp, 'index', index, 'covar',covar
    
    return [amp * (x**index),amp,index]

def makeFactorizationPlots():
    ptype,psize,tsym,tnum=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_factorization.txt",unpack=True,usecols=(1,2,6,7))

    
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(9.5,3.5))
    plt.setp(axes,ylim=[0.1,1500],xlim=[5000,2e6])

#    axes.tick_params(axis='x',which='minor',bottom='off')
#    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)

    plt.subplot(1,3,1)
    x=psize[ptype==1]
    y1=tsym[ptype==1]
    y2=tnum[ptype==1]

    plt.plot(x,y1,label='sym.',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    plt.plot(x,y2,label='num. ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    myfit=powerfit(x, y1)
    plt.plot(x,myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='-',color='red' )
    myfit=powerfit(x, y2)
    plt.plot(x,myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='-',color='blue' )
  #  plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
 #   plt.xscale('log')
#    plt.yscale('log')
    plt.xscale('log')
    plt.yscale('log')
  #  plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.ylabel('Walltime (s)')
   # plt.xticks(x,map(lambda i : "%d" % i, x))
   # plt.legend(loc='upper left', prop={'size':10})
    plt.legend(loc='lower right', prop={'size':10},ncol=2)
    plt.ylim(0.1,1500)
    plt.xlim(5000,6e5)

    plt.subplot(1,3,2)
    x=psize[ptype==2]
    y1=tsym[ptype==2]
    y2=tnum[ptype==2]
    plt.plot(x,y1,label='sym.',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    #plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x,y2,label='num. ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    myfit=powerfit(x, y1)
    plt.plot(x,myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='-',color='red' )
    myfit=powerfit(x, y2)
    plt.plot(x,myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    frame1 = plt.gca()
    frame1.get_yaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower right', prop={'size':10},ncol=2)
    plt.xlabel('Number of basis functions')
    plt.ylim(0.1,1500)
    plt.xlim(5000,6e5)
    
    plt.subplot(1,3,3)
    x=psize[ptype==3]
    y1=tsym[ptype==3]
    y2=tnum[ptype==3]
    print x,y2
    print y2[0:3]
    plt.plot(x,y1,label='sym.',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    #plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x,y2,label='num. ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    myfit=powerfit(x, y1)
    plt.plot(x,myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='-',color='red' )
    myfit=powerfit(x[0:3], y2[0:3])
    plt.plot(x[0:3],myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
    #plt.xticks(x,map(lambda i : "%d" % i, x))

   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.ylim(0.1,1500)
    plt.xlim(5000,6e5)
    frame1 = plt.gca()
    frame1.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower right', prop={'size':10},ncol=2)
    fig.tight_layout()


    plt.savefig('factor_size.png')
    plt.savefig('factor_size.eps')
    return

def makeFactorizationPlots3():
    #symbolic and numeric factorization
    ptype,psize,tsym,tnum=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/data_factorization.txt",unpack=True,usecols=(1,2,6,7))

    
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(9.5,3.5))
    plt.setp(axes,ylim=[0.1,1500],xlim=[5000,2e6])

#    axes.tick_params(axis='x',which='minor',bottom='off')
#    axes.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    font = {'weight' : 'normal',
        'size'   : 12}
    matplotlib.rc('font', **font)

    plt.subplot(1,3,1)
    x=psize[ptype==1]
    y1=tsym[ptype==1]
    y2=tnum[ptype==1]

    plt.plot(x,y1,label='sym.',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    plt.plot(x,y2,label='num. ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[y1[0]*x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,[y2[0]*x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
  #  plt.plot(x2,[y2[0]/x2[i]/x2[0] for i in range(len(x2))],linestyle='-',color='blue')
 #   plt.xscale('log')
#    plt.yscale('log')
    plt.xscale('log')
    plt.yscale('log')
  #  plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.ylabel('Walltime (s)')
   # plt.xticks(x,map(lambda i : "%d" % i, x))
   # plt.legend(loc='upper left', prop={'size':10})
    plt.legend(loc='lower right', prop={'size':10})
    plt.ylim(0.1,1500)
    plt.xlim(5000,6e5)

    plt.subplot(1,3,2)
    x=psize[ptype==2]
    y1=tsym[ptype==2]
    y2=tnum[ptype==2]
    plt.plot(x,y1,label='sym.',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    #plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x,y2,label='num. ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[y1[0]*x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    plt.plot(x,[y2[0]*x[i]/x[0] for i in range(len(x))],linestyle='-',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    frame1 = plt.gca()
    frame1.get_yaxis().set_ticklabels([])
   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.legend(loc='lower right', prop={'size':10})
    plt.xlabel('Number of basis functions')
    plt.ylim(0.1,1500)
    plt.xlim(5000,6e5)
    
    plt.subplot(1,3,3)
    x=psize[ptype==3]
    y1=tsym[ptype==3]
    y2=tnum[ptype==3]
    print x,y2
    print y2[0:3]
    plt.plot(x,y1,label='sym.',linestyle='None',marker='o',markersize=5,mfc='none',mec='red')
    #plt.plot(x1,[y1[0]/x1[i]/x1[0] for i in range(len(x1))],linestyle='-',color='red' )
    plt.plot(x,y2,label='num. ',linestyle='None',marker='o',markersize=5,mfc='blue',mec='blue')
    plt.plot(x,[y1[0]*x[i]/x[0] for i in range(len(x))],linestyle='-',color='red' )
    myfit=powerfit(x[0:3], y2[0:3])
    plt.plot(x[0:3],myfit[0],label='$y={0:.2e}x^{{{1:.2f}}}$'.format(myfit[1],myfit[2]),linestyle='--',color='blue' )
    plt.xscale('log')
    plt.yscale('log')
    #plt.xticks(x,map(lambda i : "%d" % i, x))

   # plt.xticks(x,map(lambda i : "%d" % i, x))
    plt.ylim(0.1,1500)
    plt.xlim(5000,6e5)
    frame1 = plt.gca()
    frame1.get_yaxis().set_ticklabels([])
    plt.legend(loc='lower right', prop={'size':10})
    fig.tight_layout()


    plt.savefig('factor_size3.png')
    plt.savefig('factor_size3.eps')
    return
def makeSparsityPlots():
    import scipy.io as spio
    import scipy.sparse as sp
    mmdir="/Volumes/s/matrices/matrixmarket/"
    mmfiles=[mmdir+"nanotube2-r_P2_A.mm",mmdir+"nanowire25-2r_P2_A.mm",mmdir+"diamond-2r_P2_A.mm",
             mmdir+"rcm_tube_P2.mm",mmdir+"rcm_wire25-2r_P2.mm",mmdir+"rcm_diamond-2r_P2.mm",
             mmdir+"lower_tube_P2.mm",mmdir+"lower_wire25_2r_P2.mm",mmdir+"lower_diamond_2r_P2.mm"]
    fig, axes = plt.subplots(nrows=3, ncols=3,figsize=(9.5,10.5))
    fig.tight_layout()
    for i in range(9):
        M=spio.mmread(mmfiles[i])
     #   M=sp.eye(8000)
        plt.subplot(3,3,i+1)
        plt.spy(M,marker='.', markersize=1)
        frame1=plt.gca()
        frame1.axes.get_xaxis().set_ticks([])
        frame1.axes.get_yaxis().set_ticks([])
    
    fig.tight_layout()
    plt.savefig("spartsity.png")
    return

def makeSpectrumPlots():
    fig, axes = plt.subplots(nrows=3, ncols=1)
    plt.setp(axes,xlim=[-1,2],ylim=[0,100],yticks=[25, 50,75])
    eigs=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/eigs_diamond-r_P2.txt",unpack=True)
    plt.subplot(3, 1, 1)
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])

 #   plt.xlabel("Eigenvalue spectrum (ev)")
  #  plt.ylabel("Number of eigenvalues")
    plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='red',edgecolor='red')

    eigs=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/eigs_wire25-2r_P2.txt",unpack=True)
    plt.subplot(3, 1, 2)
  #  plt.xlabel("Eigenvalue spectrum (ev)")
    plt.ylabel("Number of eigenvalues")
    plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='green',edgecolor='green')
    plt.xlim([-1,2])
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels([])
    
    eigs=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/eigs_nanotube2-r_P2.txt",unpack=True)
    plt.subplot(3, 1, 3)
    plt.xlabel("Eigenvalue spectrum (ev)")
 #   plt.ylabel("Number of eigenvalues")
    plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='blue',edgecolor='blue')
    plt.xlim([-1,2])

    
 #   fig.tight_layout()

    fig.add_axes([0.55, 0.2, 0.3, 0.1])
    plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='blue',edgecolor='blue')
    plt.xlim([2.0,5.5])
    plt.ylim([0,10])
    plt.xticks([2.0,3.0,4.0,5.0])
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticklabels(['2.0','3.0','4.0','5.0'])
   # plt.xticklabels([])
    plt.yticks([5,10])
    
    plt.savefig("spectra.png")                  
#    plt.show()                  
    return

def makeSpectrumPlots2():
    fig, axes = plt.subplots(nrows=3, ncols=2)
    mycolor = itertools.cycle(list('rgbkcmyk'))
    plt.setp(axes,xlim=[0.1,1000],ylim=[1e-8,0.5])
    plt.setp(axes,xscale='log')
    eigsfile=['eigs_diamond-r_P2.txt','eigs_wire25-2r_P2.txt','eigs_nanotube2-r_P2.txt','eigs_diamond_2r_P4.txt','eigs_wire25_2r_P4.txt','eigs_tube_P4.txt']
    for i in range(6):
        eigs=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/"+eigsfile[i],unpack=True)
        eigs=eigs[eigs<0.2]
        dist=[abs(eigs[j]-eigs[j+1]) for j in range(len(eigs)-1)]
        plt.subplot(3, 2, i+1)
  #      frame1 = plt.gca()
#    frame1.get_xaxis().set_ticklabels([])
        print min(dist), max(dist)
 #   plt.xlabel("Eigenvalue spectrum (ev)")
  #  plt.ylabel("Number of eigenvalues")
   #     plt.hist(dist,bins=100, color='red',edgecolor='red')
        clr=mycolor.next()
        plt.hist(dist,bins = 10 ** np.linspace(np.log10(min(dist)), np.log10(max(dist)), 100), color=clr,edgecolor=clr)
      #  plt.plot(dist,linestyle='None',marker='s',markersize=5,mfc='none',mec='red')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylim([0.1,1000])
        plt.xlim([1e-8,0.5])
#     eigs=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/eigs_wire25-2r_P2.txt",unpack=True)
#     plt.subplot(3, 1, 2)
#   #  plt.xlabel("Eigenvalue spectrum (ev)")
#     plt.ylabel("Number of eigenvalues")
#     plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='green',edgecolor='green')
#     plt.xlim([-1,2])
#     frame1 = plt.gca()
#     frame1.get_xaxis().set_ticklabels([])
#     
#     eigs=np.loadtxt("/Volumes/u/kecelim/Dropbox/work/SEMO/data/eigs_nanotube2-r_P2.txt",unpack=True)
#     plt.subplot(3, 1, 3)
#     plt.xlabel("Eigenvalue spectrum (ev)")
#  #   plt.ylabel("Number of eigenvalues")
#     plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='blue',edgecolor='blue')
#     plt.xlim([-1,2])
# 
#     
#  #   fig.tight_layout()
# 
#     fig.add_axes([0.55, 0.2, 0.3, 0.1])
#     plt.hist(eigs,bins=1000,range=[-1.0,5.5],color='blue',edgecolor='blue')
#     plt.xlim([2.0,5.5])
#     plt.ylim([0,10])
#     plt.xticks([2.0,3.0,4.0,5.0])
#     frame1 = plt.gca()
#     frame1.get_xaxis().set_ticklabels(['2.0','3.0','4.0','5.0'])
#    # plt.xticklabels([])
#     plt.yticks([5,10])
    
    plt.savefig("spectra2.png")                  
#    plt.show()                  
    return
def loadSIPsData(logFile):
    global matrixSize,nEigs,nCores,nSlices,totalTime,solveTime
 #   matrixSize,neigList,nCores,nSlices,totalTime,solveTime=np.loadtxt(logFile,unpack=True)
 #   matrixSize,nCores,nSlices,totalTime,solveTime=np.loadtxt(logFile,unpack=True)
 #   matrixSize,nSlices,nCores,solveTime,totalTime=np.loadtxt(logFile,unpack=True,usecols = (3,4,5,10,11))
#    matrixSize,nSlices,nCores,solveTime,totalTime=np.loadtxt(logFile,unpack=True,usecols = (2,3,4,9,10))
    matrixSize,nSlices,nCores,solveTime,totalTime=np.loadtxt(logFile,unpack=True,usecols = (1,2,3,8,9,))
 #   totalTime=solveTime
   # matrixSize,nSlices,nCores,totalTime,solveTime=np.loadtxt(logFile,unpack=True,usecols = (1,2,3,8,9,))
 #   matrixSize,nEigs,nCores,nSlices,totalTime=np.loadtxt(logFile,unpack=True)  
 #   print matrixSize,nSlices,nCores,solveTime,totalTime
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
     Reads dftb (eigenvalue solver @ https://bitbucket.org/hzhangIIT/dftb-eig14) log file and make timing plots. 
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
    if args.input is not None:
        logFile=args.input
       
        #makeScalingPlots(logFile)
        
       # makeNonzeroPlot(logFile)
        #makeProfilePlots(logFile)
        #makeoneSlicePlots(logFile)

    else:
#        plot121eigenvalues(1000)
  #      makeScalingPlotsforAnnualReport()
     #   makeSparsityPlots()
        makeScalingPlots()
        makeFactorizationPlots2()
        #makeFactorizationPlots3()
  #      makeSpectrumPlots()
  #      makeProfilePlots2()
  #      makeoneSlicePlots()
   #     makeoneSlicePlots2()
   #     makeNonzeroPlot()
        #makeScalingPlots()
   #     makeSpectrumPlots2()


if __name__ == "__main__":
    main()
