def normalize(series):
      return [series[0]/float(x) for x in series]
def main():
    import glob
    import matplotlib.pyplot as plt
    filename=[]
    matrixSize=[]
    nSlices=[]
    nProcessors=[]
    totalTime=[]
    i=0
    for file in glob.glob("log.*"):
        errorCode=1
        filename.append(file)
#        print filename[i]
        with open(file) as f:
            a = f.readline().split()
            matrixSize.append( int(a[1]))
            a = f.readline().split()
            nSlices.append(int(str(a[5]).split(',')[0]))
            for line in f:
                if "processors" in line and "with" in line:
                    a=line.split()
                    nProcessors.append(int(a[7]))
                if line.startswith("Time (sec):"):
                    errorCode=0
                    a=line.split()
                    totalTime.append(float(a[2]))
                    break
        if errorCode == 1:
#            print filename[i], " failed"
            nProcessors.append(0)
            totalTime.append(0)        
        print i,matrixSize[i], nProcessors[i],nSlices[i], totalTime[i],filename[i]
        i=i+1
    print i, "files read"
    series1x=[]
    series1y=[]
    series2x=[]
    series2y=[]
    series3x=[]
    series3y=[]      
    for i in range(len(nSlices)):
 #       print i
        if nProcessors[i]/nSlices[i]==4:
            series1x.append(nProcessors[i])
            series1y.append(totalTime[i])
 #           print "4 for slice",i, matrixSize[i], nProcessors[i]/16,16,nSlices[i], totalTime[i],filename[i]

        if nProcessors[i]/nSlices[i]==8:
            series2x.append(nProcessors[i])
            series2y.append(totalTime[i])
 #           print "8 for slice",matrixSize[i],nSlices[i], nProcessors[i], totalTime[i],filename[i]
        if nProcessors[i]/nSlices[i]==16:
            series3x.append(nProcessors[i])
            series3y.append(totalTime[i])
 #           print "16 for slice",matrixSize[i],nSlices[i], nProcessors[i], totalTime[i],filename[i]
 
            
    plt.plot(series1x,series1y,linestyle='None', marker='o',color='k') 
    plt.plot(series2x,series2y,linestyle='None', marker='s',color='b') 
    plt.plot(series3x,series3y,linestyle='None', marker='+',color='g') 
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.figure()
    print series1x
    print sorted(series1x)
    print normalize(sorted(series1x))
    print normalize(sorted(series1y,reverse=True))
    efficiency1=[]
    speedup1=[]
    speedup1=normalize(sorted(series1y,reverse=True))
    plt.plot(sorted(series1x),speedup1,linestyle='None',marker='o',color='k')
    plt.xscale('log',basex=2)
    plt.yscale('log',basey=2)
    plt.figure()
    efficiency1=[speedup1[i]*normalize(sorted(series1x))[i] for i in range(len(series1y))]
    plt.plot(sorted(series1x),efficiency1,linestyle='None',marker='o',color='k')
    plt.xscale('log',basex=2)


    
    
#    plt.show()                  
if __name__ == "__main__":
    main()
