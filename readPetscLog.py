  
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
        if errorCode == 1:
            print filename[i], " failed"
            nProcessors.append(0)
            totalTime.append(0)        
        print matrixSize[i],nSlices[i], nProcessors[i], totalTime[i],filename[i]
        i=i+1
    print i, "files read"
    series1x=[]
    series1y=[]
    series2x=[]
    series2y=[]
    series3x=[]
    series3y=[]      
    for i in len(nSlices):
        if nProcessors[i]/nSlices[i]==4:
            series1x.append(nProcessors[i])
            series1y.append(totalTime[i])
        if nProcessors[i]/nSlices[i]==8:
            series2x.append(nProcessors[i])
            series2y.append(totalTime[i])
        if nProcessors[i]/nSlices[i]==16:
            series3x.append(nProcessors[i])
            series3y.append(totalTime[i])
            
    plt.plot(series1x,series1y)                   
if __name__ == "__main__":
    main()
