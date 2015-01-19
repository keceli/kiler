#!/usr/bin/python2.7
#Original source:
#https://redmine.epfl.ch/projects/python_cookbook/wiki/Matrix_sparsity_patterns
# Modified by Murat Keceli on 05/2014.
#TODO: 
#1) Merge two binnings
#2) Read Petsc binary format
#3) Runtime options, binsize, plot options, etc.

from __future__ import division
import scipy.io as spio
import scipy.sparse as spa
import itertools
import numpy as np
import sys
from math import ceil, sqrt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time


def binning(M, blockSize):
    size=M.shape[0]
    binNbr=int(ceil(size/blockSize))
    matSize=M.shape[0]
    blockSize=ceil(matSize/binNbr)
    bins=np.zeros((binNbr, binNbr), dtype=int)
    for i,j in itertools.izip(M.row, M.col):
        bins[(i//blockSize,j//blockSize)]+=1
    return bins

def makeRowHist(M):
    size=M.shape[0]

    nRows=np.zeros((size), dtype=int)
    for i in range(size):
        nRows[i]=len(M.row[[(M.row==(i+1)-1)]])
    print nRows
    plt.figure()    
    plt.hist(nRows,bins=50)
    plt.xlabel('Number of nonzeros in a row')
    plt.ylabel('Number of rows')

    plt.savefig("diamondRowsP2.png")
    plt.figure()  
    plt.plot(nRows,linestyle='None',marker='.',markersize=8,mfc='black',mec='black')    
    plt.show()
    return 

def makeSpyPlot(M):
    matrixLength=M.shape[0]
    blockSize=matrixLength/100
    start = time.time()
    bins=binning(M, blockSize)
 #   timeElapsed = time.time() - start
  #  print("Binning computed in {2} seconds: {0[0]:,}x{0[0]:,} blocks of size {1}x{1} (total: {0[1]:,} blocks)".format((len(bins), len(bins)**2), blockSize,timeElapsed))
    myTitle='diamond2rP2'
    plotSparsity(bins, blockSize, myTitle)
    return

def maxValueBins(M, blockSize):
    size=M.shape[0]
    binNbr=int(ceil(size/blockSize))
    matSize=M.shape[0]
    blockSize=ceil(matSize/binNbr)
    bins=np.zeros((binNbr, binNbr), dtype=float)
    for i,j,k in itertools.izip(M.row, M.col, M.data):
        i_bin=i//blockSize
        j_bin=j//blockSize
        if abs(k) > bins[(i_bin,j_bin)]:
            bins[(i_bin,j_bin)]=k
    return bins

def nnzBlocks(binning):
    count=0
    for val in np.nditer(binning):
        if val!=0:
            count+=1
    return count

def draw(bins, blockSize, myTitle):

    count=nnzBlocks(bins)
    blockNbr=len(bins)

    # CUSTOM COlORMAP BASED ON JET WITH WHITE AT THE BOTTOM
    cdict = {'red': ((0., 1, 1),
                 (1e-60, 0, 0),
                 (0.11, 0, 0),
                 (0.66, 1, 1),
                 (0.89, 1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0., 1, 1),
                   (1e-60, 0, 0),
                   (0.11, 0, 0),
                   (0.375, 1, 1),
                   (0.64, 1, 1),
                   (0.91, 0, 0),
                   (1, 0, 0)),
         'blue': ((0., 1, 1),
                  (1e-60, 1, 1),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65, 0, 0),
                  (1, 0, 0))}

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    plt.figure()
    ax=plt.subplot(111)
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticks([])
    frame1.get_yaxis().set_ticks([])
  #  im=plt.imshow(bins, cmap=my_cmap, interpolation='none', vmin=0.0)
    im=plt.imshow(bins, cmap=cm.binary)

#    plt.title("{2} \n - {0:,}x{0:,} blocks of size {3:,}x{3:,} \n ({1:,} non-zero blocks )".format(blockNbr, count, fname, blockSize))
#    plt.title(myTitle)
    # COLORBAR OF THE RIGHT SIZE
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=.5)
    plt.colorbar(im, cax=cax)
    
    #plt.savefig(myTitle+".png") 
    plt.savefig(myTitle+".eps") 
def plotSparsity(bins, blockSize,mytitle):


    # CUSTOM COlORMAP BASED ON JET WITH WHITE AT THE BOTTOM
    cdict = {'red': ((0., 1, 1),
                 (1e-60, 0, 0),
                 (0.11, 0, 0),
                 (0.66, 1, 1),
                 (0.89, 1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0., 1, 1),
                   (1e-60, 0, 0),
                   (0.11, 0, 0),
                   (0.375, 1, 1),
                   (0.64, 1, 1),
                   (0.91, 0, 0),
                   (1, 0, 0)),
         'blue': ((0., 1, 1),
                  (1e-60, 1, 1),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65, 0, 0),
                  (1, 0, 0))}

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)

    plt.figure()
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticks([])
    frame1.get_yaxis().set_ticks([])
    ax=plt.subplot(111)
#    im=plt.imshow(bins, cmap=my_cmap, interpolation='none', vmin=0.0, vmax=2000.0)
    plt.imshow(bins, vmin=0, vmax=6400, cmap=my_cmap)
    plt.colorbar()
    plt.savefig(mytitle+".sparsity.png")
    plt.figure()
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticks([])
    frame1.get_yaxis().set_ticks([])
    ax=plt.subplot(111)
    plt.imshow(bins, vmin=0, vmax=6400, cmap=cm.binary)
    plt.colorbar()
    plt.savefig(mytitle+".bw_sparsity.png")         
    
def plotSpy(M,eps):
    plt.figure()
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticks([])
    frame1.get_xaxis().set_ticklabels([])
    frame1.get_yaxis().set_ticks([])
    frame1.get_yaxis().set_ticklabels([])
    plt.subplot(111)
  #  plt.title(myTitle+",eps="+str(eps))
    plt.spy(M, precision=eps, marker='.', markersize=1)
    plt.show
    #plt.savefig(myTitle+".spy.png")
    
def makeTest(M):
    import rcm
    import sparse
    Mcsr=M.tocsr()
    plt.figure()
    plt.spy(Mcsr)
    p=rcm._reverse_cuthill_mckee(Mcsr.indices,Mcsr.indptr,M.shape[0])
    Mrcm=sparse.sp_permute(Mcsr,p,p)
    print 'Bandwidth before',sparse.sp_bandwidth(Mcsr)
    print 'Bandwidth after',sparse.sp_bandwidth(Mrcm)
    plt.figure()
    plt.spy(Mrcm)      
    plt.show()
    return

    

def doall():
    import glob
    import rcm
    import sparse
    for f in glob.glob("*.mm"):
        start = time.time()
        M = spio.mmread(f).tocsr()
        timeElapsed = time.time() - start
        print 'file:',f
        print 't_read:',timeElapsed
        print 'length:',M.shape[0]
        print 'nnz:',M.nnz
        print 'explicit_zeros:',len(M.data[(M.data==0)])
        print 'min:',min(abs(M.data[(M.data!=0)]))
        print 'max:',max(abs(M.data))
        print 'bandwidth:',sparse.sp_bandwidth(M)
        p=rcm._reverse_cuthill_mckee(M.indices,M.indptr,M.shape[0])
        Mrcm=sparse.sp_permute(M,p,p)
        print 'rcm_bandwidth:',sparse.sp_bandwidth(Mrcm)           
    return
def getCouplings(myfile): 
    with open(myfile) as f:
        f.readline()      
        line = f.readline() 
        tmp=line.split()
        nnz=int(tmp[2])
        v=[0]*10
        for i in range(nnz):
            line = f.readline() 
            tmp=line.split()
            r=int(tmp[0])-1
            c=int(tmp[1])-1
            if r%4==0:
                if c%4==0:
                    v[0]+=1
                elif c%4==1:
                    v[1]+=1    
                elif c%4==2:
                    v[2]+=1    
                elif c%4==3:
                    v[3]+=1
            elif r%4==1: 
                if c%4==1:
                    v[4]+=1
                elif c%4==2:
                    v[5]+=1    
                elif c%4==3:
                    v[6]+=1 
            elif r%4==2: 
                if c%4==2:
                    v[7]+=1
                elif c%4==3:
                    v[8]+=1 
            elif r%4==3: 
                if c%4==3:
                    v[9]+=1
   # print v                
    plt.plot(v,linestyle='none', marker='o',color='r')
    plt.show()
    return

def getNnzAtoms(myfile):
    with open(myfile) as f:
        f.readline()      
        line = f.readline() 
        tmp=line.split()
        nnz=int(tmp[2])
        natoms=int(tmp[0])//4
        v=np.array([0]*natoms)
        vself=np.array([0]*natoms)
        
        for i in range(nnz):
            line = f.readline() 
            tmp=line.split()
            r=int(tmp[0])-1
            c=int(tmp[1])-1
            ratom=r//4
            catom=c//4
            if ratom==catom:
                vself[ratom]+=1
            else:
                v[ratom]+=1
                v[catom]+=1
    print myfile,vself[(vself!=4)],vself[0]   
    print myfile,v[(v>2000)],max(v),np.argmax(v)  
    return v         
    plt.figure()
    plt.plot(v,linestyle='none', marker='o',color='r')
    plt.figure()
    plt.plot(vself,linestyle='none', marker='o',color='r')
    plt.show()
    return

def getNnzPairs(myfile):
    with open(myfile) as f:
        f.readline()      
        line = f.readline() 
        tmp=line.split()
        nnz=int(tmp[2])
        natoms=int(tmp[0])//4
        v=np.array([0]*natoms)
        vself=np.array([0]*natoms)
        
        for i in range(nnz):
            line = f.readline() 
            tmp=line.split()
            r=int(tmp[0])-1
            c=int(tmp[1])-1
            ratom=r//4
            catom=c//4
            if ratom==catom:
                vself[ratom]+=1
            else:
                v[ratom]+=1
                v[catom]+=1
    print myfile,vself[(vself!=4)],vself[0]   
    print myfile,v[(v>2000)],max(v),np.argmax(v)  
    return v         
    plt.figure()
    plt.plot(v,linestyle='none', marker='o',color='r')
    plt.figure()
    plt.plot(vself,linestyle='none', marker='o',color='r')
    plt.show()
    return

def getAtomMatrix(myfile):
    with open(myfile) as f:
        f.readline()      
        line = f.readline() 
        tmp=line.split()
        nnz=int(tmp[2])
        length=int(tmp[0])
        natoms=length//4
        rows=np.array([0]*length)
        v=np.array([0]*natoms)
        vself=np.array([0]*natoms)
        mat=np.zeros((natoms,natoms),dtype=np.int)
        for i in range(nnz):
            line = f.readline() 
            tmp=line.split()
            r=int(tmp[0])-1
            c=int(tmp[1])-1
            rows[r]+=1
            if r!=c:
                rows[c]+=1
            ratom=r//4
            catom=c//4
            mat[ratom][catom]+=1
            if ratom==catom:
                vself[ratom]+=1
            else:
                v[ratom]+=1
                v[catom]+=1       

 #   plt.figure()
 #   plt.plot(rows,linestyle='none', marker='o',color='k')
    return spa.coo_matrix(mat)

def makeNnzDist():
    import parseXYZ
    matD=getAtomMatrix('/Volumes/s/matrices/matrixmarket/diamond-2r_P2_A.mm')
    matW=getAtomMatrix('/Volumes/s/matrices/matrixmarket/nanowire25-2r_P2_A.mm')
    nnzD=matD.nnz
    nnzW=matW.nnz
    distD=np.array([0.]*nnzD)
    distW=np.array([0.]*nnzW)
    xyzD=parseXYZ.getXYZ('diamond-2r_P2.xyz')
    xyzW=parseXYZ.getXYZ('nanowire25-2r_P2.xyz')
    
    for i in range(matW.nnz):
        if i < matD.nnz:
            atom1=matD.row[i]
            atom2=matD.col[i]
            distD[i]=parseXYZ.getdistance(xyzD[1][atom1], xyzD[2][atom1], xyzD[3][atom1], xyzD[1][atom2], xyzD[2][atom2], xyzD[3][atom2])
        atom1=matW.row[i]
        atom2=matW.col[i]
        distW[i]=parseXYZ.getdistance(xyzW[1][atom1], xyzW[2][atom1], xyzW[3][atom1], xyzW[1][atom2], xyzW[2][atom2], xyzW[3][atom2])    
    print 'done'
    plt.figure()
    plt.hist(distD,bins=1000,cumulative=False,normed=True,histtype=u'step',color='red',edgecolor='red',alpha=0.7)
    plt.hist(distW,bins=1000,cumulative=False,normed=True,histtype=u'step',color='green',edgecolor='green',alpha=0.7)
    plt.show()
    return

def makeNnzHist(mat,xyz):
    import parseXYZ
    nnz=mat.nnz
    dist=np.array([0.]*nnz)
    
    for i in range(nnz):
        atom1=mat.row[i]
        atom2=mat.col[i]
        dist[i]=parseXYZ.getdistance(xyz[1][atom1], xyz[2][atom1], xyz[3][atom1], xyz[1][atom2], xyz[2][atom2], xyz[3][atom2])    
    plt.figure()
    plt.hist(dist,bins=1000,cumulative=False,normed=True,histtype=u'step',color='red',edgecolor='red',alpha=0.7)
    return

def getDistMat(xyz,thresh):
    import parseXYZ
    natoms=len(xyz[1])
    mat=np.zeros((natoms,natoms),dtype=np.int)
    centerlist=[]
    for i in range(natoms-1):
        for j in range(i+1,natoms):
            tmp=parseXYZ.getdistance(xyz[1][i], xyz[2][i], xyz[3][i], xyz[1][j], xyz[2][j], xyz[3][j])  
            if tmp<thresh:
                mat[i][j]=tmp
                if abs(i-j)<152 and abs(i-j)>148:
                    centerlist.append(i) 
                    centerlist.append(j) 
    print 'jmol command:',str(['C'+str(centerlist[i]+1) for i in range(len(centerlist))]).replace('[','').replace(']','').replace("'","")                         
    return spa.coo_matrix(mat)

def makeCouplingAtomPlot(n,tag):
    import parseXYZ
    from mpl_toolkits.mplot3d import Axes3D

    mmdir='/Volumes/s/matrices/matrixmarket/'
    mmfile=mmdir+tag+'_A.mm'
    xyzdir='/Volumes/s/keceli/Dropbox/work/SEMO/xyz/'
    xyzfile=xyzdir+tag+'.xyz'
    mat=getAtomMatrix(mmfile)
    nnz=mat.nnz
    distW=np.array([0.]*nnz)
    xyz=parseXYZ.getXYZ(xyzfile)
    matD=getDistMat(xyz,5.6)
    makeNnzHist(mat,xyz)
    xyzW=np.asarray(xyz[1:4])
    myrow=np.asarray(mat.row)
    mycol=np.asarray(mat.col)
    listofAtoms1=myrow[mycol==n]
    listofAtoms2=mycol[myrow==n]
    listofAtoms=np.unique(np.concatenate((listofAtoms1,listofAtoms2)))
    print 'focused atom number and its coordinates:',n+1, xyzW[:,n]
    print 'number of atoms interacting with this atom:',len(listofAtoms)
    print 'list of interacting atoms:',listofAtoms+1
    print 'jmol command:',str(['C'+str(listofAtoms[i]+1) for i in range(len(listofAtoms))]).replace('[','').replace(']','').replace("'","")
    print 'coordinates of interacting atoms',xyzW[:,listofAtoms]
    fig = plt.figure()
    plt.spy(mat,markersize=1)
    fig = plt.figure()
    plt.spy(matD,markersize=1)
    fig = plt.figure()
    plt.plot(listofAtoms,linestyle='None',marker='s')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(xyzW[0,:],xyzW[1,:],xyzW[2,:],linestyle='None',marker='.',color='black',markersize=2) 
    plt.plot(xyzW[0,listofAtoms],xyzW[1,listofAtoms],xyzW[2,listofAtoms],linestyle='None',marker='o',color='red',markersize=3) 
    plt.plot([xyzW[0,n]],[xyzW[1,n]],[xyzW[2,n]],linestyle='None',marker='s',color='blue',markersize=4) 
    plt.grid(which='major', axis='all')
    plt.gca().set_aspect('equal', adjustable='box')  
#    ax.auto_scale_xyz()
    #plt.axis('equal')
    ax.set_xlim([-5,25])
    ax.set_ylim([-5,25])
    ax.set_zlim([0,45])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return
    
def makeNnzAtomsPlots():
    plt.figure()
    plt.plot(getNnzAtoms('diamond-2r_P2_A.mm'),label='diamond',linestyle='None',marker='d',markersize=4,mfc='red',mec='red',alpha=0.7)
    plt.figure()
    plt.plot(getNnzAtoms('nanowire25-2r_P2_A.mm'),label='nanowire',linestyle='None',marker='s',markersize=4,mfc='green',mec='green',alpha=0.7)
    plt.show()
    return 

def makeAtomsPlots():
    M2=getAtomMatrix('diamond-2r_P2_A.mm')
    plt.figure()
    plt.plot(M2.data,label='diamond',linestyle='None',marker='d',markersize=4,mfc='red',mec='red',alpha=0.7)
    M2=getAtomMatrix('nanowire25-2r_P2_A.mm')
    plt.figure()
    plt.plot(M2.data,label='diamond',linestyle='None',marker='d',markersize=4,mfc='green',mec='green',alpha=0.7)
    plt.show()
    return  
             
if __name__ == "__main__":
    
#    doall()
  #  fname=sys.argv[1]


  #  getCouplings(fname)
  
   # makeAtomsPlots()
   # makeNnzDist()
    n=0#16
#    makeNnzDist()
    makeCouplingAtomPlot(n,'nanotube2-r_P2')
    n=0#1050
    makeCouplingAtomPlot(n,'diamond-2r_P2')
    n=0#1012
    makeCouplingAtomPlot(n,'nanowire25-2r_P2')
    plt.show()
#     start = time.time()
#     M = spio.mmread(fname)
#     timeElapsed = time.time() - start
#     matrixLength=M.shape[0]
#     matrixNNZ=M.nnz
# 
#     print("Loaded: {0} in {3} seconds (size {1:,}, {2:,} non-zeros)".format(fname, matrixLength, matrixNNZ, timeElapsed))
#     print 'min:',min(abs(M.data[(M.data!=0)]))
#     print 'max:',max(abs(M.data))
#  #   myTitle=sys.argv[2]
#    # plotSpy(M,0,myTitle)
#  #   plotSpy(M,0.1,myTitle)
#    # makeRowHist(M)
#     makeTest(M)
    
   # myTitle="bins_of_max_values_"+fname
   # plotSparsity(bins, blockSize, myTitle)
   


