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
def draw2(bins, blockSize, myTitle):


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
    print 'test'
#    im=plt.imshow(bins, cmap=my_cmap, interpolation='none', vmin=0.0, vmax=2000.0)
    im=plt.imshow(bins, vmin=0, vmax=3000.0, cmap=my_cmap)
 #   im=plt.imshow(bins, cmap=my_cmap, interpolation='none', norm=matplotlib.colors.LogNorm(vmin=0.01, vmax=1))

#    plt.title("{2} \n - {0:,}x{0:,} blocks of size {3:,}x{3:,} \n ({1:,} non-zero blocks )".format(blockNbr, count, fname, blockSize))
 #   plt.title(myTitle)
    # COLORBAR OF THE RIGHT SIZE
 #   divider = make_axes_locatable(ax)
 #   cax = divider.append_axes("right", size="5%", pad=.5)
  #  plt.colorbar(im, cax=cax)
    plt.colorbar()
    plt.savefig(myTitle+".eps")     
    
def plotSpy(M,eps,myTitle):
    plt.figure()
    frame1 = plt.gca()
    frame1.get_xaxis().set_ticks([])
    frame1.get_yaxis().set_ticks([])
  #  plt.title(myTitle+",eps="+str(eps))
    plt.spy(M, precision=eps, marker='.', markersize=3)
    plt.savefig(myTitle+"eps"+str(eps)+".eps")   

if __name__ == "__main__":

    fname=sys.argv[1]
  #  blockSize=int(sys.argv[2])
    start = time.time()
    M = spio.mmread(fname)
    timeElapsed = time.time() - start
    matrixLength=M.shape[0]
    blockSize=matrixLength/100
    matrixNNZ=M.nnz
    #myTitle=fname+"\n Matrix length="+ str(matrixLength)+", Number of nonzeros="+str(matrixNNZ)

    print("Loaded: {0} in {3} seconds (size {1:,}, {2:,} non-zeros)".format(fname, matrixLength, matrixNNZ, timeElapsed))
    myTitle="spy_"+fname
 #   plotSpy(M,0,myTitle)
 #   plotSpy(M,0.1,myTitle)
    start = time.time()
    bins=binning(M, blockSize)
#    bins=maxValueBins(M, blockSize)
    timeElapsed = time.time() - start
    print("Binning computed in {2} seconds: {0[0]:,}x{0[0]:,} blocks of size {1}x{1} (total: {0[1]:,} blocks)".format((len(bins), len(bins)**2), blockSize,timeElapsed))
    myTitle="bins_of_max_values_"+fname
    draw2(bins, blockSize,myTitle)
    #start = time.time()
   # bins=binning(M, blockSize)
   # timeElapsed = time.time() - start
   # print("Binning computed in {2} seconds: {0[0]:,}x{0[0]:,} blocks of size {1}x{1} (total: {0[1]:,} blocks)".format((len(bins), len(bins)**2), blockSize,timeElapsed))
    myTitle="bins_of_nnz_values_"+fname
   # draw(bins, blockSize,myTitle)

