#!/usr/bin/env python

import subprocess
pSize=4
nCoresPerSlice=2
for i in range(14):
    nSlices=2**i
    nCores=nSlices*nCoresPerSlice
    nMode=16
    if nCores<16: nMode=nCores
    command="s.sub"
    subprocess.call([command,str(pSize),str(nSlices),str(nCores),str(nMode)])
