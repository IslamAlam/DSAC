# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 17:45:33 2019

@author: ABC
"""

from scipy import ndimage as ndi

from skimage import io
import numpy as np
from skimage.measure import label
gtname='munich_ground_reference.png'
#gtname='munich_370_fp.png'
gtimage=io.imread(gtname)
opname='satellite_munich.png'
#opname='munich_370_op.png'
opimage=io.imread(opname)
height=gtimage.shape[0]
width=gtimage.shape[1]
energy=gtimage
energy_ths = np.where(energy>0,0,1)
labels = label(energy_ths)

num=np.max(labels)
print("Maximum number of lables:" + num)
centroids = ndi.measurements.center_of_mass(energy_ths, labels,np.arange(1,(num+1)))


rows, cols = zip(*centroids)

rows2=np.asarray(rows).astype(int)
cols2=np.asarray(cols).astype(int)
a=np.zeros((width,height))

j=1
for i in range(0,num):
    rows2start=rows2[i]-32
    rows2end=rows2[i]+32
    cols2start=cols2[i]-32
    cols2end=cols2[i]+32
    if (rows2start>=0 and rows2end<=height and cols2start>=0 and cols2end<=width):
        gtimage2=np.where(labels==(i+1),255,0)
        fpimagepatch=gtimage2[rows2start:rows2end,cols2start:cols2end]
        opimagepatch=opimage[rows2start:rows2end,cols2start:cols2end,:]
        fpname='./fp/'+str(j).zfill(3)+'_fp.png'
        opname='./op/'+str(j).zfill(3)+'_op.png'
        io.imsave(fpname,fpimagepatch.astype(np.uint8))
        io.imsave(opname,opimagepatch.astype(np.uint8))
        j=j+1