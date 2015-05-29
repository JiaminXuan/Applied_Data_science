import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.ndimage.filters import median_filter as mf
import matplotlib.pylab as pd
d1=sys.argv[1];d2=sys.argv[2]
path=os.path.join(d1,d2)
img_ml=nd.imread(path)

pd.ion()
print pd.isinteractive()

fig1, ax1 = plt.subplots(num=1,figsize=(6,1.0*img_ml.shape[0]/img_ml.shape[1]*6))
fig1.subplots_adjust(0,0,1,1);ax1.grid('off')
ax1.axis('off')
fig1.canvas.set_window_title(d2)
im1 = ax1.imshow(img_ml)
fig1.canvas.draw()

fig2, ax2 = plt.subplots(3,1,num=2,figsize=[14,10])
ax2[0].hist(img_ml[:,:,0].reshape(img_ml.shape[0]*img_ml.shape[1]),bins=256,color='r')
ax2[0].set_xlim([0,256])
ax2[1].hist(img_ml[:,:,1].reshape(img_ml.shape[0]*img_ml.shape[1]),bins=256,color='g')
ax2[1].set_xlim([0,256])
ax2[2].hist(img_ml[:,:,2].reshape(img_ml.shape[0]*img_ml.shape[1]),bins=256,color='b')
ax2[2].set_xlim([0,256])
plt.show(block=False)

print 'yes Im running'
x_y=[[],[]]
print 'still running'
while True:
	print 'always running'
	for j in xrange(2):
		L = [int(round(i)) for i in fig1.ginput()[0]]
		x_y[j]=L
	print x_y
	x_y=zip(*x_y)
	print x_y
	img_crop=img_ml[min(x_y[0]):max(x_y[0]),min(x_y[1]):max(x_y[1]),:]
	if int(min(x_y[0]))==int(max(x_y[0])) and int(min(x_y[1]))==int(max(x_y[1])):
		fig2.clf()
		fig2, ax2 = plt.subplots(3,1,num=2,figsize=[14,10])
		ax2[0].hist(img_ml[:,:,0].reshape(img_ml.shape[0]*img_ml.shape[1]),bins=256,color='r')
		ax2[0].set_xlim([0,256])
		ax2[1].hist(img_ml[:,:,1].reshape(img_ml.shape[0]*img_ml.shape[1]),bins=256,color='g')
		ax2[1].set_xlim([0,256])
		ax2[2].hist(img_ml[:,:,2].reshape(img_ml.shape[0]*img_ml.shape[1]),bins=256,color='b')
		ax2[2].set_xlim([0,256])
		fig2.canvas.draw()
	else:

		fig2.clf()
		fig2, ax2 = plt.subplots(3,1,num=2,figsize=[14,10])
		ax2[0].hist(img_crop[:,:,0].reshape(img_crop.shape[0]*img_crop.shape[1]),bins=256,color='r')
		ax2[0].set_xlim([0,256])
		ax2[1].hist(img_crop[:,:,1].reshape(img_crop.shape[0]*img_crop.shape[1]),bins=256,color='g')
		ax2[1].set_xlim([0,256])
		ax2[2].hist(img_crop[:,:,2].reshape(img_crop.shape[0]*img_crop.shape[1]),bins=256,color='b')
		ax2[2].set_xlim([0,256])
		fig2.canvas.draw()
		 