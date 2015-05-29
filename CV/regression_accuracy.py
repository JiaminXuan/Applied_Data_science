import os
import sys
import time
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
import math
img = nd.imread('images/digits.png')
imgM = img.reshape(50,20,100,20).transpose(0,2,1,3).reshape(5000,20,20)

fig0, ax0 = plt.subplots(num=0,figsize=[5,5])
fig0.subplots_adjust(0,0,1,1)
ax0.axis('off')
img0 = ax0.imshow(imgM[0])
fig0.canvas.draw()

# fig0.show()

imgmean = np.array([imgM[i*500:(i+1)*500].mean(0) for i in range(10)])

fDict = {}
fail = []
PT = imgmean.reshape(10,400)
P  = PT.transpose()
PTPinv = np.linalg.inv(np.dot(PT,P))
for j in xrange(10):
	wrong = 0
	wDict = {}
	coef = []

	for i in xrange(500):
		index = j * 500 + i
		samp = imgM[index]
		PTyy = np.dot(PT,samp.flatten())
		avec = np.dot(PTPinv,PTyy)
		coef.append(avec)

		if np.argmax(avec) != j:
			fDict[index] = np.argmax(avec)
			fail.append(index)
			wrong += 1
			if np.argmax(avec) in wDict:
				wDict[np.argmax(avec)] += 1
			else:
				wDict[np.argmax(avec)] = 1
				
	fig1, ax1 = plt.subplots(10,1,figsize=[8,9], sharex=True,sharey=True)

	
	coef = np.vstack(coef)
	coef = coef.T

	for p in xrange(10):
		ax1[p].hist(coef[p], bins = 100,edgecolor='r')
		[i.get_yaxis().set_visible(False) for i in ax1]
		ax1[p].set_title("distribution of regression coef of %s's against %s's"% (j,str(p)),fontsize=11)
	fig1.subplots_adjust(hspace=2)
	# fig1.subplots_adjust(0,0,1,1)
	fig1.canvas.draw()
	fig1.show()
	sorted_x = sorted(wDict.items(), key=lambda x: -x[1])
	print "%s%% of %s's were incorrectly identified,the most common guess for those failures was %s's" % \
	((wrong/500.0) * 100, j, sorted_x[0][0])

t0 = time.time()
dt = 0.0
while dt<30.:
	i = int(math.floor(len(fail)*np.random.rand()))
	j = fail[i]

	if dt == 0.0:
		img0.set_data(imgM[j])
		lab=fig0.suptitle('Guess: '+str(fDict[j]),fontsize=20,color='w')
	else:
		img0.set_data(imgM[j])
		lab.set_text('Guess: '+str(fDict[j]))

	fig0.canvas.draw()
	fig0.show()
	time.sleep(1.0)
	dt = time.time()-t0

plt.close(fig0)

print "\n"
print "Removing zero point offset:\n"
PT1 = imgmean.reshape(10,400)
PT = np.vstack((PT1, np.ones(400)))
P  = PT.transpose()
PTPinv = np.linalg.inv(np.dot(PT,P))
fDict = {}
fail = []
for j in range(10):

	wrong = 0
	wDict = {}

	l = []

	for i in xrange(500):
		index = j * 500 + i
		samp = imgM[index]
		PTyy = np.dot(PT,samp.flatten())
		avec = np.dot(PTPinv,PTyy)
		avec = avec[0:10]
		l.append(avec[0:10])

		if np.argmax(avec) != j:
			fDict[index] = np.argmax(avec)
			fail.append(index)
			#fail[index] = np.argmax(avec)
			wrong += 1
			if np.argmax(avec) in wDict:
				wDict[np.argmax(avec)] += 1
			else:
				wDict[np.argmax(avec)] = 1
	sorted_x = sorted(wDict.items(), key=lambda x:-x[1])


	print "%s%% of %s's were incorrectly identified, the most common guess for those failures was %s's" % \
	((wrong/500.0) * 100, j, sorted_x[0][0])

# plt.show()