
from numpy import random as r
from matplotlib import pyplot as plt
import statsmodels.api as sm


def get_z (p):
	z=[]
	z.append(0)
	for i in xrange(1000):
		z.append(float(p*z[i]+r.normal(0,1,1)))
	return z[1:]

def get_stats(p):
	z=get_z(p)
	y=z[1:]
	x=z[:-1]
	x = sm.add_constant(x)
	results = sm.OLS(y, x).fit()
	print results.summary()
	print 
	print
	print

def get_plot(p):
	z=get_z(p)
	plt.plot(range(1000),z,'grey')
	plt.plot([0,1000],[0,0],'r--')
	plt.show()

def single_walk():
	w1=get_z(1)
	w2=get_z(1)
	w2= sm.add_constant(w2)
	results = sm.OLS(w1,w2).fit()
	return results




def problem1():
	########problem1
	pvalue=[0.5,-0.5,1]
	for p in pvalue:
		print 'the statistic summary for p=%s'%(p)
		get_stats(p)
		get_plot(p)




def problem2():
	
	print 'answer for question a'
	results=single_walk()
	print results.summary()
	print
	print
	print
	


	print 'answer for question b'
	coef=[]
	for i in xrange(1000):
		results=single_walk()
		coef.append(results.params[1])
	plt.hist(coef,bins=20,alpha=0.5,linewidth=0,color='r')
	plt.show()

if __name__ == '__main__':
	problem1()
	problem2()