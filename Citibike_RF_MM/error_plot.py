
# coding: utf-8

# In[1]:

import pandas as pd
import matplotlib.pyplot as plt

get_ipython().magic(u'matplotlib inline')

from sklearn.metrics import *


# In[3]:

creeping = pd.DataFrame.from_csv('prediction.csv')
# creepingDate = creeping['2013-10-04']
# creepingDate = creepingDate.sort(columns='target').reset_index()
creeping


# In[20]:

print r2_score(creeping.target, creeping.predict)
print np.sqrt(mean_squared_error(creeping.target, creeping.predict))


# In[21]:

figure = plt.figure(0)
ax = plt.subplot(111)

ax.plot(creeping.index, creeping.predict, '.', c='dimgray', label='Predicted')
ax.plot(creeping.index, creeping.target, '-', c='tomato', lw=3, alpha=0.8, label='True Value')

ax.set_title('Random Forest prediction on Sep 2013', fontsize=14)
ax.set_xlabel('each hour of the last week of Sep 2013', fontsize=14)
ax.set_ylabel('predicted amount of bike use', fontsize=14)

ax.legend(loc='upper right')

ax.text(10,4300, 'R^2 = 0.78')
ax.text(50,4300, 'RMSE = 541')

figure.set_size_inches(8,6)
figure.savefig('RandomForest.png', dpi=150)


# In[79]:

# poisson = pd.DataFrame.from_csv('../Metrics/7-22_2220_extended_regression/Predictions.csv')
# poissonDate = poisson['2013-10-05']
# poissonDate = poissonDate.sort(columns='target').reset_index()


# In[80]:

# print r2_score(poissonDate.target, poissonDate.predicted)
# print mean_squared_error(poissonDate.target, poissonDate.predicted)


# In[81]:

# figure = plt.figure(1)
# ax = plt.subplot(111)

# ax.plot(poissonDate.index, poissonDate.predicted, '.', c='dimgray', label='Predicted')
# ax.plot(poissonDate.index, poissonDate.target, '-', c='tomato', lw=6, alpha=0.8, label='True Value')

# ax.set_title('Poisson 911 Crime Call Predictions for Oct. 5th, 2013', fontsize=14)
# ax.set_xlabel('Sector 8-hour bins (ordered by true value)', fontsize=14)
# ax.set_ylabel('Count of crime calls', fontsize=14)

# ax.legend(loc='upper left')

# ax.text(50,24, 'R^2 (coefficient of determination) = 0.161')
# ax.text(50,22.5, 'RMSE = 4.89')

# figure.set_size_inches(8,6)
# figure.savefig('/Users/alex/Desktop/poisson.png', dpi=150)


# In[120]:

# twentyfour = pd.DataFrame.from_csv('../Metrics/7-23_2011_creeping_regression_Random-Forest_24hrBins/Predictions.csv')
# twentyfourDate = twentyfour.loc['2013-10-05']
# twentyfourDate = twentyfourDate.sort(columns='target').reset_index()


# In[121]:

# print r2_score(twentyfourDate.target, twentyfourDate.predicted)
# print mean_squared_error(twentyfourDate.target, twentyfourDate.predicted)


# In[124]:

# figure = plt.figure(2)
# ax = plt.subplot(111)

# ax.plot(twentyfourDate.index, twentyfourDate.predicted, '.', c='dimgray', label='Predicted')
# ax.plot(twentyfourDate.index, twentyfourDate.target, '-', c='tomato', lw=6, alpha=0.8, label='True Value')

# ax.set_title('Random Forest 911 Crime Call Predictions for Oct. 5th, 2013', fontsize=14)
# ax.set_xlabel('Sector predictions (ordered by true value)', fontsize=14)
# ax.set_ylabel('Count of crime calls', fontsize=14)

# ax.legend(loc='upper left')

# ax.text(12,32, 'R^2 (coefficient of determination) = 0.694')
# ax.text(12,30, 'RMSE = 9.646')

# figure.set_size_inches(8,6)
# figure.savefig('/Users/alex/Desktop/24hrRF.png', dpi=150)


# In[39]:

twentyfourDate.ix[:,[1,3]] = twentyfourDate.ix[:,[1,3]].astype(int)


# In[40]:

twentyfourDate.dtypes


# In[87]:

redBaseline = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-23_2235_creeping_regression_Random-Forest_reducedFeatureswPLUTO/Predictions.csv')
red150 = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-24_0918_creeping_regression_Random-Forest_reducedFeatures150/Predictions.csv')
red200 = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-24_0919_creeping_regression_Random-Forest_reducedFeatures200/Predictions.csv')
red300 = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-24_0919_creeping_regression_Random-Forest_reducedFeatures300/Predictions.csv')


# In[88]:

redBaselineGrouped = redBaseline.groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))
red150Grouped = red150.groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))
red200Grouped = red200.groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))
red300Grouped = red300.groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))


# In[92]:

figure = plt.figure(3)
ax = plt.subplot(111)

ax.plot(redBaselineGrouped.index, redBaselineGrouped, lw=2, label='Baseline (100 trees)')
ax.plot(red150Grouped.index, red150Grouped, lw=2, label='150 trees')
ax.plot(red200Grouped.index, red200Grouped, lw=2, label='200 trees')
ax.plot(red300Grouped.index, red200Grouped, lw=2, label='300 trees')

loc = ax.xaxis.get_major_locator()
loc.maxticks[DAILY] = 8

ax.legend(loc="lower right", fontsize=11)

ax.set_title('Random Forest predictions from Sep. 17th, 2013 - Oct 11th, 2013')
ax.set_ylabel('R^2 (coefficient of determination)', fontsize=12)

figure.set_size_inches(8,6)
figure.savefig('/Users/alex/Desktop/more_trees.png', dpi=150)


# In[98]:

poisson = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-22_2220_extended_regression/Predictions.csv').groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))
extended = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-24_2213_extended_regression_Random-Forest_100treesExtended_ReducedFeatures/Predictions.csv').groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))


# In[99]:

poisson = poisson[:'2013-10-12 00:00:00']


# In[104]:

figure = plt.figure(4)
ax = plt.subplot(111)

ax.plot(extended.index, extended, lw=2, label='Extended predictions', c="#377eb8")
ax.plot(poisson.index, poisson, lw=2, label='Poisson', c="#4daf4a")
ax.plot(redBaselineGrouped.index, redBaselineGrouped, lw=2, label='Rolling predictions', c="#e41a1c")

loc = ax.xaxis.get_major_locator()
loc.maxticks[DAILY] = 8

ax.legend(loc="lower right", fontsize=11)

ax.set_title('Random Forest predictions from Sep. 17th, 2013 - Oct 11th, 2013')
ax.set_ylabel('R^2 (coefficient of determination)', fontsize=12)

figure.set_size_inches(8,6)
figure.savefig('/Users/alex/Desktop/model_variation.png', dpi=150)


# In[117]:

weather = pd.DataFrame.from_csv('/Users/alex/Desktop/metrics_7-24/7-24_0059_creeping_regression_Random-Forest_weatherReducedFeatures/Predictions.csv').groupby(level=0).apply(lambda x: r2_score(x.target, x.predicted))


# In[119]:

figure = plt.figure(5)
ax = plt.subplot(111)

ax.plot(weather.index, weather, lw=2, label='Rolling predictions w/ weather', c="#377eb8")
#ax.plot(poisson.index, poisson, lw=2, label='Poisson', c="#4daf4a")
ax.plot(redBaselineGrouped.index, redBaselineGrouped, lw=2, label='Rolling predictions', c="#e41a1c")

loc = ax.xaxis.get_major_locator()
loc.maxticks[DAILY] = 8

ax.legend(loc="lower right", fontsize=11)

ax.set_title('Random Forest predictions from Sep. 17th, 2013 - Oct 11th, 2013')
ax.set_ylabel('R^2 (coefficient of determination)', fontsize=12)

figure.set_size_inches(8,6)
figure.savefig('/Users/alex/Desktop/weather.png', dpi=150)

