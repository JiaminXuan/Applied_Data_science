
# coding: utf-8

# In[2]:

from sklearn import cross_validation as cv
from datetime import datetime as dt
from pandas import DataFrame as df
import pandas as pd
from sklearn import cluster as c


# In[4]:

data=df.from_csv('my_csv.csv',index_col=None)


# In[18]:

data = data[data.gender != 0]


# In[67]:

data['starttime']=map(lambda x:dt.strptime(x,'%Y-%m-%d %H:%M:%S'),data['starttime'])
data['stoptime']=map(lambda x:dt.strptime(x,'%Y-%m-%d %H:%M:%S'),data['stoptime'])


# In[27]:

station=pd.read_json('/Users/MSAUSI2014/Desktop/station.json')


# In[29]:

station=station[['id','latitude','longitude','stationName']]


# In[44]:

from sklearn import cluster as c


# In[54]:

Kmean_station=c.KMeans(n_clusters=8)
station['cluster']=Kmean_station.fit_predict(station[['latitude','longitude']])
station.to_csv('cluster.csv')


# In[73]:

train=data[['tripduration','start station id','end station id','bikeid','birth year']]
target=data['gender']


# In[58]:

newdata=pd.merge(data,station,how='left',left_on='start station id',right_on='id')
newdata=pd.merge(data,station,how='left',left_on='end station id',right_on='id')


# In[70]:

data['start station cluster']=map(lambda x: station[station['id']==x]['cluster'],data['start station id'])
data['end station cluster']=map(lambda x: station[station['id']==x]['cluster'],data['end station id'])


# In[62]:

station[station['id']==147]['cluster']


# In[75]:

from sklearn.ensemble import RandomForestClassifier as rf
rf1=rf(n_estimators=10,n_jobs=4)


# In[76]:

rf1.fit(train,target)


# In[77]:

rf1


# In[79]:




# In[80]:

data=data[['tripduration','start station id','end station id','bikeid','birth year','gender']]


# In[85]:

data.index=[i for i in range(len(data))]


# In[87]:

data.to_csv('/Users/MSAUSI2014/Desktop/stream_test/traindata.csv')


# In[84]:




# In[ ]:



