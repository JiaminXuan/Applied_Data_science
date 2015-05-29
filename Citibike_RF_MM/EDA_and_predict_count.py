
# coding: utf-8

# In[78]:

from pandas import DataFrame as df
from datetime import datetime as dt


# In[79]:

def functionreplace(x):
    try:
        return float(x)
    except:
        return float(0)


# In[80]:

mylist=[]
with open('weather.csv') as f:
    for row in f:
        row=row.strip()
        row=row.split('\t')
        if row[0][:7]=='2013-09':
            mylist.append(row[:-2])
for row in mylist:
    row[0]=row[0][:13]
weatherdf = df.from_dict(mylist)
weatherdf.columns=('date', 'temperature', 'skycover','humidity','precipitation','solar_radiation')
weatherdf.drop_duplicates(subset='date',inplace=True)


# In[81]:

weatherdf['temperature']=map(lambda x:functionreplace(x),weatherdf['temperature'])
weatherdf['skycover']=map(lambda x:functionreplace(x),weatherdf['skycover'])
weatherdf['humidity']=map(lambda x:functionreplace(x),weatherdf['humidity'])
weatherdf['precipitation']=map(lambda x:functionreplace(x),weatherdf['precipitation'])
weatherdf['solar_radiation']=map(lambda x:functionreplace(x),weatherdf['solar_radiation'])
weatherdf['hour']=map(lambda x:float(x[-2:]),weatherdf['date'])
weatherdf['day']=map(lambda x:float(x[8:10]),weatherdf['date'])


# In[82]:

data=weatherdf.ix[:,1:].values


# In[88]:

mydict={}
with open('citibike-Sep.txt') as f:
    for row in f:
        row=row.strip()
        row=row.split(',')
        hour=row[1][:13]
        try:
            mydict[hour]+=1
        except:
            mydict[hour]=1
countdf=df.from_dict(mydict,orient='index')
countdf['count']=countdf[0]
countdf=countdf['count']
countdf=countdf.sort_index().values


# In[89]:

target=countdf


# In[90]:

from sklearn.ensemble import RandomForestRegressor as rf
model=rf(n_estimators=100,n_jobs=-1)


# In[91]:

model.fit(X=data[:552,:],y=target[:552])


# In[92]:

predicts=model.predict(data[552:])


# In[93]:

result =df({ 'target' :target[552:],'predict' : predicts})


# In[77]:

result.to_csv('prediction.csv')


# In[94]:

weatherdf


# In[209]:

citibike['starttime']=map(lambda x:dt.strptime(x,'%Y-%m-%d %H:%M:%S'),citibike['starttime'])
citibike['stoptime']=map(lambda x:dt.strptime(x,'%Y-%m-%d %H:%M:%S'),citibike['stoptime'])
citibike['dow']=map(lambda x:x.weekday(),citibike['starttime'])  
citibike['month']=map(lambda x:x.month,citibike['starttime'])  
citibike['day']=map(lambda x:x.day,citibike['starttime'])


# In[ ]:

citibike['starttime']=map(lambda x:dt.strptime(x,'%Y-%m-%d %H:%M:%S'),citibike['starttime'])
citibike['hour']=map(lambda x:x.hour,citibike['starttime'])
dowcount=citibike.groupby('hour').size()
dowcount.plot(kind='bar')


# In[23]:

citibike['month']=map(lambda x:x.month,citibike['starttime'])  
citibike['day']=map(lambda x:x.day,citibike['starttime'])


# In[25]:

dowcount=citibike.groupby('dow').size()
dowcount.plot(kind='bar')


# In[26]:

monthcount=citibike.groupby('month').size()
monthcount.plot(kind='bar')


# In[27]:

daycount=citibike.groupby('day').size()
daycount.plot(kind='bar')


# In[39]:

gendercount=citibike.groupby('gender').size()
gendercount[1:3].plot(kind='bar')


# In[36]:




# In[45]:

citibike['duration']=map(lambda x:True if x<1800 else False,citibike['tripduration'])


# In[47]:

citibike['duration'].plot(kind='hist')


# In[50]:

citibike[citibike['duration']==True]['tripduration'].plot().savefig('try.png')


# In[ ]:



