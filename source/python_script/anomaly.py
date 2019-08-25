#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from datetime import datetime

import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from statsmodels.tsa.arima_model import ARIMA
#import pmdarima as pm

from pyculiarity import detect_anoms
from pyculiarity import detect_vec


# In[4]:


df = pd.read_csv('anomaly_det_dashboard_shopper_conv.csv')

df = df[['totalshoppertraffic_visitors','digital_orders','cust_prospect_ind','visit_device_type','event_dt']]

df.columns = ['visitor','order','customer','device','date']

df['rate'] = df.order/df.visitor

df.date = pd.to_datetime(df.date)
df = df.sort_values(by=['date']).reset_index(drop=True)

df = df[df.device != 'Gaming Consle']
df = df[df.device != 'E-Reader']

df0 = df[(df.customer == 'All Visitors') & (df.device == 'All Devices')].reset_index(drop=True)

df1 = df[(df.customer == 'CUSTOMER') & (df.device == 'Mobile Phone')].reset_index(drop=True)
df2 = df[(df.customer == 'CUSTOMER') & (df.device == 'Desktop')].reset_index(drop=True)
df3 = df[(df.customer == 'CUSTOMER') & (df.device == 'Tablet')].reset_index(drop=True)

df4 = df[(df.customer == 'UNDETERMINED') & (df.device == 'Mobile Phone')].reset_index(drop=True)
df5 = df[(df.customer == 'UNDETERMINED') & (df.device == 'Desktop')].reset_index(drop=True)
df6 = df[(df.customer == 'UNDETERMINED') & (df.device == 'Tablet')].reset_index(drop=True)

df7 = df[(df.customer == 'PROSPECT') & (df.device == 'Mobile Phone')].reset_index(drop=True)
df8 = df[(df.customer == 'PROSPECT') & (df.device == 'Desktop')].reset_index(drop=True)
df9 = df[(df.customer == 'PROSPECT') & (df.device == 'Tablet')].reset_index(drop=True)


# In[364]:


#!pip install pmdarima


# ### ARIMA

# In[ ]:





# In[ ]:





# In[620]:


def model1(data):
    try:
        model = ARIMA(data, order=(1,0,1))
        fit = model.fit()
    except:
        model = ARIMA(data, order=(2,0,1))
        fit = model.fit()
    fitted = fit.fittedvalues
    diff = abs(fitted - data)
    sd = data.std()
  
    result = np.where(diff > 1.5*sd)[0]

    return result
    


# ### twitter

# In[ ]:





# In[157]:


#!pip install pyculiarity


# In[454]:





# In[483]:


#  Anomaly Detection Using Seasonal Hybrid ESD Test
def model2(data):
    result = detect_vec(data, max_anoms=0.2, direction='both', period=14,alpha=0.5)
    return np.array(result['anoms'].index)


# In[740]:


def model3(data_df):
    copy = data_df.copy()
    result = detect_anoms.detect_anoms(copy, k=0.25, num_obs_per_period=14,one_tail=False, alpha=0.6)
    index = np.where(copy.timestamp.isin(result['anoms']))
    np.where(data.timestamp.isin(a['anoms']))
    return index[0]


# In[747]:


model2(df1.visitor)


# In[746]:


model3(df1[['date','visitor']])


# ### detect anomaly

# In[743]:


d_list = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9]


# In[748]:


def anomaly_detection(df):     
    v1 = np.zeros(len(df), dtype=int)
    v2 = np.zeros(len(df), dtype=int)
    v3 = np.zeros(len(df), dtype=int)
    index1 = model1(df.visitor)
    index2 = model2(df.visitor)
    index3 = model3(df[['date','visitor']])
    v1[index1] = 1
    v2[index2] = 1
    v3[index3] = 1    
    
    
    o1 = np.zeros(len(df), dtype=int)
    o2 = np.zeros(len(df), dtype=int)
    o3 = np.zeros(len(df), dtype=int)
    index1 = model1(df.order)
    index2 = model2(df.order)
    index3 = model3(df[['date','order']])
    o1[index1] = 1
    o2[index2] = 1
    o3[index3] = 1
    
    
    r1 = np.zeros(len(df), dtype=int)
    r2 = np.zeros(len(df), dtype=int)
    r3 = np.zeros(len(df), dtype=int)
    index1 = model1(df.rate)
    index2 = model2(df.rate)
    index3 = model3(df[['date','rate']])
    r1[index1] = 1
    r2[index2] = 1
    r3[index3] = 1
    
    result = df.copy()
    result['visitor_anomaly_arima'] = v1
    result['visitor_anomaly_stl'] = v2
    result['visitor_anomaly_twitter'] = v3
    result['order_anomaly_arima'] = o1
    result['order_anomaly_stl'] = o2
    result['order_anomaly_twitter'] = o3
    result['rate_anomaly_arima'] = r1
    result['rate_anomaly_stl'] = r2
    result['rate_anomaly_twitter'] = r3
    
    return result


# In[749]:


result = anomaly_detection(d_list[0])
for d in d_list[1:]:
    result = pd.concat([result,anomaly_detection(d)])


# In[751]:


result = result.reset_index(drop=True)


# In[752]:


result.to_csv('anomaly.csv',index=False)


# In[ ]:




