#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import timedelta
import math


# In[27]:


from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# In[3]:


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


# In[23]:


def model1(data):
    try:
        model = ARIMA(data, order=(1,0,1))
    except:
        model = ARIMA(data, order=(2,0,1))
   
    fit = model.fit()
    result = fit.forecast(7)[0]

    return result
    


# In[42]:


def model2(data):
    data = data.astype(float)
    model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=7)
    result = model.fit().forecast(7)
    return result
    


# In[48]:


def model3(data):
    data = data.astype(float)
    model = ExponentialSmoothing(data, seasonal="add", seasonal_periods=7)
    result = model.fit().forecast(7)
    return result


# In[80]:


d_list = [df0, df1, df2, df3, df4, df5, df6, df7, df8, df9]


# In[82]:


def forecast(df):
    v = model2(df.visitor)
    o = model3(df.order)
    r = model1(df.rate)
    customer = df.customer[:7]
    device = df.device[:7]
    days = pd.date_range(max(df.date) + timedelta(days=1), periods=7)
    result = pd.DataFrame({'date':days, 
                           'visitor_forecast':v, 'order_forecaset':o, 'rate_forecast':r}).reset_index(drop=True)
    result['custoer'] = customer
    result['device'] = device
    return result


# In[83]:


result = forecast(d_list[0])
for d in d_list[1:]:
    result = pd.concat([result,forecast(d)])


# In[85]:


result = result.reset_index(drop=True)


# In[86]:


result.to_csv('forecast.csv',index=False)


# In[ ]:




