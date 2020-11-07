#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import scipy.stats as stats
import os
import random

import statsmodels.api as sm
import statsmodels.stats.multicomp

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


# In[2]:


population=pd.read_csv(r'F:\DATA ANALYTICS\COVID\population.csv')
test_per_state=pd.read_csv(r'F:\DATA ANALYTICS\COVID\statetest.csv')


population=population.rename(columns={'State / Union Territory':'State'})
population.head() 


# In[3]:


test_per_state.head(80)


# In[4]:


test_per_state['Positive'].sort_values().head()


# In[5]:


test_per_state['State'][test_per_state['Positive']==1].unique()


# ### Missing value imputation(Using median of each state)

# In[6]:


median=test_per_state.groupby('State')[['Positive']].median().reset_index().rename(columns={'Positive':'Median'})
median.head()


# In[7]:


for index,row in test_per_state.iterrows():

    if pd.isnull(row['Positive']):

        test_per_state['Positive'][index]=int(median['Median'][median['State']==row['State']])
test_per_state.head(30)  


# In[8]:


test_per_state['Positive'].sum()


# In[9]:


data=pd.merge(test_per_state,population,on='State')
data.head()


# In[10]:


def densityCheck(data):
    data['density_Group']=0
    for index,row in data.iterrows():
        status=None
        i=row['Density'].split('/')[0]
        
        
        try:   
            if (',' in i):
                i=int(i.split(',')[0]+i.split(',')[1])
            elif ('.' in i):
                i=round(float(i))
            else:
                i=int(i)

        except ValueError as err:
            pass
        
        try:
  
            if (0<i<=300):
                status='Dense1'
            elif (300<i<=600):
                status='Dense2'
            elif (600<i<=900):
                status='Dense3'
            else:
                status='Dense4'
        except ValueError as err:
            pass      
                
        data['density_Group'].iloc[index]=status     
    return(data)
        
        
        


# In[11]:


data=densityCheck(data)
data.head()


# In[12]:


data.describe()


# In[13]:


df=pd.DataFrame({'Dense1':data[data['density_Group']=='Dense1']['Positive'],
                 'Dense2':data[data['density_Group']=='Dense2']['Positive'],
                 'Dense3':data[data['density_Group']=='Dense3']['Positive'],
                 'Dense4':data[data['density_Group']=='Dense4']['Positive']})
               


# In[14]:


df.head()


# In[15]:


np.random.seed(14)
dataNew=pd.DataFrame({'Dense1':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10),
                      'Dense2':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10),
                      'Dense3':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10),
                      'Dense4':random.sample(list(data['Positive'][data['density_Group']=='Dense1']), 10)})


# In[16]:


dataNew.describe()


# In[17]:



fig = plt.figure(figsize=(10,10))
title = fig.suptitle("Corona cases across different density groups", fontsize=14)

ax1 = fig.add_subplot(2,2,1)
sns.distplot(dataNew['Dense1'])

ax1 = fig.add_subplot(2,2,2)
sns.distplot(dataNew['Dense2'])

ax1 = fig.add_subplot(2,2,3)
sns.distplot(dataNew['Dense3'])

ax1 = fig.add_subplot(2,2,4)
sns.distplot(dataNew['Dense4'])


# In[18]:


dataNew['Dense1'],fitted_lambda = stats.boxcox(dataNew['Dense1'])
dataNew['Dense2'],fitted_lambda = stats.boxcox(dataNew['Dense2'])
dataNew['Dense3'],fitted_lambda = stats.boxcox(dataNew['Dense3'])
dataNew['Dense4'],fitted_lambda = stats.boxcox(dataNew['Dense4'])


# In[ ]:





# In[19]:


fig = plt.figure(figsize=(10,10))
title = fig.suptitle("Corona cases across different density groups", fontsize=14)

ax1 = fig.add_subplot(2,2,1)
sns.distplot(dataNew['Dense1'])

ax1 = fig.add_subplot(2,2,2)
sns.distplot(dataNew['Dense2'])

ax1 = fig.add_subplot(2,2,3)
sns.distplot(dataNew['Dense3'])

ax1 = fig.add_subplot(2,2,4)
sns.distplot(dataNew['Dense4'])


# In[20]:


f_stati,p_value=stats.f_oneway(dataNew['Dense1'],dataNew['Dense2'],dataNew['Dense3'],dataNew['Dense4'])


# In[21]:


print("The F statistic value is %.3f and the p_value is %.3f" %(f_stati,p_value))


# ### Since the p_value is <0.05 we can reject the null hypothesis and accept the alternative hypothesis

# ### Using StatsModel

# In[22]:


dataNew.head()


# In[23]:


dataNew_1=dataNew.stack().to_frame().reset_index().rename(columns={'level_1':'Density_group',0:'Count'})
dataNew_1.head()


# In[25]:


model=ols('Count~C(Density_group)',dataNew_1).fit()

model.summary()


# ### POST-HOC Test Tukey HSD

# In[32]:


comp=sm.stats.multicomp.MultiComparison(dataNew_1['Count'],dataNew_1['Density_group'])
Post_Hoc=comp.tukeyhsd()
print(Post_Hoc)


# In[ ]:




