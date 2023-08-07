#!/usr/bin/env python
# coding: utf-8

# # AutoRegressive Integrated Moving Average - LONDON DATASET

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import folium
import imageio
import warnings 
warnings.filterwarnings("ignore")
from tqdm import tqdm_notebook
from folium.plugins import MarkerCluster
import imageio
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.stattools import acf

plt.style.use('ggplot')
plt.rcParams['font.family'] = 'sans-serif' 
plt.rcParams['font.serif'] = 'Ubuntu' 
plt.rcParams['font.monospace'] = 'Ubuntu Mono' 
plt.rcParams['font.size'] = 14 
plt.rcParams['axes.labelsize'] = 12 
plt.rcParams['axes.labelweight'] = 'bold' 
plt.rcParams['axes.titlesize'] = 12 
plt.rcParams['xtick.labelsize'] = 12 
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['figure.titlesize'] = 12 
plt.rcParams['image.cmap'] = 'jet' 
plt.rcParams['image.interpolation'] = 'none' 
plt.rcParams['figure.figsize'] = (12, 10) 
plt.rcParams['axes.grid']=True
plt.rcParams['lines.linewidth'] = 2 
plt.rcParams['lines.markersize'] = 8
colors = ['xkcd:pale orange', 'xkcd:sea blue', 'xkcd:pale red', 'xkcd:sage green', 'xkcd:terra cotta', 'xkcd:dull purple', 'xkcd:teal', 'xkcd: goldenrod', 'xkcd:cadet blue',
'xkcd:scarlet']


# In[2]:


data = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv')


# # Data Visualization

# In[3]:


city_data = data.drop_duplicates(['City'])


# In[5]:


city_data.head()


# In[6]:


explodes = (0,0.3)
plt.pie(data[data['City']=='London'].AverageTemperature.isna().value_counts(),explode=explodes,startangle=0,colors=['firebrick','indianred'],
   labels=['Non NaN elements','NaN elements'], textprops={'fontsize': 20})


# In[22]:


london_data = data[data['City']=='London']


# In[8]:


london_data['AverageTemperature']=london_data.AverageTemperature.fillna(method='bfill')


# In[9]:


london_data['AverageTemperatureUncertainty']=london_data.AverageTemperatureUncertainty.fillna(method='bfill')


# In[10]:


london_data = london_data.reset_index()


# In[11]:


london_data = london_data.drop(columns=['index'])


# In[12]:


london_data.dt = pd.to_datetime(london_data.dt)


# In[13]:


YEAR = []
MONTH = []
DAY = []
WEEKDAY = []
for i in range(len(london_data)):
    WEEKDAY.append(london_data.dt[i].weekday())
    DAY.append(london_data.dt[i].day)
    MONTH.append(london_data.dt[i].month)
    YEAR.append(london_data.dt[i].year)


# In[14]:


london_data['Year'] = YEAR
london_data['Month'] = MONTH
london_data['Day'] = DAY 
london_data['Weekday'] = WEEKDAY


# In[15]:


change_year_index = []
change_year = []
year_list = london_data['Year'].tolist()
for y in range(0,len(year_list)-1):
    if year_list[y]!=year_list[y+1]:
        change_year.append(year_list[y+1])
        change_year_index.append(y+1)


# In[16]:


london_data.loc[change_year_index].head()


# In[29]:


last_year_data = london_data[london_data.Year>=2010].reset_index().drop(columns=['index'])
P = np.linspace(0,len(last_year_data)-1,5).astype(int)


# In[30]:


def get_timeseries(start_year,end_year):
    last_year_data = london_data[(london_data.Year>=start_year) & (london_data.Year<=end_year)].reset_index().drop(columns=['index'])
    return last_year_data


# In[31]:


def plot_timeseries(start_year,end_year):
    last_year_data = get_timeseries(start_year,end_year)
    P = np.linspace(0,len(last_year_data)-1,5).astype(int)
    plt.plot(last_year_data.AverageTemperature,marker='.',color='firebrick')
    plt.xticks(np.arange(0,len(last_year_data),1)[P],last_year_data.dt.loc[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')


# In[32]:


def plot_from_data(data,time,c='firebrick',with_ticks=True,label=None):
    time = time.tolist()
    data = np.array(data.tolist())
    P = np.linspace(0,len(data)-1,5).astype(int)
    time = np.array(time)
    if label==None:
        plt.plot(data,marker='.',color=c)
    else:
        plt.plot(data,marker='.',color=c,label=label)
    if with_ticks==True:
        plt.xticks(np.arange(0,len(data),1)[P],time[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')


# In[33]:


plt.figure(figsize=(20,20))
plt.suptitle('Plotting 4 random decades of dataset to clearly understand data trends',fontsize=30,color='firebrick')

plt.subplot(2,2,1)
plt.title('Starting year: 1800, Ending Year: 1810',fontsize=15)
plot_timeseries(1800,1810)
plt.subplot(2,2,2)
plt.title('Starting year: 1900, Ending Year: 1910',fontsize=15)
plot_timeseries(1900,1910)
plt.subplot(2,2,3)
plt.title('Starting year: 1950, Ending Year: 1960',fontsize=15)
plot_timeseries(1900,1910)
plt.subplot(2,2,4)
plt.title('Starting year: 2000, Ending Year: 2010',fontsize=15)
plot_timeseries(1900,1910)
plt.tight_layout()


# # Checking on Stationarity
# For ARIMA models, we should be considering stationary time series. In order to check if the timeseries we are considering is stationary, we can check the correlation and autocorrelation plots

# In[34]:


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(london_data.AverageTemperature, ax=ax1,color ='firebrick')
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(london_data.AverageTemperature, ax=ax2,color='firebrick')


# # 
# It is suggesting us that the timeseries is not stationary.

# In[35]:


result = adfuller(london_data.AverageTemperature)
print('ADF Statistic on the entire dataset: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


# #
# The AD Fuller Test on the entire dataset tells us that the dataset is stationary.

# In[36]:


result = adfuller(london_data.AverageTemperature[0:120])
print('ADF Statistic on the first decade: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


# #
# For a single decade, it is clear that the dataset is absolutely not stationary for the decade period of time.

# In[37]:


plt.title('The dataset used for prediction: 1992-2013', fontsize=30,color='firebrick')
plot_timeseries(1992,2013)


# # 
# In order to take account of this non-stationarity, a differentiation term will be considered in the ARIMA models.

# # 
# Performing the train/test split:

# In[38]:


temp = get_timeseries(1992,2013)
N = len(temp.AverageTemperature)
split = 0.95
training_size = round(split*N)
test_size = round((1-split)*N)
series = temp.AverageTemperature[:training_size]
date = temp.dt[:training_size]
test_series = temp.AverageTemperature[len(date)-1:len(temp)]
test_date = temp.dt[len(date)-1:len(temp)]
#test_date = test_date.reset_index().dt
#test_series = test_series.reset_index().AverageTemperature


# In[39]:


test_date


# In[40]:


plt.title('Plotting the split', fontsize=30,color='firebrick')
plot_from_data(series,date,label='Training Set')
plot_from_data(test_series,test_date,'navy',with_ticks=False,label='Test Set')
plt.legend()


# # 
# Loads a time series dataset between 1992 and 2013.
# Splits the dataset into a training set (95% of the data) and a test set (5% of the data).
# Extracts the 'AverageTemperature' and 'dt' (dates) columns for the training and test sets separately, along with their corresponding date values.

# #
# ARIMA models are based on an optimization procedure that adopts the Maximum Likelihood function.

# In[41]:


def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm_notebook(order_list):
        #try: 
        model = SARIMAX(exog, order=order).fit(disp=-1)
    #except:
    #        continue
            
        aic = model.aic
        results.append([order, model.aic])
    #print(results)
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# #
# The zero-differentiated ARIMA models are considered and evaluated using the AIC.

# In[42]:


ps = range(0, 10, 1)
d = 0
qs = range(0, 10, 1)

# Create a list with all possible combination of parameters
parameters = product(ps, qs)
parameters_list = list(parameters)

order_list = []

for each in parameters_list:
    each = list(each)
    each.insert(1, d)
    each = tuple(each)
    order_list.append(each)
    
result_d_0 = optimize_ARIMA(order_list, exog = series)


# In[43]:


result_d_0.head()


# In[44]:


import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
ps = range(0, 10, 1)
d = 1
qs = range(0, 10, 1)

# Create a list with all possible combination of parameters
parameters = product(ps, qs)
parameters_list = list(parameters)

order_list = []

for each in parameters_list:
    each = list(each)
    each.insert(1, d)
    each = tuple(each)
    order_list.append(each)
    
result_d_1 = optimize_ARIMA(order_list, exog = series)

result_d_1


# In[45]:


result_d_1.head()


# In[46]:


final_result = result_d_0.append(result_d_1)


# In[47]:


best_models = final_result.sort_values(by='AIC', ascending=True).reset_index(drop=True).head()


# In[48]:


best_models.head()


# #
# The total summary is highlighted with this function and it shows that the (2,1,5) model and the (2,1,6) model are the best ones.

# In[49]:


best_model_params_0 = best_models[best_models.columns[0]][0]
best_model_params_1 = best_models[best_models.columns[0]][1]


# In[50]:


best_model_0 = SARIMAX(series, order=best_model_params_0).fit()
print(best_model_0.summary())
best_model_1 = SARIMAX(series, order=best_model_params_1).fit()
print(best_model_1.summary())


# # Model (2,1,5) results:

# In[51]:


best_model_0.plot_diagnostics(figsize=(15,12))
plt.show()


# # Model (2,1,6) results:

# In[52]:


best_model_1.plot_diagnostics(figsize=(15,12))
plt.show()


# #
# It is preferable to use low index models both to avoid overfitting and reduce the computational stress. For this reason, the (2, 1, 5) model has been considered

# ## Forecasting

# In[53]:


fore_l= test_size-1
forecast = best_model_0.get_prediction(start=training_size, end=training_size+fore_l)
forec = forecast.predicted_mean
ci = forecast.conf_int(alpha=0.05)


# In[54]:


error_test=london_data.loc[test_date[1:].index.tolist()].AverageTemperatureUncertainty
index_test = test_date[1:].index.tolist()
test_set = test_series[1:]


# In[55]:


lower_test = test_set-error_test
upper_test = test_set+error_test


# In[56]:


fig, ax = plt.subplots(figsize=(16,8), dpi=300)
x0 = london_data.AverageTemperature.index[0:training_size]
x1=london_data.AverageTemperature.index[training_size:training_size+fore_l+1]#ax.fill_between(forec, ci['lower Load'], ci['upper Load'])
plt.plot(x0, london_data.AverageTemperature[0:training_size],'k', label = 'Average Temperature')

plt.plot(london_data.AverageTemperature[training_size:training_size+fore_l], '.k', label = 'Actual')

forec = pd.DataFrame(forec, columns=['f'], index = x1)
forec.f.plot(ax=ax,color = 'Darkorange',label = 'Forecast (d = 2)')
ax.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence inerval (95%)',color='grey')

forec = pd.DataFrame( columns=['f'], index = x1)
# forec.f.plot(ax=ax,color = 'firebrick',label = 'Forecast  (2,1,5) model')
ax.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence inerval (95%)',color='grey')


plt.legend(loc = 'upper left')
plt.xlim(120,265)
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
plt.show()


# In[60]:


#plt.plot(forec)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plt.plot(test_set,marker='.',label="Actual",color='navy')
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
#plt.fill_between(x1, s_ci['lower AverageTemperature'], s_ci['upper AverageTemperature'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
plt.legend()
plt.subplot(2,1,2)
#plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plt.plot(test_set,marker='.',label="Actual",color='navy')
plt.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
plt.legend()
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')


# In[61]:


plt.fill_between(np.arange(0,len(test_set),1), lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plot_from_data(test_set,test_date,c='navy',label='Actual')
plt.legend(loc=2)

