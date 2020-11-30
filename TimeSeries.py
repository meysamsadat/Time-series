from datetime import datetime
import itertools
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')
matplotlib.rcParams['axes.labelsize']=14
matplotlib.rcParams['xtick.labelsize'] =12
matplotlib.rcParams['ytick.labelsize'] = 12
matplotlib.rcParams['text.color'] = 'k'

df = pd.read_excel(r'C:\Users\meysam-sadat\PycharmProjects\sample.xls')
df.info(verbose=True)
df_object_des = df.describe(include='object').T
df_des = df.describe()
df = df.sort_values('Order Date',ascending=True)

df['lead_time'] = df['Ship Date'] - df['Order Date']

furniture = df[df['Category']=='Furniture']
furniture.columns
furniture = furniture[['Order Date','Sales']]
furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
furniture = furniture.set_index('Order Date')
furniture.index.max()
y = furniture['Sales'].resample('MS').mean()#resampling from days to month
y['2017':]
y.plot(figsize=(15,6))
from pylab import rcParams
rcParams['figure.figsize'] = 16,5
decomposition = sm.tsa.seasonal_decompose(y)
fig = decomposition.plot()

x = furniture['Sales'].resample('Q').sum()
rcParams['figure.figsize'] = 16,5
decomposition = sm.tsa.seasonal_decompose(x)
fig = decomposition.plot()
x.plot(kind='area')
x[0]
#Time Series Analysis
p=d=q = range(0,2)
pdq = list(itertools.product(p,d,q))
pdq_seasonal = [(x[0],x[1],x[2],12) for x in pdq]
print(f'SRIMAX: {pdq[1]} x {pdq_seasonal[1]}')
print(f'SRIMAX: {pdq[2]} x {pdq_seasonal[2]}')
print(f'SRIMAX: {pdq[3]} x {pdq_seasonal[3]}')
print(f'SRIMAX: {pdq[4]} x {pdq_seasonal[4]}')

for param in pdq:
    for param_seasonal in pdq_seasonal:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
            result = mod.fit()
            print(f'ARIMA {param} x {param_seasonal} x12-AIC:{result.aic} ')
        except:
            continue
mod = sm.tsa.statespace.SARIMAX(y,order=(1,1,1),seasonal_order=(1,1,1,12),enforce_stationarity=False,enforce_invertibility=False)
mod.fit()
print(result.summary().tables[1])

pred = result.get_prediction(start=pd.to_datetime('2017-01-01'),end=pd.to_datetime('2018-01-30'))
pred_ci = pred.conf_int()
ax = y['2014':].plot(label='Seasonal')
pred.predicted_mean.plot(ax=ax,label='One-step ahead Forcats',alpha=0.7)
ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='k',alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()

y_forcasted = pred.predicted_mean
y_truth = y['2017-01-01':]
#compute the mean square error
mse = ((y_forcasted-y_truth)**2).mean()
print(f'Mean Squared error of forcast is {round(mse,2)} ')
print(f'Root Mean squared error of forcast is {(round(np.sqrt(mse),2))}')


pred_uc = result.get_forecast(steps=20)
pred_ci = pred.conf_int()
ax = y.plot(label='Seasonal')
pred_uc.predicted_mean.plot(ax=ax,label='Forcats',alpha=0.7)
ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='k',alpha=0.2)
ax.set_xlabel('Date')
ax.set_ylabel('Furniture Sales')
plt.legend()












time = datetime.now()
end = time.strftime('%d-%B-%Y')
start = time
end = datetime.strptime(end,'%d-%B-%Y')

import pandas as pd
a = pd.date_range('20 January 2020',periods=10,freq='MS',tz='Asia/Tehran')
a_df = pd.DataFrame(a,columns=['time'])
