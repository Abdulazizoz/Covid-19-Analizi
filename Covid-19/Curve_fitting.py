import pylab as plb
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
df=pd.read_excel("Türkiye-time_series_covid19_confirmed_global-20082020.xlsx",index=["Country/Region","Date","Cases","Deaths","Recovered"])
print(df.head())
dataCases=df.Cases.values
dataDeaths = df.Deaths.values
dataRecovered = df.Recovered.values
datas=[dataCases,dataDeaths,dataRecovered]
column=["Cases","Deaths","Recovered"]
p0=[0,1,1,0]
def deneme(dataRecovered,column):
    dataDate=df.Date.values
    x=range(len(dataDate))
    y=dataRecovered

    n = len(x)
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
    def gaus(x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    # def logistic_function(x, a, x0, sigma):
    #     import numpy as np
    #     return sigma / (1 + np.exp(-(x - x0) / a))
    # def logistic_function(x, a, b, c, d):
    #     return a / (1. + np.exp(-c * (x - d))) + b
    # popt, pcov = curve_fit(logistic_function, x, y, p0=p0, bounds=(0, [500000., 10., 1000., 1000., ]))
    popt,pcov = curve_fit(gaus,x,y,p0=[1,mean,sigma])

    plt.plot(x,y,'b:',label='data')
    x=range(len(dataDate))
    plt.plot(x,logistic_function(x,*popt),'r:',label='fit')
    plt.legend()
    plt.title(''+column+' - Fit for Time Constant')
    plt.xlabel('Time (s)')
    plt.ylabel('')
    plt.show()
for i in range(0,3):
    deneme(datas[i],column[i])