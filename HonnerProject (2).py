#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import datetime as dt
import quantstats as qs
import backtrader as bt


# In[2]:


Multiple = 8


# ## Normalize Data, Creating DF based on stock data from 2012 to 2020

# In[871]:


year = 2012
ticker = ("SPY")
df = pd.DataFrame()
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(Multiple):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(129.74).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i}"] = normBasket


# In[872]:


year = 2012
ticker = ("NVDA")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(3.17).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple}"] = normBasket


# In[873]:


year = 2012
ticker = ("AMAT")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(10.46).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple}"] = normBasket


# In[874]:


year = 2012
ticker = ("CAT")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(88.389999).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple}"] = normBasket


# In[875]:


year = 2012
ticker = ("TMO")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(50.540001).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[876]:


year = 2012
ticker = ("JNJ")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(63.349998).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[877]:


year = 2012
ticker = ("PFE")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(21.413662).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[878]:


year = 2012
ticker = ("AZN")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(20.730000).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[879]:


year = 2012
ticker = ("HUM")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(74.529999).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[880]:


year = 2012
ticker = ("UNH")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(53.990002).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[881]:


year = 2012
ticker = ("MKC")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(28.215000).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[882]:


year = 2012
ticker = ("NKE")
# "NVDA", "AMZN", "META", "AMD", "MSFT", "AAPL", "TSLA"
for i in range(8):
    Basket = yf.download(ticker, f"{year}-05-18",f"{year+1}-05-18")
    Basket = Basket.Close.copy().dropna()
    normBasket = Basket
    normBasket.to_frame()
    normBasket=normBasket.div(26.360001	).mul(100)
    print(normBasket)
    year += 1 
    normBasket = normBasket.reset_index(level='Date', drop=True)
    df[f"{i+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple+Multiple}"] = normBasket


# In[883]:


Stocks = 12
Sectors = 5
Years = 8


# ## Here, we begin arranging the data into an array to perform SVD

# In[952]:


array = np.array(df)


# In[953]:


#Perform SVD
u, s, vh = np.linalg.svd(array, full_matrices=True,compute_uv=True)


# ## Convert back into a DF, working with different basis vectors. mframe represents the original outerproduct of a given Basis Vector

# In[954]:


j = 5
#BV 6. Interesting results.


# In[887]:


testmatrix = s[j]*np.outer(u[:,j],vh[j])

#Interestingly enough, basis vectors 3, 6, 9, 11 show negative correlation.


# In[888]:


mframe = pd.DataFrame(testmatrix)


# ---------------------------------------------------------------------------------------------------------------------

# ## 0-1(Tech), 2-3(PMAT), 4-7(HTech), 8-9(HServ),10-11(CNonD)

# ---------------------------------------------------------------------------------------------------------------------

# ## Creating Cframe, the sector averages within the outer product

# In[825]:


Stocks = ["SPY", "NVDA", "AMAT", "CAT", "TMO", "UNH", "JNJ", "PFE", "AZN", "HUM", "MKC","NKE"]
#TECH (SPY, NVDA), Producer Manufacturing(CAT, AMAT), Health Tech (JNJ, PFE, AZN, TMO) Health Services (UNH, HUM)
#Consumer nondurables, "MKC", "NKE"


# In[826]:


Cframe = pd.DataFrame()
Cframe["Tech"] = mframe.iloc[:,0]/mframe.iloc[:,0] - 1
Cframe["PMAT"] = mframe.iloc[:,0]/mframe.iloc[:,0] - 1
Cframe["HTech"] = mframe.iloc[:,0]/mframe.iloc[:,0] - 1
Cframe["HServ"] = mframe.iloc[:,0]/mframe.iloc[:,0] - 1
Cframe["CNonD"] = mframe.iloc[:,0]/mframe.iloc[:,0] - 1


# In[829]:


for i in range(16):
    Cframe["Tech"] += mframe.iloc[:,i]
Cframe["Tech"] = Cframe["Tech"]/16
for i in range(16):
    Cframe["PMAT"] += mframe.iloc[:,i+16]
Cframe["PMAT"] = Cframe["PMAT"]/16
for i in range(32):
    Cframe["HTech"] += mframe.iloc[:,i+32]
Cframe["HTech"] = Cframe["HTech"]/32
for i in range(16):
    Cframe["HServ"] += mframe.iloc[:,i+64]
Cframe["HServ"] = Cframe["HServ"]/16
for i in range(16):
    Cframe["CNonD"] += mframe.iloc[:,i+80]
Cframe["CNonD"] = Cframe["CNonD"]/16


# ## Cframe explained: We have to take the average value of the vectors in a sector based on the basis vector. 
# ## Then we find sectors with opposing correlation from the basis vector.

# In[908]:


import seaborn as sns
plt.figure(figsize=(12,8))
sns.set(font_scale=1.4)
sns.heatmap(Cframe.corr(method="pearson"),cmap="Reds", annot=True, annot_kws={"size":15},vmax=0.6)
plt.show()


# 
# ## The Correlation matrix is determined by 1's and -1's because the percentage change is the same for all stocks, 
# just in different directions. (That's what the basis vectors do, they scale the same percentage change.) 
# From this, we understand that HTech, HServ are negatively correlated.)


# In[3]:


#Let's create a function that takes active price data, scales our sixth basis vector as close as possible, then compares the gradient.

def ScaleBasis(Sector1,Sector2):
    # data is a dataframe, 250 long set of the last prices of two negatively correlated stocks
    # u[:,j] scaled by what equals the sector vaue?
    # I need to innovate a solution that allows me to access a dataframe of a stock's close values 
    # in the form of a vector, then scale u[:,j] to that vector as close as possible.
    # Afterwords, the norm of the two vectors is subtracted. If it's above a certain level, we sell.
    
    try:
        BasisVector1 = np.dot(u[:,j],Sector1)/np.linalg.norm(u[:,j])
        BasisVector2 = np.dot(u[:,j],Sector2)/np.linalg.norm(u[:,j])
        BasisDiff = np.linalg.norm(BasisVector1) - np.linalg.norm(BasisVector2)
        return BasisDiff
    #y * np.dot(x, y) / np.dot(y, y)
    except:
        print("DataSet is not ready yet")


# In[17]:


# This just exists to feed our backtesting library readable data

def openbb_data_to_bt_data(symbol, start_date, end_date):
    
    df = yf.download(symbol, start_date, end_date)
    
    fn = f"{symbol.lower()}.csv"
    df.to_csv(fn)
    
    return bt.feeds.YahooFinanceCSVData(
        dataname=fn,
        fromdate=dt.datetime.strptime(start_date, '%Y-%m-%d'),
        todate=dt.datetime.strptime(end_date, '%Y-%m-%d')
    )


# In[271]:


# Let's establish some variables we'll need for the actual Code
PFE=[]
UNH=[]
BPoints = []
PFEStartPrice = 43.512924 #NKE 2015 43.512924 #JNJ 113.289429 #PFE 21.793074
UNHStartPrice = 72.221115 # CAT 2015 72.221115 #UNH 88.923103
# Start Prices exist to norm the data, so the scaled basis makes sense.
Basis = []

class CorrFlows(bt.Strategy):

    #-0.2, 0.35 good.
    params = (
    ("EnterBasisValue", -0.15),
    ("ExitBasisValue", 0.35)
    )
    #Potential Params - Basis vector? Trading window?
    
    def __init__(self):
        self.order = None
        self.dataclose = self.datas[0].close
        
    def notify_order(self, order):
        # No more orders
        self.order = None    
    
    def next(self):
        
        # Get today's date, day of month, and 
        # last day of current month
        dt_ = self.datas[0].datetime.date(0)
        
        #Length - 250
        
        if len(PFE) < 250:
            PFE.append(self.data0.adjclose[0]/PFEStartPrice)
        
        if len(UNH) < 250:
            UNH.append(self.data1.adjclose[0]/UNHStartPrice)
            
        if len(PFE) == 250:
            PFE.pop(0)
            PFE.append(self.data0.adjclose[0]/PFEStartPrice)
            
        if len(UNH) == 250:
            UNH.pop(0)
            UNH.append(self.data1.adjclose[0]/UNHStartPrice)
     
            
        BasisDiff = ScaleBasis(PFE, UNH)
        
        if not BasisDiff is None:
            BPoints.append(BasisDiff)
            Basis.append(BasisDiff)
            # If an order is pending, exit
            if self.order:
                return

            # Check if we are in the market. If self.position = TRUE, False True : False. If self.position = FALSE, False False = TRUE
            # The command below executes when we are not in the market.
            if not self.position:

                # Condition for Buying
                if BasisDiff < self.params.EnterBasisValue:

                    # Buy the entire portfolio
                    self.order = self.order_target_percent(target=1)

                    print(f"BUY at {self.data_close[0]}")

            # We are in the market
            else:

                # If we're long
                if self.position.size > 0:

                    # And not ___, then sell.
                    if BasisDiff > self.params.ExitBasisValue:

                        print(f"CLOSE Long at {self.data_close[0]}")

                        self.order = self.order_target_percent(target=0.0)

                # If we're short
                if self.position.size < 0:

                    # And not ___, then cover.
                    if BasisDiff < -0.5:

                        print(f"COVER Short at {self.data_close[0]}")

                        # self.order = self.close()
                        self.order = self.order_target_percent(target=0.0)


# In[272]:


data0 = openbb_data_to_bt_data(
    "NKE", 
    start_date="2015-01-01",
    end_date="2019-06-30"
)

data1 = openbb_data_to_bt_data(
    "CAT", 
    start_date="2015-01-01",
    end_date="2019-06-30"
) 
cerebro = bt.Cerebro(stdstats=False)

cerebro.adddata(data0)
cerebro.adddata(data1)
cerebro.broker.setcash(10000.0)

cerebro.addstrategy(CorrFlows)

cerebro.addobserver(bt.observers.Value)

cerebro.addanalyzer(
    bt.analyzers.Returns, _name="returns"
)
cerebro.addanalyzer(
    bt.analyzers.TimeReturn, _name="time_return"
)

backtest_result = cerebro.run()


# In[1162]:


# Get the strategy returns as a dictionary
returns_dict = backtest_result[0].analyzers.time_return.get_analysis()

# Convert the dictionary to a DataFrame
returns_df = (
    pd.DataFrame(
        list(returns_dict.items()),
        columns = ["date", "return"]
    )
    .set_index("date")
)


# In[1163]:


# In[273]:


bench = yf.download(
    "NKE",
    "2015-01-01",
    "2019-06-30"
)["Adj Close"]

qs.reports.metrics(
    returns_df,
    benchmark=bench,
    mode="full"
)


# ## Some notes: 
# 
# Cframe is the sector matrix, derived from a basis vector's outer product and taking the mean of all the stocks in a sector's valuation.
# 
# mframe is the outer product itself. 
# 
# df is the dataframe containing all our stock data.
# 
# meanframe is the dataframe taking the mean of all the years.




# In[270]:


plt.plot(BPoints)


# In[249]:


yf.download("NKE","2019-01-01")


# In[ ]:




