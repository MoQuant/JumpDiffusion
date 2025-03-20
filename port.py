import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import time
import random 

jump = lambda: random.randint(0, 10)/10.0

def MarketData(ticker):
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey='
    resp = requests.get(url).json()
    return pd.DataFrame(resp['historical'])[::-1]

def JumpDiffusion(St, mu, vi, dt, J, N):
    prices = [St]
    dWt = np.random.randn(N)
    dNt = np.random.poisson(0.5*dt, size=N)
    for i in range(N):
        St += mu*St*dt + vi*St*dWt[i] + St*(np.exp(J) - 1)*dNt[i]
        prices.append(St)
    return prices

def StochasticModel(close, T, N):
    dt = T / N
    ror = close[1:]/close[:-1] - 1.0
    m, n = ror.shape
    er = (1/m)*np.ones(m).dot(ror)
    vi = np.sqrt(np.diag((1/(m-1))*(ror - er).T.dot(ror - er)))
    
    sclose = []
    for t in range(n):
        prices = JumpDiffusion(close[-1, t], er[t], vi[t], dt, jump(), N)
        sclose.append(prices)
    
    return np.array(sclose).T

def Stats(close):
    ror = close[1:]/close[:-1] - 1.0
    m, n = ror.shape
    er = (1/m)*np.ones(m).dot(ror)
    cov = (1/(m-1))*(ror - er).T.dot(ror - er)
    sd = np.sqrt(np.diag(cov))
    return sd, er, cov

def EF(close, n=50):

    def Optimize(cv, mu, target_rate):
        cv = (2.0*cv).tolist()
        m = len(cv)
        for i in range(m):
            cv[i].append(mu[i])
            cv[i].append(1.0)
        cv.append(mu.tolist() + [0, 0])
        cv.append(np.ones(m).tolist() + [0, 0])
        cv = np.array(cv)
        b = np.array(np.zeros(m).tolist() + [target_rate, 1.0])
        weights = np.linalg.inv(cv).dot(b)
        return weights[:-2]

    sd, mu, cov = Stats(close)
    m0 = np.min(mu)
    m1 = np.max(mu)
    dm = (m1 - m0)/(n - 1)
    ux, uy = [], []
    for i in range(n):
        target_rate = m0 + i*dm
        W = Optimize(cov, mu, target_rate)
        risk = np.sqrt(W.T.dot(cov.dot(W)))
        reward = W.T.dot(mu)
        ux.append(risk)
        uy.append(reward)
    return ux, uy

tickers = ['AAPL','AMZN','MSFT','JPM','TSLA','SPY']
close = []

for stock in tickers:
    resp = MarketData(stock)['adjClose'].values.tolist()
    close.append(resp)
    print(stock, " has loaded")
    time.sleep(0.3)

close = np.array(close).T

T = 1.0
N = 365



fig = plt.figure()
ax = fig.add_subplot(111)

while True:
    sclose = StochasticModel(close, T, N)

    sd1, mu1, cov1 = Stats(close)
    sd2, mu2, cov2 = Stats(sclose)

    ax.cla()
    ax.scatter(sd1, mu1, color='red', label='Historical')
    ax.scatter(sd2, mu2, color='blue', label='Stochastic')

    for x1, y1, x2, y2, stock in zip(sd1, mu1, sd2, mu2, tickers):
        ax.annotate(stock, xy=(x1, y1))
        ax.annotate(stock, xy=(x2, y2))

    ax.legend()

    ux1, uy1 = EF(close)
    ux2, uy2 = EF(sclose)

    ax.plot(ux1, uy1, color='red')
    ax.plot(ux2, uy2, color='blue')

    plt.pause(2)

plt.show()