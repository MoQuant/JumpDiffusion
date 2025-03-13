import numpy as np
import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

#np.random.poisson(lambda_ * dt, size=N)
#Z = np.random.randn(N)
#dW_t = np.sqrt(dt) * Z


ticker = 'AMZN'
data = MarketData(ticker)

close = data['adjClose'].values
ror = close[1:]/close[:-1] - 1.0

St = close[-1]
mu = np.mean(ror)
vi = np.std(ror)

T = 1.0/365.0
N = 300
dt = T / N

jump = np.arange(0, 1.01, 0.01)
timex = list(range(N+1))

x, y = np.meshgrid(timex, jump)
z = []

for j in jump:
    z.append(JumpDiffusion(St, mu, vi, dt, j, N))

z = np.array(z)

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection='3d')


ax.plot_surface(x, y, z, cmap='viridis', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Jump')
ax.set_zlabel('Stock Price')
ax.set_title('Jump Diffusion Stock Price Simulation for ' + ticker)

plt.show()







