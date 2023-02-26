# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

n = 50
#np.random.seed(1)
#x = np.linspace(-5,5,n)
#y = x**3 + x**2 - 15*x + np.random.normal(0, 5, n)
#
#model = LinearRegression()
#model.fit(x.reshape(-1,1), y.reshape(-1,1))
#y_hat = model.predict(x.reshape(-1,1))
#print(model.intercept_)
##[8.84185277]
#print(model.coef_)
##[[0.82887073]]
#plt.scatter(x, y)
#plt.plot(x, y_hat, color='r', linestyle='--')
#plt.show()

x = np.linspace(-5,5,n)
for item in range(3):
    y = x**3 + x**2 - 15*x + np.random.normal(0, 7, n)
    model = LinearRegression()
    model.fit(x.reshape(-1,1), y.reshape(-1,1))
    y_hat = model.predict(x.reshape(-1,1))
    plt.subplot(1, 3, item + 1)
    plt.scatter(x, y)
    plt.plot(x, y_hat, color='r', linestyle='--')
    plt.show()

#x = np.linspace(-5,5,n)
#X = pd.DataFrame([x, x**2, x**3, x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11,
#                  x**12, x**13, x**14, x**15, x**16, x**17, x**18, x**19, x**20]).T
#for item in range(3):
#    y = x**3 + x**2 - 15*x + np.random.normal(0, 7, n)
#    model = LinearRegression()
#    model.fit(X, y.reshape(-1,1))
#    y_hat = model.predict(X)
#    plt.subplot(1, 3, item + 1)
#    plt.scatter(x, y)
#    plt.plot(x, y_hat, color='r', linestyle='--')
#    plt.show()

