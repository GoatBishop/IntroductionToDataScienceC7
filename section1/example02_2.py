# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from mlxtend.evaluate import bias_variance_decomp

#读取原始数据

data_ron_deal = pd.read_csv("../data/data_ron_deal.csv")


y = data_ron_deal["RON"]
X = data_ron_deal.iloc[:, 1: (len(data_ron_deal) + 1)]

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 10)

#构建模型
model = LinearRegression()
model.fit(X_train, y_train)
print(round(model.score(X_train, y_train), 4))
print(round(model.score(X_test, y_test), 4))


#标准化(随机梯度下降法之前需要标准化)
#stand_train = StandardScaler()
#stand_train.fit(X_train)
#X_train_standard = stand_train.transform(X_train)
#X_test_standard = stand_train.transform(X_test)
#model = SGDRegressor()
#model.fit(X_train_standard, y_train)
#print(round(model.score(X_train_standard, y_train), 4))
#print(round(model.score(X_test_standard, y_test), 4))


def mape(y_true, y_pre):
    n = len(y_true)
    mape = (sum(np.abs((y_true - y_pre)/y_true))/n)*100
    return mape

y_hat = model.predict(X_test)
MSE = metrics.mean_squared_error(y_test, y_hat)
RMSE = metrics.mean_squared_error(y_test, y_hat)**0.5
MAE = metrics.mean_absolute_error(y_test, y_hat)
MAPE = mape(y_test, y_hat)


print("MSE:{:.4f}, RMSE:{:.4f}, MAE:{:.4f}, MAPE:{:.4f}".format(MSE, RMSE, MAE, MAPE))
#MSE:0.0370, RMSE:0.1924, MAE:0.1511, MAPE:0.1710

#不进行转换会报错
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

mse, bias, var = bias_variance_decomp(model, X_train, y_train, X_test, y_test,loss='mse', 
                                     num_rounds=30, random_seed=1)

print("mse:{:.4f}, bias:{:.4f}, var:{:.4f}".format(mse, bias, var))
#mse:0.3947, bias:0.3443, var:0.0504
