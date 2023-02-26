import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

#print(type(data_ron))
X, y = make_regression(n_samples = 300, n_features = 20, 
                       n_informative = 10, noise = 100, random_state = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.2, random_state = 10)

#普通线性
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

#岭回归
model_r = Ridge(alpha = 0.5)
model_r.fit(X_train, y_train)

#Lasso回归
model_l = Lasso(alpha = 0.1)
model_l.fit(X_train, y_train)

#ElasticNet回归
model_en = ElasticNetCV(cv = 8)
model_en.fit(X_train, y_train)

print("lambda:{:.4f}, p:{}".format(model_en.alpha_, model_en.l1_ratio_))


def get_score(model):
    print("=====================================")
    print("train_score:{:.4f}, test_score:{:.4f}".format(model.score(X_train, y_train), 
          model.score(X_test, y_test)))


for item in [model_lr, model_r, model_l, model_en]:
    get_score(item)


