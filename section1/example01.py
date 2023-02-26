# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#读取数据
data_cars = pd.read_csv("../data/cars.csv", 
                        usecols = ["speed", "dist"])

print(data_cars.head())

speed = data_cars["speed"]
dist = data_cars["dist"]

#plt.scatter(speed, dist)
#plt.title("Scatter plot of vehicle speed and braking distance")
#plt.show()

import sympy
#设定回归系数
alpha, beta = sympy.symbols("alpha beta")

#设定损失函数
L = 0.5*np.sum((dist - beta*speed - alpha)**2)

#求偏导
print(sympy.diff(L, alpha))
#50.0*alpha + 770.0*beta - 2149.0
print(sympy.diff(L, beta))
#770.0*alpha + 13228.0*beta - 38482.0
f1 = sympy.diff(L, alpha)
f2 = sympy.diff(L, beta)

#求解线性方程组
outcome = sympy.solve([f1, f2], [alpha, beta])
print(outcome)
#{alpha: -17.5790948905109, beta: 3.93240875912409}

#绘图
alpha_num = outcome[alpha]
beta_num = outcome[beta]
##得到预测值
#dist_pre = beta_num*speed + alpha_num
#plt.scatter(speed, dist)
#plt.plot(speed, dist_pre, c = "r")
#plt.title("Fitting results")
#plt.show()


#迭代法
import random


#定义递推关系,更新迭代变量
def update_var(old_alpha, old_beta, y, x, learning_rate):
    len_x = len(x)
    alpha_delta = np.sum(-(y - old_beta*x - old_alpha))/len_x
    beta_delta = np.sum(-x*(y - old_beta*x - old_alpha))/len_x
    new_alpha = old_alpha - learning_rate*alpha_delta
    new_beta = old_beta - learning_rate*beta_delta
    
    return (new_alpha, new_beta)

#迭代
def iterative_func(y, x, start_alpha, start_beta,
                   learning_rate, iterative_num, 
                   sample_num):
    alpha_list = []
    beta_list = []
    alpha = start_alpha
    beta = start_beta
    num_list = list(range(1, len(y)))
    for i in range(iterative_num):
        alpha_list.append(alpha)
        beta_list.append(beta)
        random.shuffle(num_list)
        
        index = num_list[:sample_num]
        alpha, beta = update_var(alpha, beta, 
                                 y[index], x[index], learning_rate)
#        print("alpha: {}, beta:{}".format(alpha, beta))
    return (alpha_list, beta_list)

#在[0, 10)之间按照均匀分布随机产生alpha和beta的初始值
start_alpha = np.random.random()*10
start_beta = np.random.random()*10

#设置学习率为0.01,迭代次数为500次,每次计算8个数据
learning_rate = 0.002
iterative_num = 20000
sample_num = 16

alpha_list, beta_list = iterative_func(dist, speed, start_alpha, start_beta,
                                     learning_rate, iterative_num,
                                     sample_num)

print("alpha: {}, beta:{}".format(alpha_list[-1], beta_list[-1]))

#写出
import csv
parameter_data = zip(alpha_list, beta_list)
with open("./outcome/gradient_descent_parameter.csv", 'w', newline = '') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["alpha","beta"])
    csv_writer.writerows(parameter_data)

#绘图

#plt.subplot(121)
#plt.plot(alpha_list)
#plt.title("alpha change process")
#plt.subplot(122)
#plt.plot(beta_list)
#plt.title("beta change process")
#plt.show()


#判定系数R2
dist_pre = beta_num*speed + alpha_num
dist_mean = np.mean(dist)
R_2 = np.sum((dist_pre - dist_mean)**2)/np.sum((dist - dist_mean)**2)
print(R_2)
#0.651079380758251

#预测
new_speed = pd.Series([10, 20, 30])
new_dist_pre = beta_num*new_speed + alpha_num
print(new_dist_pre)
