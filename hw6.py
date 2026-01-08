# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""

from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
import numpy.linalg as la
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# calculate the eigenvalues and eigenvectors of a squared matrix
# the eigenvalues are decreasing ordered
def myeig(A, symmetric=False):
    if symmetric:
        lambdas, V = np.linalg.eigh(A)
    else:
        lambdas, V = np.linalg.eig(A)
    # lambdas, V may contain complex value
    lambdas_real = np.real(lambdas)
    sorted_idx = lambdas_real.argsort()[::-1] 
    return lambdas[sorted_idx], V[:, sorted_idx]


# class 1
mean1 = np.array([0, 5])
sigma1 = np.array([[0.3, 0.2],[0.2, 1]])
N1 = 200
X1 = np.random.multivariate_normal(mean1, sigma1, N1)

# class 2
mean2 = np.array([3, 4])
sigma2 = np.array([[0.3, 0.2],[0.2, 1]])
N2 = 100
X2 = np.random.multivariate_normal(mean2, sigma2, N2)

# m1: mean of class 1
# m2: mean of class 2
m1 = np.mean(X1, axis = 0, keepdims=1)
m2 = np.mean(X2, axis = 0, keepdims=1)

# write you code here
# Sw = S1 + S2
S1 = (X1 - m1).T @ (X1 - m1)
S2 = (X2 - m2).T @ (X2 - m2)
Sw = S1 + S2

# 根據 LDA 公式尋找最佳投影方向 w
# w = inv(Sw) * (m1 - m2)
w = la.inv(Sw) @ (m1 - m2).T

# 將 w 單位化 (向量長度變為 1)，方便後續投影計算與繪圖
w = w / la.norm(w)

plt.figure(dpi=288)

plt.plot(X1[:, 0], X1[:,1], 'r.')
plt.plot(X2[:, 0], X2[:,1], 'g.')

# write you code here

y1 = X1 @ w
p1 = y1 @ w.T
plt.plot(p1[:, 0], p1[:, 1], 'r.', markersize=2) # 繪製投影後的紅色點

# 2. 將類別 2 投影到 w 方向上
y2 = X2 @ w
p2 = y2 @ w.T
plt.plot(p2[:, 0], p2[:, 1], 'g.', markersize=2) # 繪製投影後的綠色點

# 3. 畫出投影軸 (一條通過原點且方向為 w 的直線)
# 設定一條足夠長的線段來表示投影軸
axis_line = np.linspace(-5, 5, 10)
plt.plot(axis_line * w[0], axis_line * w[1], 'k-', alpha=0.2)

plt.axis('equal')  
plt.show()

