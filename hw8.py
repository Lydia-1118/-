# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:04:38 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd


hw8_csv = pd.read_csv('hw8.csv')
hw8_dataset = hw8_csv.to_numpy(dtype = np.float64)

X0 = hw8_dataset[:, 0:2]
y = hw8_dataset[:, 2]

# write your code here
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=6)
X_poly = poly.fit_transform(X0)

clf = LogisticRegression(C=1e5, max_iter=1000)
clf.fit(X_poly, y)

fig = plt.figure(dpi=288)

# write your code here
x_min, x_max = -15, 15
y_min, y_max = -10, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_poly = poly.transform(grid_points)
Z = clf.predict(grid_poly)

Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn, alpha=0.3)

plt.plot(X0[y == 1, 0], X0[y == 1, 1], 'r.', label='$\omega_1$')
plt.plot(X0[y == -1, 0], X0[y == -1, 1], 'b.', label='$\omega_2$')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axis('equal')
plt.legend()
plt.show()

