# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 09:37:05 2021

@author: htchen
"""
# If this script is not run under spyder IDE, comment the following two lines.
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')

import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import cv2

import numpy
# 修正 NumPy 2.x 的導入錯誤，防止無限遞迴
if not hasattr(numpy, 'linalg'):
    import numpy.linalg
    numpy.linalg = numpy.linalg

plt.rcParams['figure.dpi'] = 144 

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

# SVD: A = U * Sigma * V^T
# V: eigenvector matrix of A^T * A; U: eigenvector matrix of A * A^T 
def mysvd(A):
    lambdas, V = myeig(A.T @ A, symmetric=True)
    lambdas, V = np.real(lambdas), np.real(V)
    # if A is full rank, no lambda value is less than 1e-6 
    # append a small value to stop rank check
    lambdas = np.append(lambdas, 1e-12)
    rank = np.argwhere(lambdas < 1e-6).min()
    lambdas, V = lambdas[0:rank], V[:, 0:rank]
    U = A @ V / np.sqrt(lambdas)
    Sigma = np.diag(np.sqrt(lambdas))
    return U, Sigma, V

# 讀取影像檔, 並保留亮度成分
img = cv2.imread('svd_demo1.jpg', cv2.IMREAD_GRAYSCALE)

# convert img to float data type
A = img.astype(dtype=np.float64)

# SVD of A
U, Sigma, V = mysvd(A)
VT = V.T


def compute_energy(X: np.ndarray):
    # return energy of X
    # For more details on the energy of a 2D signal, see the 
    # class notebook: 內容庫/補充說明/Energy of a 2D Signal.
    # remove pass and write your code here
    return np.sum(np.square(X))
    
    
# img_h and img_w are image's height and width, respectively
img_h, img_w = A.shape
# Compute SNR
keep_r = 201
rs = np.arange(1, keep_r)


# compute energy of A, and save it to variable Energy_A
energy_A = compute_energy(A)

# Decalre an array to save the energy of noise vs r.
# energy_N[r] is the energy of A - A_bar(sum of the first r components)
energy_N = np.zeros(keep_r) # energy_N[0]棄置不用

for r in rs:
    # A_bar is the sum of the first r comonents of SVD
    # A_bar is an approximation of A
    A_bar = U[:, 0:r] @ Sigma[0:r, 0:r] @ VT[0:r, :] 
    Noise = A - A_bar 
    energy_N[r] = compute_energy(Noise) 

# 計算snr和作圖
# write your code here
snrs = []
for r in rs:
    # A_bar 是前 r 個成份組成的近似影像
    # Noise (Nr) 是原圖與近似圖的差異
    if energy_N[r] > 0:
        snr_r = 10 * np.log10(energy_A / energy_N[r])
    else:
        snr_r = 0
    snrs.append(snr_r)

# 繪製 SNR 隨 r 變化的圖形
plt.figure(figsize=(8, 5))
plt.plot(rs, snrs, 'r-')
plt.title('$A_{SNR}[r]$ vs. $r$')
plt.xlabel('r')
plt.ylabel('SNR (dB)')
plt.grid(True)
plt.show()

# --- 驗證雜訊能量與特徵值的關係 ---
# Nr 的能量等於從 r+1 到 n 個特徵值之和
lambdas_all, _ = myeig(A.T @ A, symmetric=True)
r_check = 100
print(f"雜訊能量 (矩陣減法): {energy_N[r_check]}")
print(f"雜訊能量 (特徵值求和): {np.sum(lambdas_all[r_check:])}")

# --------------------------
# verify that energy_N[r] equals the sum of lambda_i, i from r+1 to i=n,
# lambda_i is the eigenvalue of A^T @ A
# write your code here
# 取得 A^T @ A 的所有特徵值 (lambdas)
lambdas_all, _ = myeig(A.T @ A, symmetric=True)

# 隨機選一個 r 來驗證，例如 r = 50
r_test = 50
energy_from_loop = energy_N[r_test]
energy_from_lambdas = np.sum(lambdas_all[r_test:])

print(f"驗證 r={r_test}:")
print(f"從矩陣減法得到的雜訊能量: {energy_from_loop}")
print(f"從特徵值求和得到的雜訊能量: {energy_from_lambdas}")
print(f"誤差值: {abs(energy_from_loop - energy_from_lambdas)}")
