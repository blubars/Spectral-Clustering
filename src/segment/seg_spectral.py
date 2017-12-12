import cPickle as pickle
from scipy.sparse import lil_matrix
import numpy as np
from helper import rgb2gray, plot_gray
import scipy.sparse.linalg as ling

A = pickle.load(open('aff3.mat', 'rb')).todense()

from numpy.linalg import norm
from math import exp

num_clusters = 9
sigma_sq = .03

D = np.zeros(A.shape)
for i in range(A.shape[0]):
    D[i,i] = np.sum(A[i,:])

print ('D done')

# Construct Laplacian Matrix:
#   L = D^{-1/2} A D^{-1/2} --> L[i,j] = -A[i,j]/sqrt(d_i * d_j)

# D^{-1/2}:
Dinvsq = np.sqrt(np.linalg.inv(D))

print ('1')

L = np.dot(Dinvsq, D-A)
L = np.dot(L, Dinvsq)
#L = np.identity(len(A)) - L

# print(L)
# print(np.isclose(L[0,1], -A[0,1]/np.sqrt(D[1,1]*D[0,0])))

# Find the K largest eigenvectors of L
eigvals, eigvects = np.linalg.eigh(L)
# eigvals, eigvects = ling.eigs(L, k=11)

print ('2')

best_eigens = []
for i in range(L.shape[0]):
    #if np.isclose(eigvals[i], 1):
    if 0:
        continue
    else:
        if len(best_eigens) == num_clusters:
            break
        else:
            best_eigens.append(i)
print(best_eigens)

# TODO: verify not 1, and verify orthogonal
LX = eigvects[:, range(num_clusters)]
LX = LX/np.sum(LX, axis=0)
# print(LX)
from sklearn.cluster import KMeans
# verify: L v = \lamda v
print("Eigenvalues:")
print(eigvals)
# print("Verify an eigenvector + eigenvalue")
# print(np.isclose(np.dot(L,eigvects[:,1]), eigvals[1]*eigvects[:,1]))
km2 = KMeans(n_clusters=num_clusters, n_init=20)
km2.fit(LX)
y_pred = km2.labels_

from collections import defaultdict

label2ind = defaultdict(list)
for i, lab in enumerate(y_pred):
    label2ind[lab]+=[i]
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread('test2.png')
gray = rgb2gray(img)
x_shape = gray.shape[0]
y_shape = gray.shape[1]
plot_gray(gray)
for key, indices in label2ind.items():
    gray_clus = np.zeros(gray.shape)
    for ind in indices:
        gray_clus[ind/y_shape, ind%y_shape] = gray[ind/y_shape, ind%y_shape]
    plot_gray(gray_clus)



# plot results
# plot_results(LX, y_pred, 3, "Spectral clustering: embedded domain")
# plot_results(X, y_pred, 3, "Spectral clustering: original domain")