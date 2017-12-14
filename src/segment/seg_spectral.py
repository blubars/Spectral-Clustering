import pickle
from scipy.sparse import lil_matrix
import numpy as np
from helper import rgb2gray, plot_gray, image2affinity
import scipy.sparse.linalg as ling
from scipy.sparse import csr_matrix, csc_matrix, identity
from scipy.sparse.linalg import inv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_file = 'trivedi.png'
img = mpimg.imread(image_file)
A = image2affinity(image_file, r=10)

from numpy.linalg import norm
from math import exp

num_clusters = 5
sigma_sq = .03

D = csc_matrix(np.diag(np.sum(A, axis=1).A1))

print ('D done')

# Construct Laplacian Matrix:
#   L = D^{-1/2} A D^{-1/2} --> L[i,j] = -A[i,j]/sqrt(d_i * d_j)

# D^{-1/2}:
# Dinvsq = inv(D)
Dinvsq = np.sqrt(inv(D))

print ('Inv done')
L = identity(D.shape[0]) - Dinvsq.dot(A)
# L = D-A
# L = Dinvsq.dot(L)
# L = L.dot(Dinvsq)

#L = np.identity(len(A)) - L
print ('L done')
# print(L)
# print(np.isclose(L[0,1], -A[0,1]/np.sqrt(D[1,1]*D[0,0])))

# Find the K largest eigenvectors of L
# eigvals, eigvects = np.linalg.eigh(L)
eigvals, eigvects = ling.eigs(L, k=num_clusters)

print ('Eigen done')

best_eigens = []


# TODO: verify not 1, and verify orthogonal
LX = eigvects[:, range(num_clusters)]
LX = LX/np.linalg.norm(LX, axis=1).reshape(-1,1)
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



gray = rgb2gray(img)
x_shape = gray.shape[0]
y_shape = gray.shape[1]
# plot_gray(gray)
for key, indices in label2ind.items():
    gray_clus = np.zeros(gray.shape)
    for ind in indices:
        gray_clus[ind//y_shape, ind%y_shape] = gray[ind//y_shape, ind%y_shape]
    plot_gray(gray_clus, img_name='Figure '+str(key), show=False)



# plot results
# plot_results(LX, y_pred, 3, "Spectral clustering: embedded domain")
# plot_results(X, y_pred, 3, "Spectral clustering: original domain")
