from sklearn.cluster import SpectralClustering
from helper import rgb2gray, plot_gray, image2affinity
import matplotlib.image as mpimg
import pickle
import numpy as np
from scipy.sparse import csr_matrix
image_file = 'test3.png'
# aff_mat = pickle.load(open('aff4.mat', 'rb'))
aff_mat = image2affinity(image_file, r=5)

print ("converting to sparse")
aff_mat = csr_matrix(aff_mat)
print ("converted. running SC")
cluster = SpectralClustering(n_clusters=10, affinity='precomputed', eigen_solver='amg')
y_pred = cluster.fit_predict(aff_mat)

from collections import defaultdict

label2ind = defaultdict(list)
for i, lab in enumerate(y_pred):
    label2ind[lab]+=[i]
import matplotlib.pyplot as plt


img = mpimg.imread(image_file)
gray = rgb2gray(img)
x_shape = gray.shape[0]
y_shape = gray.shape[1]
plot_gray(gray)

for key, indices in label2ind.items():
    gray_clus = np.zeros(gray.shape)
    for ind in indices:
        gray_clus[ind/y_shape, ind%y_shape] = gray[ind/y_shape, ind%y_shape]
    plot_gray(gray_clus)
