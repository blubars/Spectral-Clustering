#!/usr/bin/env python3

###############################################################
#  FILE DESCRIPTION
###############################################################
# Clustering using sklearn's implementation of spectral clustering.
# Just for reference, so we can compare the performance of our
# implementation.

###############################################################
#  IMPORTS
###############################################################
import time
import numpy as np
from sklearn import datasets
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt


###############################################################
#  GENERATED TOY DATASETS:
#  Note: This code section is copied & modified from sklearn's 
#  clustering example:
#  http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
###############################################################
# Generate datasets. We choose the size big enough to see the scalability
n_samples = 1000
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)

datasets = [noisy_circles, noisy_moons]

###############################################################

###############################################################
#  CLUSTERING PARAMETERS
###############################################################
sc_params = {'n_clusters': 2, 'affinity': 'nearest_neighbors'}

# *[N_CLUSTERS: integer, optional] TODO: why is this optional?
# *[GAMMA: float, def=1.0]: kernel coefficient
# *[AFFINITY: string/array-like/callable]: kernel to use to
#    construct affinity matrix
# *[N_NEIGHBORS: int]: # neighbors to use when constructing 
#    affinity matrix
# *[ASSIGN_LABELS: {'kmeans','discrete'}]: strategy to use to 
#    assign labels in the embedding space. see docs.
# *[N_INIT]: number of times to run k-means

###############################################################
#  AFFINITY MATRIX
###############################################################
# Ways to provide an affinity matrix:
#  - precomputed: user provides affinity matrix
#  - sklearn constructs one using a kernel function, 
#      (e.g. Gaussian/rbf or euclidean distance)
#  - k-nearest neighbors connectivity matrix

###############################################################
#  MAIN
###############################################################
def main():
    plot_num = 1
    plot_shape = (2+2, 2*len(datasets)+2)
    fig = plt.figure(1, plot_shape)
    plt.title('Spectral Clustering for different datasets')
    for i, dataset in enumerate(datasets):
        print("Dataset {}:".format(i))
        X, y = dataset
        model = SpectralClustering(
                    n_clusters=sc_params['n_clusters'], 
                    affinity=sc_params['affinity'])
        ts_start = time.time()
        model.fit(X)
        ts_end = time.time()
        y_pred = model.labels_

        # plot:
        colors = ['#377eb8', '#ff7f00', '#4daf4a']
        y_colors = [colors[label] for label in y_pred]
        plt.subplot(len(datasets), 1, plot_num)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=y_colors)
        plt.gca().set_aspect('equal', 'box')
        plot_num += 1
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

