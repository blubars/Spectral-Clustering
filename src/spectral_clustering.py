#!/usr/bin/env python3

###############################################################
#  FILE DESCRIPTION
###############################################################
# Our implementation of Spectral Clustering.
# Interface follows sklearn's API convention of .fit()

###############################################################
#  IMPORTS
###############################################################
from math import exp
import numpy as np
from numpy.linalg import norm
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import scipy
import pyamg

plot_colors = ['#377eb8', '#ff7f00', '#4daf4a']

###############################################################
#  UTILITY FUNCTIONS
###############################################################
# create toy datasets for cluster experiments
def generate_datasets(num_samples=100):
    noisy_circles = datasets.make_circles(
            n_samples=num_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(
            n_samples=num_samples, noise=0.05)
    data = [noisy_circles, noisy_moons]
    return data

def plot_results(X, y, fignum, title):
    # plot with true labels
    fig = plt.figure(fignum)
    plt.title(title)
    colors = ['#377eb8', '#ff7f00']
    y_colors = [colors[label] for label in y]
    plt.scatter(X[:,0], X[:,1], color=y_colors)
    plt.gca().set_aspect('equal')
    plt.show()

def add_subplot(subplot_list, X, y, title, plot_type=None, limits=None):
    # plot types: None, 'sc-circle'
    subplot_list.append((X, y, title, plot_type, limits))

def plot_subplots(fignum, subplot_list):
    # make subplots and display
    num_plots = len(subplot_list)
    num_rows = num_plots // 3
    if num_plots % 3 != 0:
        num_rows += 1
    if num_plots < 3:
        num_cols = num_plots
    else:
        num_cols = 3
    plot_shape = (2*num_cols+2,2*num_rows+2)
    fig = plt.figure(fignum, plot_shape)
    for i, subplot_item in enumerate(subplot_list):
        #print("rows:{}, cols:{}, i:{}".format(num_rows, num_cols, i+1))
        X, y, title, plot_type, limits = subplot_item
        y_colors = [plot_colors[label] for label in y]
        plt.subplot(num_rows, num_cols, i+1)
        plt.scatter(X[:, 0], X[:, 1], s=10, color=y_colors)
        ax = plt.gca()
        if plot_type == 'sc-circle':
            limits = ([-1.2,1.2],[-1.2,1.2])
            # draw unit circle
            circle = Circle((0, 0), 1, color='gray', alpha=0.2)
            ax.add_artist(circle)
        if limits:
            ax.set(xlim=limits[0], ylim=limits[1])
        ax.set_aspect('equal')#, 'box')
        ax.set_title(title, fontsize='10')
    fig.tight_layout()
    plt.show()

###############################################################
#  CLASS: SPECTRAL CLUSTERING:
#    This is the vanilla implementation from Ng-Jordan-Weiss
###############################################################
class SpectralClustering:
    def __init__(self, num_clusters, sigma_sq=None, debug=False):
        self.num_clusters = num_clusters
        self.sigma_sq = sigma_sq
        self.scale_affinity = 1
        self._debug = debug
        self._L = None
        self._LX = None
        self._labels = None

    def fit_params(self, X):
        if self.sigma_sq is None:
            self.sigma_sq = self.guess_sigma_sq(X)
        if self.sigma_sq < 1:
            # scale values to avoid numerical instability
            self.scale_affinity = 1 / self.sigma_sq

    def fit_predict(self, X):
        return self.fit(X)

    def fit(self, X):
        # (0) -------
        # Set up data autoparams if missing
        self.fit_params(X)
        # (1) -------
        # construct affinity matrix
        A = self.make_affinity(X)
        # (2) -------
        # Construct Laplacian matrix
        L = self.make_laplacian(A)
        # (3) -------
        # Find the K largest eigenvectors of L
        LX = self.make_eigenvector_matrix(L)
        # (4) -------
        # Finally, do clustering on reduced space using KMeans:
        km = KMeans(n_clusters=2, n_init=20)
        km.fit(LX)
        self._labels = km.labels_
        self.auto_eval_cluster()
        return self._labels


    def auto_eval_cluster(self):
        # 'cluster quality' can be approximated by tightness of
        # clustering in transformed eigenspace.
        stdlist = []
        for c in range(self.num_clusters):
            std = np.std(self._LX[self._labels == c], axis=0)
            stdlist.append(np.average(std))
        #print("Std dev of clusters: {}".format(stdlist))
        if np.average(stdlist) > 0.1:
            print("High std deviation (%.2f) in embedded space, maybe try smaller sigma" % np.average(stdlist))


    def guess_sigma_sq(self, X):
        var = np.var(X) / self.num_clusters**2
        print("Guessing variance:{}".format(var))
        return var

    def gaus_affinity_kernel(self, x1, x2):
        # compute the affinity of samples X1, X2
        # scale values to avoid numerical instability
        gaus = exp(-(norm(x1-x2)**2)/(2*self.sigma_sq))
        return self.scale_affinity * gaus

    def make_affinity(self, X, kernel=None):
        if kernel is None:
            kernel = self.gaus_affinity_kernel
        A = np.zeros((len(X), len(X)))
        for i in range(len(X)-1):
            for j in range(i+1, len(X)):
                A[i,j] = kernel(X[i], X[j])
                A[j,i] = A[i,j]
        if self._debug:
            # print affinity matrix
            np.set_printoptions(precision=2)
            print(A)
        return A

    def make_laplacian(self, A):
        # Construct Laplacian Matrix:
        #   L = D^{-1/2} A D^{-1/2} -->
        #   L[i,j] = -A[i,j]/sqrt(d_i * d_j)
        # D = diagonal degree matrix
        D = np.diag(np.sum(A, axis=0))
        # construct normalized laplacian
        Dinvsq = np.sqrt(np.linalg.inv(D)) * 1000
        L = np.dot(Dinvsq, D - A)
        L = np.dot(L, Dinvsq)
        self._L = L
        if self._debug:
            print("Laplacian:")
            print(L)
            print(np.isclose(L[0,1], -A[0,1]/np.sqrt(D[1,1]*D[0,0])))
        return L

    def make_eigenvector_matrix(self, L):
        # find eigenvalues/eigenvectors, pick best ones
        # to use for spectral clustering, construct matrix
        # out of eigenvectors, normalize, and return
        eigvals, eigvects = np.linalg.eigh(L)
        best_eigens = [i for i in range(self.num_clusters)]
        if self._debug:
            print("Best eigenvalues: {}".format(best_eigens))
        LX = np.array(eigvects[:,best_eigens])
        # verify orthogonal
        for i in range(self.num_clusters-1):
            for j in range(i+1, self.num_clusters):
                if not np.isclose(np.dot(LX[:,i], LX[:,j]), 0):
                    print("WARNING: eigenvectors {},{} not orthogonal!".format(i,j))
        # normalize new eigenvector-column-matrix
        LX = (LX.T / np.linalg.norm(LX, axis=1)).T
        self._LX = LX
        if self._debug:
            print(LX)
            # verify: L v = \lamda v
            print("Eigenvalues:")
            print(eigvals)
            #print("Verify an eigenvector + eigenvalue")
            #print(np.isclose(np.dot(L,eigvects[:,1]),
            #                 eigvals[1]*eigvects[:,1]))
        return LX

###############################################################
#  CLASS: MORPHEME SPECTRAL CLUSTERING:
#    This inherits from the vanilla implementation, and is tweaked for working
#    with word/character embeddings

###############################################################
class WordEmbeddingsSpectralClustering(SpectralClustering):
    def make_affinity(self, X, sigma=None, algorithm="gaussian"):
        """
        X is assumed to be the matrix of embeddings.

        To compute the affinity matrix, we take the product of
        that matrix and its transpose, and run it through a gaussian
        """
        if algorithm == "gaussian":
            return self._make_gaussian(X, sigma)
        elif algorithm == "epsilon-neighborhood":
            return self._make_epsilon_neighborhood(X, sigma)
        elif algorithm == "k-nearest-neighbors":
            return self._make_k_nearest(X, sigma)

    def _make_gaussian(self, X, sigma=None):
        A = X.dot(X.T)
        if sigma == None:
            sigma = np.std(A)

        return np.exp(-A**2/(sigma**2))

    def _make_euclidean(self, X):
        A = np.zeros((len(X), len(X)))

        for i in range(len(X)-1):
            for j in range(i+1, len(X)):
                A[i,j] = np.linalg.norm(X[i]-X[j])
                A[j,i] = A[i,j]

        return A

    def _make_epsilon_neighborhood(self, X, sigma=None):
        def _find_epsilon(v):
            EPSILON = 0.9
            # Replace each with a 0 if it is less than the threshhold set by epsilon
            return np.array([1.0 if s < EPSILON else 0.0 for s in v])

        M = self._make_euclidean(X)
        print(M)

        print(np.apply_along_axis(_find_epsilon, 1, M))
        return np.apply_along_axis(_find_epsilon, 1, M)


    def make_eigenvector_matrix(self, A):
        # Sparse matrix of the diagonal
        D = np.diag(np.ravel(np.sum(A, axis=1)))
        # square root of the inverse of the sparse amtrix D
        Dinvsq = np.sqrt(np.linalg.inv(D))

        L = D-A
        L = Dinvsq.dot(L)
        L = L.dot(Dinvsq)

        # Find the K smallest eigenvectors of L
        eigvals, eigvects = scipy.sparse.linalg.eigs(L, k=self.num_clusters, which='SM')
        # ml = pyamg.smoothed_aggregation_solver(L, 'csr')
        # M = ml.aspreconditioner()
        # eigvals, eigvects = np.linalg.eigh(M)

        # lowest_eigs = eigvects[:, :self.num_clusters]
        # print("Found %i lowest Eigenvalues" % self.num_clusters)
        # print(eigvals[:self.num_clusters])

        LX = Dinvsq.dot(eigvects)
        LX = (LX.T / np.linalg.norm(LX, axis=1)).T
        self._LX = LX
        print(LX)
        return self._LX

    def fit(self, X):
        # Set up data autoparams if missing
        self.fit_params(X)
        # construct affinity matrix
        A = self.make_affinity(X, algorithm="gaussian")
        # Find the K largest eigenvectors of the Laplacian of A
        LX = self.make_eigenvector_matrix(A)
        # (4) -------
        # Finally, do clustering on reduced space using KMeans:
        km = KMeans(n_clusters=2, n_init=20)
        km.fit(LX)
        self._labels = km.labels_
        self.auto_eval_cluster()
        return self._labels

###############################################################
#  MAIN: Test on toy datasets and plot
###############################################################
def main():
    datasets = generate_datasets()
    X, y = datasets[0]

    print("Running spectral clustering")
    model = SpectralClustering(num_clusters=2, sigma_sq=0.01)
    y_pred = model.fit_predict(X)

    # plot results
    plots = []
    add_subplot(plots, X, y, "Ground Truth")
    add_subplot(plots, model._LX, y_pred,
                "Spectral clustering:\nembedded domain",
                plot_type='sc-circle')
    add_subplot(plots, X, y_pred,
                "Spectral clustering:\noriginal domain")
    plot_subplots(1, plots)

if __name__ == "__main__":
    main()

