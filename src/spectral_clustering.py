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
    def __init__(self, num_clusters, sigma_sq=0.1, debug=False):
        self.num_clusters = num_clusters
        self.sigma_sq = sigma_sq
        self._debug = debug
        self._L = None
        self._LX = None
        self._labels = None

    def gaus_affinity_kernel(self, x1, x2):
        # compute the affinity of samples X1, X2
        return exp(-(norm(x1-x2)**2)/(2*self.sigma_sq))

    def fit_predict(self, X):
        return self.fit(X)

    def fit(self, X):
        # (1) -------
        # construct affinity matrix
        A = np.zeros((len(X), len(X)))
        for i in range(len(X)-1):
            for j in range(i+1, len(X)):
                A[i,j] = self.gaus_affinity_kernel(X[i], X[j])
                A[j,i] = A[i,j]
        if self._debug:
            # print affinity matrix
            np.set_printoptions(precision=3)
            print(A)
        # (2) -------
        # Construct diagonal degree matrix
        D = np.zeros(A.shape)
        for i in range(A.shape[0]):
            D[i,i] = np.sum(A[i,:])
        # (3) -------
        # Construct Laplacian matrix
        L = self.make_laplacian(D, A)
        # (4) -------
        # Find the K largest eigenvectors of L
        LX = self.make_eigenvector_matrix(L)
        # (5) -------
        # Finally, do clustering on reduced space using KMeans:
        km = KMeans(n_clusters=2, n_init=20)
        km.fit(LX)
        self._labels = km.labels_
        return self._labels

    def make_laplacian(self, D, A):
        # Construct Laplacian Matrix:
        #   L = D^{-1/2} A D^{-1/2} -->
        #   L[i,j] = -A[i,j]/sqrt(d_i * d_j)
        Dinvsq = np.sqrt(np.linalg.inv(D))
        L = np.dot(Dinvsq, A)
        L = np.dot(L, Dinvsq)
        self._L = L
        #L = np.identity(len(A)) - L
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
        best_eigens = []
        for i in range(L.shape[0]-1,0,-1):
            if len(best_eigens) == self.num_clusters:
                break
            else:
                best_eigens.append(i)
        if self._debug:
            print("Best eigenvalues: {}".format(best_eigens))
        # TODO: verify orthogonal
        LX = np.zeros((L.shape[0], self.num_clusters))
        for i in range(self.num_clusters):
            LX[:,i] = eigvects[:,best_eigens[i]]
        # normalize new eigenvector-column-matrix
        for row in range(len(LX)):
            LX[row,:] = LX[row,:] / np.linalg.norm(LX[row,:])
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
#  CLASS: PROPER LAPLACIAN SPECTRAL CLUSTERING
#    The Ng-Jordan-Weiss algorithm changes the laplacian
#    slightly by not subtracting from the identity matrix.
#    They say this won't change the result, just the eigenvalues.
#    Here, just making sure. (CONFIRMED!)
###############################################################
class ProperLaplacianSpectralClustering(SpectralClustering):
    def make_laplacian(self, D, A):
        print("Building alternate laplacian: L = I - D^{-1/2} A D^{-1/2}")
        # Construct Laplacian Matrix:
        #   L = D^{-1/2} A D^{-1/2} -->
        #   L[i,j] = -A[i,j]/sqrt(d_i * d_j)
        Dinvsq = np.sqrt(np.linalg.inv(D))
        L = np.dot(Dinvsq, A)
        L = np.dot(L, Dinvsq)
        self._L = L
        L = np.identity(len(A)) - L
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
        best_eigens = []
        for i in range(L.shape[0]):
            if len(best_eigens) == self.num_clusters:
                break
            else:
                best_eigens.append(i)
        if self._debug:
            print("Best eigenvalues: {}".format(best_eigens))
        # TODO: verify orthogonal
        LX = np.zeros((L.shape[0], self.num_clusters))
        for i in range(self.num_clusters):
            LX[:,i] = eigvects[:,best_eigens[i]]
        # normalize new eigenvector-column-matrix
        for row in range(len(LX)):
            LX[row,:] = LX[row,:] / np.linalg.norm(LX[row,:])
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
#  MAIN: Test on toy datasets and plot
###############################################################
def main():
    datasets = generate_datasets()
    X, y = datasets[0]

    print("Fitting new model")
    model = ProperLaplacianSpectralClustering(num_clusters=2, sigma_sq=0.01)
    y_pred = model.fit_predict(X)
    plot_results(X, y_pred, 1, "alternate")

    print("Running spectral clustering")
    model = SpectralClustering(num_clusters=2, sigma_sq=0.01)
    y_pred = model.fit_predict(X)

    # plot results
    plots = []
    add_subplot(plots, X, y, "Ground Truth")
    add_subplot(plots, model._LX, y_pred, 
                "Spectral clustering:\nembedded domain",
                plot_type='sc-circle')
                #limits=([-1.2, 1.2],[-1.2,1.2]))
    add_subplot(plots, X, y_pred, 
                "Spectral clustering:\noriginal domain")
    plot_subplots(1, plots)

if __name__ == "__main__":
    main()

