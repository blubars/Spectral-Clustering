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
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.linalg as sla
import timeit
import random

SUBPLOT_COLS = 4
SEED = 212
random.seed(SEED)

plot_colors = ['#377eb8', '#ff7f00', '#4daf4a']

###############################################################
#  UTILITY FUNCTIONS
###############################################################
# create toy datasets for cluster experiments: 
# this function was modified from sklearn.
def generate_datasets(num_samples=500):
    noisy_circles = datasets.make_circles(
            n_samples=num_samples, factor=0.5, noise=0.05)
    noisy_moons = datasets.make_moons(
            n_samples=num_samples, noise=0.05)
    random_state = 170
    X, y = datasets.make_blobs(n_samples=num_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    aniso = (X_aniso, y)
    varied = datasets.make_blobs(n_samples=num_samples,
                                 cluster_std=[1.0, 2.5, 0.5],
                                 random_state=random_state)
    data = [noisy_circles, noisy_moons, aniso, varied]
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

def add_subplot(subplot_list, X, y, title, plot_type=None, limits=None, accuracy=None):
    # plot types: None, 'sc-circle'
    subplot_list.append((X, y, title, plot_type, limits))

def plot_subplots(fignum, subplot_list):
    # make subplots and display
    num_plots = len(subplot_list)
    num_rows = num_plots // SUBPLOT_COLS
    if num_plots % SUBPLOT_COLS != 0:
        num_rows += 1
    if num_plots < SUBPLOT_COLS:
        num_cols = num_plots
    else:
        num_cols = SUBPLOT_COLS
    plot_shape = (2*num_cols+2,1.8*num_rows+2)
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

def calc_class_accuracy(Y, Y_pred, label1, label2):
    #print("finding acc btw {} and {}".format(label1, label2))
    correct = 0
    total = 0
    filt = np.where(Y == label1)
    preds = Y_pred[filt]
    for p in preds:
        if p == label2:
            correct += 1
        total += 1
    acc = correct / total
    #print("Accuracy ({},{}): {}".format(label1, label2, acc))
    return acc

def calc_align_accuracy(Y, Y_pred):
    # automatically find label alignments and accuracy:
    # try every alignment of label :(
    labels = np.unique(Y)
    aligns = []
    accuracies = []
    for i,l1 in enumerate(labels):
        aligns.append([])
        for l2 in labels:
            aligns[i].append((l1,l2))
    for i, row in enumerate(aligns):
        if i == len(aligns) - 1:
            # don't check the last row, should be only one
            break
        found = False
        for j, algn in enumerate(row):
            if found:
                aligns[i][j] = None
            elif algn:
                acc = calc_class_accuracy(Y, Y_pred, algn[0], algn[1])
                if acc < 0.5:
                    # bad alignment, hopefully. or just awful accuracy
                    aligns[i][j] = None
                else:
                    # found it! remove possibility from other rows
                    found = True
                    foundlabel = algn[1]
                    for k,l3 in enumerate(labels):
                        if l3 == algn[0]:
                            continue
                        else:
                            aligns[k][foundlabel] = None
                    # now set rest of possibilities here to false
    # return list of alignments
    final_aligns = []
    align_dict = {}
    for row in aligns:
        for a in row:
            if a is not None:
                align_dict[a[1]] = a[0]
                final_aligns.append(a)
    # get overall acc w/ dict mapping
    return calc_accuracy(Y, Y_pred, align_dict)

def calc_accuracy(Y, Y_pred, mapping=None):
    if mapping:
        mapped = np.copy(Y_pred)
        for i,label in enumerate(mapped):
            mapped[i] = mapping[label]
    else:
        mapped = Y_pred
    correct = 0
    # try combos of labels
    for y, y_pred in zip(Y, mapped):
        if y == y_pred:
            correct += 1
    return correct / len(Y)

def cluster_and_plot_dataset(X, y, n_clusters, plots, sigma_sq=None):
    # run spectral clustering, KMeans, and plot
    print("Running spectral clustering, original")
    model = SpectralClustering(num_clusters=n_clusters, sigma_sq=sigma_sq)
    y_pred = model.fit_predict(X)
    acc_sc = calc_align_accuracy(y, y_pred)
    print("SC Accuracy: {}".format(acc_sc))
    km = KMeans(n_clusters=n_clusters, n_init=10)
    km.fit(X)
    acc_km = calc_align_accuracy(y, km.labels_)
    print("KM Accuracy: {}".format(acc_km))
    add_subplot(plots, X, y, "Ground Truth")
    add_subplot(plots, X, km.labels_, "KMeans")
    add_subplot(plots, model._LX, y_pred, 
                "Spectral clustering:\nembedded domain",
                plot_type='sc-circle')
    add_subplot(plots, X, y_pred, 
                "Spectral clustering:\noriginal domain")

# alternate main to run profiling of spectral clustering
def run_profiling(fast=False, percent=0.2):
    n_samps = [100, 500, 1000, 2000]
    res = []
    for n in n_samps:
        print("Profiling spectral clustering, dataset size={}".format(n))
        datasets = generate_datasets(n)
        X, y = datasets[0]
        model = SpectralClustering(num_clusters=2, sigma_sq=0.01)
        #y_pred = model.fit_predict(X)
        res.append(model.profile(X, knn=fast, y=y, percent=percent))
    print("(n_samples, time)")
    for n, t in zip(n_samps, res):
        print("({}, {})".format(n, t))

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
        L, Dinvsq = self.make_laplacian(A)
        # (3) -------
        # Find the K largest eigenvectors of L
        LX = self.make_eigenvector_matrix(L, Dinvsq)
        # (4) -------
        # Finally, do clustering on reduced space using KMeans:
        self._labels = self.partition_embedding(LX)
        self.auto_eval_cluster()
        return self._labels

    def fit_predict_fast(self, X, n_neighbors=3, ratio=0.2):
        # try to speed up clustering by only running
        # spectral clustering on part of the dataset, then
        # using knn for the rest
        splitx = round(ratio*len(X))
        Xs = X[0:splitx]
        Xk = X[splitx:]
        ys = self.fit_predict(Xs)
        yk = self.knn_predict(Xs, ys, Xk)
        return np.append(ys, yk)

    # try to classify other points using KNN
    def knn_predict(self, X, y, X_pred, n_neighbors=3):
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X, y)
        return knn.predict(X_pred)

    def partition_embedding(self, LX):
        # do clustering on reduced space using KMeans:
        km = KMeans(n_clusters=self.num_clusters, n_init=10)
        km.fit(LX)
        labels = km.labels_
        return labels

    def auto_eval_cluster(self):
        # 'cluster quality' can be approximated by tightness of
        # clustering in transformed eigenspace.
        stdlist = []
        for c in range(self.num_clusters):
            std = np.std(self._LX[self._labels == c], axis=0)
            stdlist.append(np.average(std))
        #print("Std dev of clusters: {}".format(stdlist))
        if np.average(stdlist) > 0.1:
            print("High std deviation in embedded space, maybe try smaller sigma")


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
        return L, Dinvsq

    def make_eigenvector_matrix(self, L, Dinvsq):
        # find eigenvalues/eigenvectors, pick best ones
        # to use for spectral clustering, construct matrix
        # out of eigenvectors, normalize, and return
        #eigvals, eigvects = np.linalg.eigh(L)
        eigvals, eigvects = sla.eigh(L, check_finite=False, eigvals=(0,self.num_clusters))
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
        if self._debug:
            print("Best eigenvects, no normalization:")
            print(LX)
        # transform eigenvects to indicator matrix H
        LX = Dinvsq.dot(LX)
        norms = np.linalg.norm(LX, axis=1)
        bad_norms = np.where(norms == 0)
        norms[bad_norms] = 1
        LX = (LX.T / norms).T
        self._LX = LX
        if self._debug:
            print("Normalized best eigenvects:")
            print(LX)
            print("Eigenvalues:")
            print(eigvals)
            # verify: L v = \lamda v
            #print("Verify an eigenvector + eigenvalue")
            #print(np.isclose(np.dot(L,eigvects[:,1]), 
            #                 eigvals[1]*eigvects[:,1]))
        return LX

    def profile(self, X, y=None, repeat=8, knn=False, percent=0.2):
        t = timeit.default_timer
        times = []
        # profile each step, returning fastest run from each step
        print("Profiling({}x)... ".format(repeat), end=' ')
        for i in range(repeat):
            print("{}".format(i), end=' ', flush=True)
            if knn:
                t_start = t()
                yp = self.fit_predict_fast(X, ratio=percent)
                t_end = t()
            else:
                t_start = t()
                yp = self.fit_predict(X)
                t_end = t()
            times.append(t_end-t_start)
            # evaluate accuracy
            if y is not None:
                acc = calc_align_accuracy(y, yp)
                #print("Accuracy: {}".format(acc))
                if (acc < 0.6):
                    print("x", end=' ')
                    if i == 0:
                        plot_results(X, yp, 1, "")
        times.sort()
        print("\nBest time: {}".format(times[0]))
        return times[0]

###############################################################
#  MAIN: Test on toy datasets and plot
###############################################################
def main():
    datasets = generate_datasets(200)
    plots = []
    X, y = datasets[0]
    cluster_and_plot_dataset(X, y, 2, plots, 0.015)
    X, y = datasets[1]
    cluster_and_plot_dataset(X, y, 2, plots, 0.02)
    X, y = datasets[3]
    cluster_and_plot_dataset(X, y, 3, plots)
    plot_subplots(1, plots)

if __name__ == "__main__":
    #run_profiling(True, percent=0.75)
    main()
# end
