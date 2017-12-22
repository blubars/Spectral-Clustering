import codecs

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import cluster as skcluster
import matplotlib.pyplot as plt

import gensim
from gensim.models.word2vec import Word2Vec

from scipy.sparse import csr_matrix, csc_matrix, identity, linalg
from scipy.sparse.linalg import inv

# Hacky way to get Brian's class in this dir
import sys
sys.path.append("..")
from spectral_clustering import WordEmbeddingsSpectralClustering

project_root = "/Users/ajwieme/Spectral-Clustering"

ROMANCE = ['romanian', 'french', 'latin', 'spanish', 'catalan', 'italian', 'portuguese']
GERMANIC = ['norwegian-bokmal', 'english', 'icelandic', 'norwegian-nynorsk', 'dutch', 'german', 'swedish', 'danish', 'faroese']
URALIC = ['hungarian', 'northern-sami', 'finnish', 'estonian']

def plot_results(X, ann_labels, y, fignum, title):
    # plot with true labels
    fig = plt.figure(fignum)
    plt.title(title)
    colors = ['#377eb8', '#ff7f00', '#008000']
    y_colors = [colors[label] for label in y]

    # scatter plot with kmeans labels as colors
    plt.scatter(X[:,0], X[:,1], color=y_colors)
    plt.gca().set_aspect('equal')

    # Add the character labels to each point
    for label, x, y in zip(ann_labels, X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

def get_affinity(vocab, wv):
  """
  A = np.zeros([len(vocab), len(vocab)])
  for i, v1 in enumerate(vocab):
    for j, v2 in enumerate(vocab):
      if i == j:
        A[i, j] = 0.0
      else:
        A[i, j] = model.similarity(v1, v2)
  sigma = np.std(A)
  A = np.divide(A, sigma)

  return A
  """
  A = wv.dot(wv.T)
  sigma = np.std(A)
  return np.exp(-A**2/(sigma**2))
  #return np.divide(wv.dot(wv.T), sigma)



if __name__=='__main__':
  num_clusters = 2
  model = Word2Vec.load(project_root + "/character-embeddings/finnish")

  wv = model[model.wv.vocab]
  vocab = model.wv.vocab.keys()
  print(vocab)

  km = KMeans(n_clusters=num_clusters, n_init=20)
  #print(wv)
  km.fit_predict(wv)

  tsne = TSNE(n_components=2, random_state=0)
  X = tsne.fit_transform(wv)

  plot_results(X, vocab, km.labels_, 1, "Finnish Characters")

  """
  Try Sklearn spectral clustering
  """
  """
  A = get_affinity(vocab, wv)
  cluster = skcluster.SpectralClustering(n_clusters=num_clusters, affinity='precomputed', eigen_solver='amg', n_init=20)
  sk_spec = cluster.fit(A)

  plot_results(X, vocab, sk_spec.labels_, 2, "finnish characters")
  """

  """
  SAME EXPERIMENT WITH OUR SPECTRAL CLUSTERING ALGORITHM
  """
  gaussian_model = WordEmbeddingsSpectralClustering(num_clusters=2, sigma_sq=0.01)
  gaussian_model.fit_predict(wv)

  plot_results(X, vocab, gaussian_model._labels, 1, "Finnish Characters")

  """
  epsilon_neighborhood_model = WordEmbeddingsSpectralClustering(num_clusters=2, sigma_sq=0.01)
  # TEST EPSILONS
  for e in range(2, 20, 2):
    print("EPSILON: %.2f" % (e/10))
    epsilon_neighborhood_model.fit(wv, algorithm="epsilon-neighborhood", epsilon=e/10)
    if epsilon_neighborhood_model._labels is not None:
      plot_results(X, vocab, epsilon_neighborhood_model._labels, 2, "Finnish Characters")

  k_nearest_model = WordEmbeddingsSpectralClustering(num_clusters=2, sigma_sq=0.01)
  # TEST EPSILONS
  for k in range(2, 20, 2):
    print("K-NEAREST WITH k=%i" % k)
    k_nearest_model.fit(wv, algorithm="k-nearest-neighbors", k=k)

    plot_results(X, vocab, k_nearest_model._labels, 2, "Finnish Characters")
  """
