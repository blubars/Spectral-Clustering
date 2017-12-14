import codecs

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

import gensim
from gensim.models.word2vec import Word2Vec

from scipy.sparse import csr_matrix, csc_matrix, identity, linalg
from scipy.sparse.linalg import inv

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
  return wv.dot(wv.T)



if __name__=='__main__':
  num_clusters = 2
  model = Word2Vec.load(project_root + "/character-embeddings/english")

  wv = model[model.wv.vocab]
  vocab = model.wv.vocab.keys()

  km = KMeans(n_clusters=num_clusters, n_init=20)
  km.fit_predict(wv)

  tsne = TSNE(n_components=2, random_state=0)
  X = tsne.fit_transform(wv)

  plot_results(X, vocab, km.labels_, 1, "english characters")

  """
  Try Sklearn spectral clustering
  """
  A = get_affinity(vocab, wv)

  cluster = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', eigen_solver='amg')
  y_pred = cluster.fit_predict(A)

  plot_results(X, vocab, y_pred, 2, "english characters")

  """
  SAME EXPERIMENT WITH OUR SPECTRAL CLUSTERING ALGORITHM
  """
  A = get_affinity(vocab, wv)
  # Sparse matrix of the diagonal
  D = csc_matrix(np.diag(np.ravel(np.sum(A, axis=1))))

  # square root of the inverse of the sparse amtrix D
  Dinvsq = np.sqrt(inv(D))
  L = identity(D.shape[0]) - Dinvsq.dot(A)

  # Find the K largest eigenvectors of L
  # eigvals, eigvects = np.linalg.eigh(L)
  eigvals, eigvects = linalg.eigs(L, k=num_clusters)

  # Top k eigenvectors, normalized
  LX = eigvects[:, range(num_clusters)]
  LX = LX/np.linalg.norm(LX, axis=1).reshape(-1,1)

  print("Eigenvalues:")
  print(eigvals)

  km2 = KMeans(n_clusters=num_clusters, n_init=20)
  km2.fit(LX)
  plot_results(X, vocab, km2.labels_, 3, "english characters")
