import codecs

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from gensim.models.word2vec import Word2Vec

project_root = "/Users/ajwieme/Spectral-Clustering"

ROMANCE = ['romanian', 'french', 'latin', 'spanish', 'catalan', 'italian', 'portuguese']
GERMANIC = ['norwegian-bokmal', 'english', 'icelandic', 'norwegian-nynorsk', 'dutch', 'german', 'swedish', 'danish', 'faroese']
URALIC = ['hungarian', 'northern-sami', 'finnish', 'estonian']

def plot_results(X, ann_labels, y, fignum, title):
    # plot with true labels
    fig = plt.figure(fignum)
    plt.title(title)
    colors = ['#377eb8', '#ff7f00']
    y_colors = [colors[label] for label in y]

    # scatter plot with kmeans labels as colors
    plt.scatter(X[:,0], X[:,1], color=y_colors)
    plt.gca().set_aspect('equal')

    # Add the character labels to each point
    for label, x, y in zip(ann_labels, X[:, 0], X[:, 1]):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.show()

if __name__=='__main__':
  model = Word2Vec.load(project_root + "/character-embeddings/english")

  wv = model[model.wv.vocab]
  vocab = model.wv.vocab.keys()

  tsne = TSNE(n_components=2, random_state=0)
  X = tsne.fit_transform(wv)

  plt.scatter(X[:, 0], X[:, 1])

  km = KMeans(n_clusters=2, n_init=20)
  km.fit_predict(X)

  plot_results(X, vocab, km.labels_, 1, "english characters")

  """
  SAME EXPERIMENT WITH OUR SPECTRAL CLUSTERING ALGORITHM
  """
