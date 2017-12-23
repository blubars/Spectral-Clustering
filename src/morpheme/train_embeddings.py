import codecs
from gensim.models.word2vec import Word2Vec

project_root = "/Users/ajwieme/Spectral-Clustering"

ROMANCE = ['romanian', 'french', 'latin', 'spanish', 'catalan', 'italian', 'portuguese']
GERMANIC = ['norwegian-bokmal', 'english', 'icelandic', 'norwegian-nynorsk', 'dutch', 'german', 'swedish', 'danish', 'faroese']
URALIC = ['hungarian', 'northern-sami', 'finnish', 'estonian']

def get_words(FN):
  #KEEP_ENGLISH = "abcdefghijklmnopqrstuvwxyz"
  KEEP_FINNISH = ['o', 'l', 'e', 'm', 'a', 'i', 'k', 'u', 'n', 't', 'v', 'r', 's', 'y', 'ä', 'j', 'ö', 'h', 'p', 'd', 'g', 'c', 'f', 'b', 'š', '-', '\xa0', 'w', 'z', 'é', 'x', ':', 'q', 'ž', 'å']
  # Get each in the tab delimited string on each line
  data = [l.strip().split('\t') for l in codecs.open(FN, 'r', 'utf-8') if l.strip() != '']
  # Let's use the lemma, and the inflected word form
  word_forms = [[c.lower() for c in w] for lemma, wf, tags in data for w in wf.split(' ') if len([c for c in w if c in KEEP_FINNISH]) == len(w)]
  lemmas = [[c.lower() for c in w] for lemma, wf, tags in data for w in lemma.split(' ') if len([c for c in w if c in KEEP_FINNISH]) == len(w)]

  return word_forms + lemmas


if __name__=='__main__':
  for lang in ROMANCE + GERMANIC + URALIC:
    print("training %s..." % lang)
    words = []
    for setting in ['low', 'medium', 'high']:
      fn = project_root + "/data/morphology/%s-train-%s" % (lang, setting)
      # Each have 300 dimensions and capture characters w/in 5 characters of the embedded one
      words += get_words(fn)

    print("training on %i words" % len(words))
    print(words)
    model = Word2Vec(words, size=300, window=1, min_count=1)
    print("saving %s..." % lang)
    model.save(project_root + "/character-embeddings/%s" % lang)
