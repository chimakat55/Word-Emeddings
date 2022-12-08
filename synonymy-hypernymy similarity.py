# -*- coding: utf-8 -*-
"""CompSem_A02_Ex03_Manisha_Kosuke.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1li44DgDgscs7qoayi7FqXCKlpYefGFh0

Step 1: Import necessary libraries
"""

import nltk
from nltk.corpus import wordnet 
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import random
from random import choice
import gensim

nltk.download('wordnet')
nltk.download('word2vec_sample')

path_w2v_sample = nltk.data.find('models/word2vec_sample/pruned.word2vec.txt')
model = gensim.models.KeyedVectors.load_word2vec_format(path_w2v_sample)

"""Step 2: Get a set of 500 synonym and antonyms of a word.
Calculate the similarities with the word.
"""

synonym_similarities = []
antonym_similarities = []

count = 0
all_words = [w for w in wordnet.words()]

while count < 500:
  word = choice(all_words)
  if word in model:
    synonyms = []
    antonyms = []

    for synset in wordnet.synsets(word):
      for lemma in synset.lemmas():
        if lemma.name() != word and lemma.name() in model and lemma.name() not in synonyms:
          synonyms.append(lemma.name())
        for antonym in lemma.antonyms():
          if antonym.name() in model:
            antonyms.append(antonym.name())
        if len(synonyms) == 1 and len(antonyms) == 1:
          break

    if len(synonyms) and len(antonyms):
      word_syn_similarities = [model.similarity(word, synonym) for synonym in synonyms]
      word_ant_similarities = [model.similarity(word, antonym) for antonym in antonyms]
      synonym_similarities.append(np.mean(word_syn_similarities))
      antonym_similarities.append(np.mean(word_ant_similarities))
      count += 1

"""Step 3: Get a set of 500 hypernym and hyponym of a word.
Calculate the similarities with the word.
"""

hypernym_similarities = []
hyponym_similarities = []

count = 0
#all_words = [w for w in wordnet.words()]

while count < 500:
  word = choice(all_words)
  if word in model:
    for synset in wordnet.synsets(word):
      hyponyms = []
      hypernyms = []

      for hyponym in synset.hyponyms():
        if hyponym.lemmas()[0].name() in model:
          hyponyms.append(hyponym.lemmas()[0].name())
        for hypernym in synset.hypernyms():
          if hypernym.lemmas()[0].name() in model:
            hypernyms.append(hypernym.lemmas()[0].name())
        if len(hyponyms) == 1 and len(hypernyms) == 1:
          break
		  
    if len(hypernyms) and len(hyponyms):
      word_hpr_similarities = [model.similarity(word, hypernym) for hypernym in hypernyms]
      word_hpo_similarities = [model.similarity(word, hyponym) for hyponym in hyponyms]
      hypernym_similarities.append(np.mean(word_hpr_similarities))
      hyponym_similarities.append(np.mean(word_hpo_similarities))
      count += 1

"""Qualitative results and graph to visualize the results."""

#Plot for cosine info
fig, ax = plt.subplots()
ax.boxplot([synonym_similarities,antonym_similarities,hypernym_similarities,hyponym_similarities], widths=0.5)
plt.xticks([1, 2, 3, 4], ['Synonyms', 'Antonyms', 'Hypernyms', 'Hyponyms'])
plt.xlabel("Distribution")
plt.ylabel("Similarity")

print("Synonym mean and std:", "{:.2f}".format(np.mean(synonym_similarities)*100),"{:.2f}".format(np.std(synonym_similarities)*100))
print("Antonym mean and std:", "{:.2f}".format(np.mean(antonym_similarities)*100),"{:.2f}".format(np.std(antonym_similarities)*100))

print("Hypernym mean and std:", "{:.2f}".format(np.mean(hypernym_similarities)*100),"{:.2f}".format(np.std(hypernym_similarities)*100))
print("Hyponym mean and std:", "{:.2f}".format(np.mean(hyponym_similarities)*100),"{:.2f}".format(np.std(hyponym_similarities)*100))