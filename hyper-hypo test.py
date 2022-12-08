# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:10:44 2022

@author: Manisha & Kosuke
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

## Step 2: Get a set of 500 synonym and antonyms of a word.
## Calculate the similarities with the word. ##

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
      
## Step 3: Get a set of 500 hypernym and hyponym of a word.
## Calculate the similarities with the word.

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
      
## Qualitative results and graph to visualize the results.

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

## 3. Analogies: solve them with the parallelogram model
## from BATS, dataset of hyponym analogies and dataset of hypernym analogies (BATS → Lexicographic Semantics → hypernymsmisc/hyponyms-misc). Use English embeddings. Use English embeddings. Obtain word pairs
## by taking the word in column 1 and the first word listed in column 2 (ignore the rest). In 
## your error analysis, make sure to address the following Q: Which kind of relationship is 
## modeled better?

def analogy_add (france, paris, italy):
    result = model.most_similar(positive=[italy,paris], negative=[france])
    return result[0][0]

def analogy_add_topn (france, paris, italy):
    result = model.most_similar(positive=[italy,paris], negative=[france])
    result_list = ([result[0][0],result[1][0],result[1][0],
                    result[2][0],result[3][0],result[4][0],
                    result[5][0],result[6][0],result[7][0],
                    result[8][0],result[9][0]])
    return result_list

def analogy_add_topn_num (france, paris, italy):
    result = model.most_similar(positive=[italy,paris], negative=[france])
    result_list = ([result[0][1],result[1][1],result[1][1],
                    result[2][1],result[3][1],result[4][1],
                    result[5][1],result[6][1],result[7][1],
                    result[8][1],result[9][1]])
    return result_list

def analogy_mul (france, paris, italy):
  result = model.most_similar_cosmul(positive=[italy,paris], negative=[france])
  return result[0][0]

#result = analogy_add_topn('France','Paris','Italy')
#print(result[0][0],result[1][0],result[2][0])
print(analogy_add_topn_num('France','Paris','Italy'))
#print(analogy_add_topn('France','Paris','Italy')[0][0],)
#print(analogy_mul('France','Paris','Italy'))

#read csv file
#make list of hyper and hypo indexed

from random import randrange

data_hyper = pd.read_csv('BATS_Hypernyms.csv')

result_prl = []
index_pair = []
index_target = []

result_topn = []
index_pair_topn = []
index_target_topn = []
index_sim = []

count = 0

while count < 1000:
  i = randrange(len(data_hyper['column 1']))
  j = randrange(len(data_hyper['column 1']))

  w_a = data_hyper['column 1'][i] 
  w_b = data_hyper['column 2'][i] 
  w_c = data_hyper['column 1'][j] 
  w_d = data_hyper['column 2'][j] 

  if (w_a in model.vocab) and (w_b in model.vocab) and (w_c in model.vocab) and (w_d in model.vocab):
    if analogy_add(w_a,w_b,w_c) == w_d:
      result_prl.append(1)
      index_pair.append(i)
      index_target.append(j)

    if w_d in analogy_add_topn(w_a,w_b,w_c):
      result_topn.append(1)
      index_pair_topn.append(i)
      index_target_topn.append(j)
      index_sim.append(np.std(analogy_add_topn_num(w_a,w_b,w_c)))


  count += 1

print('hypernym results:')
print(sum(result_prl)/1000)
print(sum(result_topn)/1000)

print('mean:',np.mean(index_sim),'std:',np.std(index_sim))

excl_list = []
for i in range(len(index_pair)):
#  print(index_pair[i])
  print('correct: ', data_hyper['column 1'][index_pair[i]],'to', 
      data_hyper['column 2'][index_pair[i]], 'analogous to', data_hyper['column 1'][index_target[i]],
      'to', data_hyper['column 2'][index_target[i]])
  if data_hyper['column 1'][index_pair[i]] not in excl_list:
    excl_list.append(data_hyper['column 1'][index_pair[i]])

print("length of exclusive list:",len(excl_list))

## Hypernym Parallelogram (Distributional Method)

from random import randrange

data_hypo = pd.read_csv('BATS_Hyponyms.csv')

result_prl = []
index_pair = []
index_target = []

result_topn = []
index_pair_topn = []
index_target_topn = []

index_sim = []

count = 0

while count < 1000:
  i = randrange(len(data_hypo['column 1']))
  j = randrange(len(data_hypo['column 1']))

  w_a = data_hypo['column 1'][i] 
  w_b = data_hypo['column 2'][i] 
  w_c = data_hypo['column 1'][j] 
  w_d = data_hypo['column 2'][j] 

  if (w_a in model.vocab) and (w_b in model.vocab) and (w_c in model.vocab) and (w_d in model.vocab):
    if analogy_add(w_a,w_b,w_c) == w_d:
      result_prl.append(1)
      index_pair.append(i)
      index_target.append(j)

    if w_d in analogy_add_topn(w_a,w_b,w_c):
      result_topn.append(1)
      index_pair_topn.append(i)
      index_target_topn.append(j)
      index_sim.append(np.std(analogy_add_topn_num(w_a,w_b,w_c)))


  count += 1

print('hyponym results:')
print(sum(result_prl)/1000)
print(sum(result_topn)/1000)

print('mean:',np.mean(index_sim),'std:',np.std(index_sim))

excl_list = []
for i in range(len(index_pair)):
#  print(index_pair[i])
  print('correct: ', data_hypo['column 1'][index_pair[i]],'to', 
      data_hypo['column 2'][index_pair[i]], 'analogous to', data_hypo['column 1'][index_target[i]],
      'to', data_hypo['column 2'][index_target[i]])
  if data_hypo['column 1'][index_pair[i]] not in excl_list:
    excl_list.append(data_hypo['column 1'][index_pair[i]])

print("length of exclusive list:",len(excl_list))

## Hyponym Parallelogram 

#opposite hyper

#read csv file
#make list of hyper and hypo indexed

from random import randrange

data_hyper = pd.read_csv('BATS_Hypernyms.csv')

result_prl = []
index_pair = []
index_target = []

result_topn = []
index_pair_topn = []
index_target_topn = []
index_sim = []

count = 0

while count < 1000:
  i = randrange(len(data_hyper['column 1']))
  j = randrange(len(data_hyper['column 1']))

  w_a = data_hyper['column 1'][i] 
  w_b = data_hyper['column 2'][i] 
  w_c = data_hyper['column 1'][j] 
  w_d = data_hyper['column 2'][j] 

  if (w_a in model.vocab) and (w_b in model.vocab) and (w_c in model.vocab) and (w_d in model.vocab):
    if analogy_add(w_b,w_a,w_d) == w_c:
      result_prl.append(1)
      index_pair.append(i)
      index_target.append(j)

    if w_c in analogy_add_topn(w_b,w_a,w_d):
      result_topn.append(1)
      index_pair_topn.append(i)
      index_target_topn.append(j)
      index_sim.append(np.std(analogy_add_topn_num(w_b,w_a,w_d)))


  count += 1

print('reverse hypernym results:')
print(sum(result_prl)/1000)
print(sum(result_topn)/1000)

print('mean:',np.mean(index_sim),'std:',np.std(index_sim))

for i in range(len(index_pair)):
#  print(index_pair[i])
  print('correct: ', data_hyper['column 1'][index_pair[i]],'to', 
      data_hyper['column 2'][index_pair[i]], 'analogous to', data_hyper['column 1'][index_target[i]],
      'to', data_hyper['column 2'][index_target[i]])

#opposite hypo

from random import randrange

data_hypo = pd.read_csv('BATS_Hyponyms.csv')

result_prl = []
index_pair = []
index_target = []

result_topn = []
index_pair_topn = []
index_target_topn = []

index_sim = []

count = 0

while count < 1000:
  i = randrange(len(data_hypo['column 1']))
  j = randrange(len(data_hypo['column 1']))

  w_a = data_hypo['column 1'][i] 
  w_b = data_hypo['column 2'][i] 
  w_c = data_hypo['column 1'][j] 
  w_d = data_hypo['column 2'][j] 

  if (w_a in model.vocab) and (w_b in model.vocab) and (w_c in model.vocab) and (w_d in model.vocab):
    if analogy_add(w_b,w_a,w_d) == w_c:
      result_prl.append(1)
      index_pair.append(i)
      index_target.append(j)

    if w_c in analogy_add_topn(w_b,w_a,w_d):
      result_topn.append(1)
      index_pair_topn.append(i)
      index_target_topn.append(j)
      index_sim.append(np.std(analogy_add_topn_num(w_a,w_b,w_c)))


  count += 1

print('reverse hyponym results:')
print(sum(result_prl)/1000)
print(sum(result_topn)/1000)

print('mean:',np.mean(index_sim),'std:',np.std(index_sim))

for i in range(len(index_pair)):
#  print(index_pair[i])
  print('correct: ', data_hypo['column 1'][index_pair[i]],'to', 
      data_hypo['column 2'][index_pair[i]], 'analogous to', data_hypo['column 1'][index_target[i]],
      'to', data_hypo['column 2'][index_target[i]])


