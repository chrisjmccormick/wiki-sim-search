# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:19:43 2017

@author: Chris
"""

from simsearch import SimSearch
from keysearch import KeySearch

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, MmCorpus
from gensim.similarities import MatrixSimilarity
from gensim import utils

import time

import sys

def fprint(msg):
    """
    Print function with stdout flush to force print statements to show.
    """
    print(msg)
    sys.stdout.flush()

def createSearchObjs():
    """
    Creates the SimSearch and KeySearch objects.    
    Returns (simsearch, keysearch, titles_to_id)
    """
    
    # Load the article titles. These have the format (pageid, article title)
    fprint('Loading Wikipedia article titles')
    sys.stdout.flush()
    id_to_titles = utils.unpickle('./data/bow.mm.metadata.cpickle')
    titles_to_id = utils.unpickle('./data/titles_to_id.pickle')
    
    # Load the dictionary (830ms on my machine)
    fprint('\nLoading dictionary...')
    t0 = time.time()
    
    dictionary = Dictionary.load_from_text('./data/dictionary.txt.bz2')
    
    fprint('    Took %.2f seconds' % (time.time() - t0))    
    
    # Load tf-idf model (60ms on my machine).
    fprint('\nLoading tf-idf model...')
    t0 = time.time()
    
    tfidf_model = TfidfModel.load('./data/tfidf.tfidf_model')    
    
    fprint('    Took %.2f seconds' % (time.time() - t0))        
    
    # We must not use `load`--that would attempt to load the corpus into 
    # memory, and it's 16.7 GB!!
    #corpus_tfidf = MmCorpus.load('./data/corpus_tfidf.mm')
    
    fprint('\nCreating tf-idf corpus object (leaves the vectors on disk)...')
    t0 = time.time()
    
    corpus_tfidf = MmCorpus('./data/corpus_tfidf.mm')
    
    fprint('    Took %.2f seconds' % (time.time() - t0))            
    
    # Create the KeySearch and SimSearch objects.    
    ksearch = KeySearch(dictionary, tfidf_model, corpus_tfidf, id_to_titles)
    simsearch = SimSearch(ksearch)
    
    # TODO - SimSearch doesn't currently have a clean way to provide the index
    # and model.
    
    fprint('\nLoading LSI model...')
    t0 = time.time()    
    simsearch.lsi = LsiModel.load('./data/lsi.lsi_model')
    
    fprint('    Took %.2f seconds' % (time.time() - t0))        
    
    # Load the Wikipedia LSI vectors into memory.
    # The matrix is 4.69GB for me, and takes ~15 seconds on my machine to load.
    fprint('\nLoading Wikipedia LSI index...')
    t0 = time.time()
        
    simsearch.index = MatrixSimilarity.load('./data/lsi_index.mm')
    
    fprint('    Took %.2f seconds' % (time.time() - t0))    

    # TODO - It would be interesting to go straight to 'Similarity' which shards the dataset for you...

    return (simsearch, ksearch, titles_to_id)

# ======== main ========
# Entry point to the script.

# Load the corpus, model, etc.
# This takes about 15 second on my machine (I have an SSD), and requires at
# least 5GB of RAM.
simsearch, ksearch, titles_to_id = createSearchObjs()

# ======== Example 1 ========
# Searches for top 10 articles most similar to a query article.
# Interprets the top match by showing which words contributed most to the
# similarity.

query_article = 'Topic model'

fprint('\nSearching for similar articles...')
t0 = time.time()

# Search for the top 10 most similar Wikipedia articles to the query.
# This takes about 12 seconds on my machine, mostly in the sorting step.
results = simsearch.findSimilarToDoc(titles_to_id[query_article], topn=10)
simsearch.printResultsByTitle(results)

fprint('\nSearch and sort took %.2f seconds' % (time.time() - t0))    

# Lookup the name of the top matching article.
top_match_article = ksearch.titles[results[0][0]][1]

fprint('\nInterpreting the match between \'' + query_article + '\' and \'' + top_match_article + '\' ...\n')
t0 = time.time()

# Get the tf-idf vectors for the two articles (the input and the top match).
vec1_tfidf = ksearch.getTfidfForDoc(titles_to_id[query_article])
vec2_tfidf = ksearch.getTfidfForDoc(results[0][0])

# Interpret the top match match. Turn off filtering since the contributions
# appear to be small with so many words.
simsearch.interpretMatch(vec1_tfidf, vec2_tfidf, topn=20, min_pos=0, max_neg=-0.001)

fprint('Interpreting match took %.2f seconds' % (time.time() - t0))    
