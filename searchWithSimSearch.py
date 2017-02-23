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
    Creates the SimSearch and KeySearch objects using the data structures
    created in `make_wikicorpus.py`.
    Returns (simsearch, keysearch, titles_to_id)
    """
    
    # Load the article titles. These have the format (pageid, article title)
    fprint('Loading Wikipedia article titles...')
    t0 = time.time()
    
    id_to_titles = utils.unpickle('./data/bow.mm.metadata.cpickle')
    titles_to_id = utils.unpickle('./data/titles_to_id.pickle')

    # id_to_titles is actually a map of indeces to (pageid, article title)
    # The 'pageid' property is unused.
    # Convert id_to_titles into a simple list of titles.
    titles = [item[1][1] for item in id_to_titles.items()]
    
    fprint('    Took %.2f seconds' % (time.time() - t0))        
    
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
    ksearch = KeySearch(dictionary, tfidf_model, corpus_tfidf, titles)
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

    # TODO - It would be interesting to try the 'Similarity' class which 
    #       shards the dataset on disk for you...

    return (simsearch, ksearch, titles_to_id)

# ======== Example 1 ========
# Searches for top 10 articles most similar to a query article.
# Interprets the top match by showing which words contributed most to the
# similarity.
def example1(simsearch, ksearch, titles_to_id):

    query_article = 'Topic model'
    
    fprint('\nSearching for similar articles...')
    t0 = time.time()
    
    # Search for the top 10 most similar Wikipedia articles to the query.
    # This takes about 12 seconds on my machine, mostly in the sorting step.
    results = simsearch.findSimilarToDoc(titles_to_id[query_article], topn=10)
    simsearch.printResultsByTitle(results)
    
    fprint('\nSearch and sort took %.2f seconds' % (time.time() - t0))    
    
    # Lookup the name of the top matching article.
    top_match_article = ksearch.titles[results[0][0]]
    
    fprint('\nInterpreting the match between \'' + query_article + '\' and \'' + top_match_article + '\' ...\n')
    t0 = time.time()
    
    # Get the tf-idf vectors for the two articles (the input and the top match).
    vec1_tfidf = ksearch.getTfidfForDoc(titles_to_id[query_article])
    vec2_tfidf = ksearch.getTfidfForDoc(results[0][0])
    
    # Interpret the top match match. Turn off filtering since the contributions
    # appear to be small with so many words.
    simsearch.interpretMatch(vec1_tfidf, vec2_tfidf, topn=20, min_pos=0, max_neg=-0.001)
    
    fprint('Interpreting match took %.2f seconds' % (time.time() - t0))    

# ======== Example 2 ========
# Use an example file as input to a search.
# For this example, I've supplied the markdown for a couple of my blog posts.
# TODO - Discuss results.
def example2(simsearch, ksearch, titles_to_id):
    
    fprint('\nSearching for articles similar to my word2vec tutorial...')
    t0 = time.time()

    # Get a tf-idf representation of my blog post.
    #input_tfidf = ksearch.getTfidfForFile('./data/2016-04-19-word2vec-tutorial-the-skip-gram-model.markdown')
    input_tfidf = ksearch.getTfidfForFile('./data/2014-08-04-gaussian-mixture-models-tutorial-and-matlab-code.markdown')
    
    # Search for Wikipedia articles similar to my word2vec blog post.
    results = simsearch.findSimilarToVector(input_tfidf)    
    
    # You can also search directly from the file.
    #results = simsearch.findSimilarToFile('./data/2016-04-19-word2vec-tutorial-the-skip-gram-model.markdown')
    
    simsearch.printResultsByTitle(results)

    topmatch_tfidf = ksearch.getTfidfForDoc(results[0][0])

    # Lookup the name of the top matching article.
    top_match_article = ksearch.titles[results[0][0]]
    
    fprint('\nInterpreting the match between my blog post and \'' + top_match_article + '\' ...\n')

    # Interpret the top match.
    simsearch.interpretMatch(input_tfidf, topmatch_tfidf, topn=10, min_pos=0, max_neg=-0.001)    
    
    fprint('\nSearch and sort took %.2f seconds' % (time.time() - t0))   

# ======== Example 3 ========
# Display and record the topic words.
def example3(simsearch, ksearch, titles_to_id):
    # Get the top 10 words for every topic.    
    # `topics` is a list of length 300.
    topics = simsearch.lsi.show_topics(num_topics=-1, num_words=10, log=False, formatted=False)
    
    with open('./topic_words.txt', 'wb') as f:
    
        # `topic_words` has the form (topic_id, topic_words)
        for topic_words in topics:
        
            # Put all the words into one line.
            topic_line = ''
            
            # `word` has the form (word, weight)
            for word in topic_words[1]:
                topic_line += word[0] + ', '
                
            # Print line.
            print topic_line
            
            # Write the topic to the text file.
            f.write(topic_line.encode('utf-8') + '\n')
    

# ======== main ========
# Entry point to the script.

# Load the corpus, model, etc.
# This takes about 15 second on my machine (I have an SSD), and requires at
# least 5GB of RAM.
simsearch, ksearch, titles_to_id = createSearchObjs()

# Search for articles similar to 'Topic model'
example1(simsearch, ksearch, titles_to_id)

# Search for articles similar to one of my blog posts.
#example2(simsearch, ksearch, titles_to_id)

# Display and record the top words for each topic.
#example3(simsearch, ksearch, titles_to_id)
