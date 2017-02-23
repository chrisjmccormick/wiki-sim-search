# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:10:20 2017

@author: Chris
"""
from gensim import similarities
from gensim import utils
import time
import sys
import operator

# Load the Wikipedia LSI vectors.
# This matrix is large (4.69 GB for me) and takes ~15 seconds to load.
print 'Loading Wikipedia LSI index (15-30sec.)...'
t0 = time.time()

index = similarities.MatrixSimilarity.load('./data/lsi_index.mm')

print '   Loading LSI vectors took %.2f seconds' % (time.time() - t0)

# Load the article titles. These have the format (pageid, article title)
print '\nLoading Wikipedia article titles...'

id_to_titles = utils.unpickle('./data/bow.mm.metadata.cpickle')
titles_to_id = utils.unpickle('./data/titles_to_id.pickle')

# Name of the article to use as the input to the search.
query_title = 'Topic model'

print '\nSearching for articles similar to \'' + query_title + '\':'

# Lookup the index of the query article.
query_id = titles_to_id[query_title]

# Select the row corresponding to the query vector.
# The .index property is a numpy.ndarray storing all of the LSI vectors,
# it's [~4.2M x 300].
query_vec = index.index[query_id, :]

t0 = time.time()

# Perform the similarity search!
sims = index[query_vec]

print '    Similarity search took %.0f ms' % ((time.time() - t0) * 1000)

t0 = time.time()

# Sort in descending order.
# `sims` is of type numpy.ndarray, so the sort() method is different...
sims = sorted(enumerate(sims), key=lambda item: -item[1])

print '    Sorting took %.2f seconds' % (time.time() - t0)

print '\nResults:'

# Display the top 10 results
for i in range(0, 10):

    # Get the index of the result.
    result_index = sims[i][0]    
    
    print '    ' + id_to_titles[result_index][1]

