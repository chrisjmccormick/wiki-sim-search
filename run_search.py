# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:10:20 2017

@author: Chris
"""
from gensim import similarities
from gensim import utils
import sys
import time

# Load the Wikipedia LSI vectors (if not already loaded)
if not globals().has_key('index'):
    print 'Loading Wikipedia LSI index...'
    sys.stdout.flush()
    index = similarities.MatrixSimilarity.load('./data/lsi_index.mm')


# Load the article titles. These have the format (pageid, article title)
if not (globals().has_key('id_to_titles') and globals().has_key('titles_to_id')):
    print 'Loading Wikipedia article titles'
    sys.stdout.flush()
    id_to_titles = utils.unpickle('./data/bow.mm.metadata.cpickle')
    titles_to_id = utils.unpickle('./data/titles_to_id.pickle')

query_title = 'Topic model'

print 'Searching for articles similar to \'' + query_title + '\':'

# Lookup the index of the query article.
query_id = titles_to_id[query_title]

# Select the row corresponding to the query vector.
# The .index property is a numpy.ndarray storing all of the LSI vectors,
# it's [~4.2M x 300].
query_vec = index.index[query_id, :]

t0 = time.time()

# Perform the similarity search!
sims = index[query_vec]

elapsed = time.time() - t0

print '    Similarity search took %.2f seconds' % elapsed

# Sort in descending order.
t0 = time.time()
sims = sorted(enumerate(sims), key=lambda item: -item[1])

elapsed = time.time() - t0

print '    Sorting took %.2f' % elapsed

print '\nResults:'

# Display the top 10 results
for i in range(0, 10):

    # Get the index of the result.
    result_index = sims[i][0]    
    
    print '    ' + id_to_titles[result_index][1]

