# -*- coding: utf-8 -*-

"""
Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This script was built on the one provided in gensim: 
`gensim.scripts.make_wikicorpus`

"""

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, WikiCorpus, MmCorpus
from gensim import similarities
from gensim import utils
import time
import sys
import logging
import os


def formatTime(seconds):
    """
    Takes a number of elapsed seconds and returns a string in the format h:mm.
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d" % (h, m)
 

# ======== main ========
# Main entry point for the script.
# This little check has to do with the multiprocess module (which is used by
# WikiCorpus). Without it, the code will spawn infinite processes and hang!
if __name__ == '__main__':
    
    # Set up logging.
    
    # This little snippet is to fix an issue with qtconsole that you may or
    # may not have... Without this, I don't see any logs in Spyder.
    # Source: http://stackoverflow.com/questions/24259952/logging-module-does-not-print-in-ipython
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Create a logger
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    # Set the timestamp format to just hours, minutes, and seconds (no ms)
    #
    # Record the log to a file 'log.txt'--There is just under 5,000 lines of 
    # logging statements, so I've chosen to write these to a file instead of 
    # to the console. It's safe to have the log file open while the script is
    # running, so you can check progress that way if you'd like.
    logging.basicConfig(filename='log.txt', format='%(asctime)s : %(levelname)s : %(message)s', datefmt='%H:%M:%S')
    logging.root.setLevel(level=logging.INFO)
  
    # Download this file to get the latest wikipedia dump:
    # https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    # On Jan 18th, 2017 it was ~13GB
    dump_file = './data/enwiki-latest-pages-articles.xml.bz2'
    
    # ======== STEP 1: Build Dictionary =========            
    # The first step is to parse through all of Wikipedia and identify all of
    # the unique words that we want to have in our dictionary.   
    # This is a long process--it took 3.2hrs. on my Intel Core i7 4770
    if True:    

        # Create an empty dictionary
        dictionary = Dictionary()
        
        # Create the WikiCorpus object. This doesn't do any processing yet since
        # we've supplied the dictionary.
        wiki = WikiCorpus(dump_file, dictionary=dictionary) 
        
        print('Parsing Wikipedia to build Dictionary...')    
        sys.stdout.flush()
        
        t0 = time.time()

        # Now it's time to parse all of Wikipedia and build the dictionary.
        # This is a long process, 3.2hrs. on my Intel i7 4770. It will update
        # you at every 10,000 documents.
        #
        # wiki.get_texts() will only return articles which pass a couple 
        # filters that weed out stubs, redirects, etc. If you included all of
        # those, Wikpedia is more like ~17M articles.
        #
        # For each article, it's going to add the words in the article to the 
        # dictionary.
        # 
        # If you look inside add_documents, you'll see that it calls doc2bow--
        # this generates a bag of words vector, but we're not keeping it. The
        # dictionary isn't finalized until all of the articles have been
        # scanned, so we don't know the right mapping of words to ids yet.
        #
        # You can use the prune_at parameter to prevent the dictionary from
        # growing too large during this process, but I think it's interesting
        # to see the total count of unique tokens before pruning.
        dictionary.add_documents(wiki.get_texts(), prune_at=None)            
                        
        print('    Building dictionary took %s' % formatTime(time.time() - t0))
        print('    %d unique tokens before pruning.' % len(dictionary))
        sys.stdout.flush()
        
        keep_words = 100000    
    
        # The initial dictionary is huge (~8.75M words in my Wikipedia dump), 
        # so let's filter it down. We want to keep the words that are neither 
        # very rare or overly common. To do this, we will keep only words that 
        # exist within at least 20 articles, but not more than 10% of all 
        # documents. Finally, we'll also put a hard limit on the dictionary 
        # size and just keep the 'keep_words' most frequent works.
        wiki.dictionary.filter_extremes(no_below=20, no_above=0.1, keep_n=keep_words)
        
        # Write out the dictionary to disk.
        # For my run, this file is 769KB when compressed.
        # TODO -- This text format lets you peruse it, but you can
        # compress it better as binary...
        wiki.dictionary.save_as_text('./data/dictionary.txt.bz2')
    else:
        # Nothing to do here.
        print('')
    
    # ======== STEP 2: Convert Articles To Bag-of-words ========    
    # Now that we have our finalized dictionary, we can create bag-of-words
    # representations for the Wikipedia articles. This means taking another
    # pass over the Wikipedia dump!
    if True:
    
        # Load the dictionary if you're just running this section.
        dictionary = Dictionary.load_from_text('./data/dictionary.txt.bz2')
        wiki = WikiCorpus(dump_file, dictionary=dictionary)    
    
        # Turn on metadata so that wiki.get_texts() returns the article titles.
        wiki.metadata = True         
    
        print('\nConverting to bag of words...')
        sys.stdout.flush()
        
        t0 = time.time()
    
        # Generate bag-of-words vectors (term-document frequency matrix) and 
        # write these directly to disk.
        # On my machine, this took 3.53 hrs. 
        # By setting metadata = True, this will also record all of the article
        # titles into a separate pickle file, 'bow.mm.metadata.cpickle'
        MmCorpus.serialize('./data/bow.mm', wiki, metadata=True, progress_cnt=10000)
        
        print('    Conversion to bag-of-words took %s' % formatTime(time.time() - t0))
        sys.stdout.flush()

        # Load the article titles back
        id_to_titles = utils.unpickle('./data/bow.mm.metadata.cpickle')
    
        # Create the reverse mapping, from article title to index.
        titles_to_id = {}

        # For each article...
        for at in id_to_titles.items():
            # `at` is (index, (pageid, article_title))  e.g., (0, ('12', 'Anarchism'))
            # at[1][1] is the article title.
            # The pagied property is unused.
            titles_to_id[at[1][1]] = at[0]
        
        # Store the resulting map.
        utils.pickle(titles_to_id, './data/titles_to_id.pickle')

        # We're done with the article titles so free up their memory.
        del id_to_titles
        del titles_to_id
    
    
        # To clean up some memory, we can delete our original dictionary and 
        # wiki objects, and load back the dictionary directly from the file.
        del dictionary
        del wiki  
        
        # Load the dictionary back from disk.
        # (0.86sec on my machine loading from an SSD)
        dictionary = Dictionary.load_from_text('./data/dictionary.txt.bz2')
    
        # Load the bag-of-words vectors back from disk.
        # (0.8sec on my machine loading from an SSD)
        corpus_bow = MmCorpus('./data/bow.mm')    
    
    # If we previously completed this step, just load the pieces we need.
    else:
        print('\nLoading the bag-of-words corpus from disk.')
        # Load the bag-of-words vectors back from disk.
        # (0.8sec on my machine loading from an SSD)
        corpus_bow = MmCorpus('./data/bow.mm')    

    
    # ======== STEP 3: Learn tf-idf model ========
    # At this point, we're all done with the original Wikipedia text, and we 
    # just have our bag-of-words representation.
    # Now we can look at the word frequencies and document frequencies to 
    # build a tf-idf model which we'll use in the next step.
    if True:
        print('\nLearning tf-idf model from data...')
        t0 = time.time()
        
        # Build a Tfidf Model from the bag-of-words dataset.
        # This took 47 min. on my machine.
        # TODO - Why not normalize?
        model_tfidf = TfidfModel(corpus_bow, id2word=dictionary, normalize=False)

        print('    Building tf-idf model took %s' % formatTime(time.time() - t0))
        model_tfidf.save('./data/tfidf.tfidf_model')
    
    # If we previously completed this step, just load the pieces we need.
    else:
        print('\nLoading the tf-idf model from disk.')
        model_tfidf = TfidfModel.load('./data/tfidf.tfidf_model') 
        

    # ======== STEP 4: Convert articles to tf-idf ======== 
    # We've learned the word statistics and built a tf-idf model, now it's time
    # to apply it and convert the vectors to the tf-idf representation.
    if True:
        print('\nApplying tf-idf model to all vectors...')
        t0 = time.time()
        
        # Apply the tf-idf model to all of the vectors.
        # This took 1hr. and 40min. on my machine.
        # The resulting corpus file is large--17.9 GB for me.        
        MmCorpus.serialize('./data/corpus_tfidf.mm', model_tfidf[corpus_bow], progress_cnt=10000)
        
        print('    Applying tf-idf model took %s' % formatTime(time.time() - t0))
    else:
        # Nothing to do here.
        print('')

    # ======== STEP 5: Train LSI on the articles ========
    # Learn an LSI model from the tf-idf vectors.
    if True:
        
        # The number of topics to use.
        num_topics = 300
        
        # Load the tf-idf corpus back from disk.
        corpus_tfidf = MmCorpus('./data/corpus_tfidf.mm')        
        
        # Train LSI
        print('\nLearning LSI model from the tf-idf vectors...')
        t0 = time.time()
        
        # Build the LSI model
        # This took 2hrs. and 7min. on my machine.
        model_lsi = LsiModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)   
    
        print('    Building LSI model took %s' % formatTime(time.time() - t0))

        # Write out the LSI model to disk.
        # The LSI model is big but not as big as the corpus.
        # The largest piece is the projection matrix:
        #  100,000 words x 300 topics x 8-bytes per val x (1MB / 2^20 bytes) = ~229MB
        #  This is saved as `lsi.lsi_model.projection.u.npy` 
        model_lsi.save('./data/lsi.lsi_model')
    
    # If we previously completed this step, just load the pieces we need.
    else:
        # Load the tf-idf corpus and trained LSI model back from disk.
        corpus_tfidf = MmCorpus('./data/corpus_tfidf.mm')
        model_lsi = LsiModel.load('./data/lsi.lsi_model')
    
    # ========= STEP 6: Convert articles to LSI with index ========
    # Transform corpus to LSI space and index it
    if True:
        
        print('\nApplying LSI model to all vectors...')
        t0 = time.time()
        
        # You could apply Apply the LSI model to all of the tf-idf vectors and 
        # write them to disk as an MmCorpus, but this is huge--33.2GB.
        #MmCorpus.serialize('./data/corpus_lsi.mm', model_lsi[corpus_tfidf], progress_cnt=10000)    
                
        # Instead, we'll convert the vectors to LSI and store them as a dense
        # matrix, all in one step.     
        index = similarities.MatrixSimilarity(model_lsi[corpus_tfidf], num_features=num_topics)
        index.save('./data/lsi_index.mm')
        
        print('    Applying LSI model took %s' % formatTime(time.time() - t0))
