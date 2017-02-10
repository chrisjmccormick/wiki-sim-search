# -*- coding: utf-8 -*-

"""
Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This script was built on the one provided in gensim: 
`gensim.scripts.make_wikicorpus`

This creates the following files:

* `wiki_wordids.txt.bz2`: mapping between words and their integer ids

"""

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary, WikiCorpus, MmCorpus
from gensim import utils
import time
import sys
import logging
import os


   

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

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
  
    # Download this file to get the latest wikipedia dump:
    # https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
    # On Jan 18th, 2017 it was ~13GB
    dump_file = './data/enwiki-latest-pages-articles.xml.bz2'
    
    # ======== STEP 1: Build Dictionary =========            
    # The first step is to parse through all of Wikipedia and identify all of
    # the unique words that we want to have in our dictionary.   
    # This is a long process--it took 3.2hrs. on my Intel Core i7 4770
    if False:    

        # Create an empty dictionary
        dictionary = Dictionary()
    
        print 'Creating WikiCorpus'    
        
        # Create the WikiCorpus object. This doesn't do any processing yet since
        # we've supplied the dictionary.
        wiki = WikiCorpus(dump_file, dictionary=dictionary) 
        
        # Turn on metadata so that wiki.get_texts() returns the article titles.
        wiki.metadata = True         
        
        # Wiki is first scanned for all distinct word types (~7M). The types that
        # appear in more than 10% of articles are removed and from the rest, the
        # `keep_words` most frequent types are kept.
        keep_words = 100000
        
        # Now it's time to parse all of Wikipedia and build the dictionary.
        # This is a long process, 3.2hrs. on my Intel i7 4770
        
        # Maintain a list of the article titles.
        article_ids = []
        
        t0 = time.time()
        
        docnum = 0
        
        # For all ~5M articles in Wikipedia...
        # wiki.get_texts() will only return articles which pass a couple 
        # filters that weed out stubs, redirects, etc. If you included all of
        # those, Wikpedia is more like ~17M articles.
        for (tokens, (pageid, title)) in wiki.get_texts():
            
            # Store the article title.
            article_ids.append((pageid, title))
            
            # Update progress every 10,000 articles.
            if docnum % 10000 == 0:
                print "Adding document %8d   dictionary size: %8d" % (docnum, len(dictionary))
                sys.stdout.flush()  
                  #if len(dictionary) > 200000:
                  #      dictionary.filter_extremes(no_below=0, no_above=1.0, keep_n=prune_at)
                  
                    # I wasn't getting this printout, so I'm trying to force it...                
                    #
        
            # Add the words in the article to the dictionary.
            # Note that this also generates a bag of words vector, but we're 
            # not keeping it. The dictionary hasn't been finalized yet, so we 
            # don't know the right mapping of words to ids yet.
            dictionary.doc2bow(tokens, allow_update=True)        
            
            docnum += 1
            
                
        print 'Building dictionary took %.2f hrs.' % ((time.time() - t0) / 3600)
        print '%d unique tokens before pruning.' % len(dictionary)
    
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
        
        # Write out the article titles to disk using some utilities from 
        # gensim.
        with utils.smart_open('./data/articles.txt.bz2', 'wb') as fout:
            for pageid, title in article_ids:
                line = "%i\t%s\n" % (pageid, title)
                fout.write(utils.to_utf8(line))
    
    # ======== STEP 2: Convert Articles To Bag-of-words ========    
    # Now that we have our finalized dictionary, we can create bag-of-words
    # representations for the Wikipedia articles. This means taking another
    # pass over the Wikipedia dump!
    if True:
    
        # Load the dictionary if you're just running this section.
        dictionary = Dictionary.load_from_text('./data/dictionary.txt.bz2')
        wiki = WikiCorpus(dump_file, dictionary=dictionary)    
    
        print 'Converting to bag of words...'
        t0 = time.time()
    
        # Generate bag-of-words vectors (term-document frequency matrix) and 
        # write these directly to disk.
        # On my machine, this took 3.53 hrs. 
        MmCorpus.serialize('./data/bow.mm', wiki, metadata=True, progress_cnt=10000)
        
        print 'Conversion to bag-of-words took %.2f hrs.' % ((time.time() - t0) / 3600)
    
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
        
    
    # ======== STEP 3: Learn tf-idf model ========
    # At this point, we're all done with the original Wikipedia text, and we 
    # just have our bag-of-words representation.
    # Now we can look at the word frequencies and document frequencies to 
    # build a tf-idf model which we'll use in the next step.
    if True:
        print 'Learning tf-idf model from data...'
        t0 = time.time()
        
        # Build a Tfidf Model from the bag-of-words dataset.
        # This took 46.66 min. on my machine.
        model_tfidf = TfidfModel(corpus_bow, id2word=dictionary, normalize=True)

        print 'Building tf-idf model took %.2f min.' % ((time.time() - t0) / 60)
        model_tfidf.save('./data/tfidf.tfidf_model')

    # ======== STEP 4: Convert articles to tf-idf ======== 
    # We've learned the word statistics and built a tf-idf model, now it's time
    # to apply it and convert the vectors to the tf-idf representation.
    if True:
        print 'Applying tf-idf model to all vectors...'
        t0 = time.time()
        
        # Load the tfidf model back from disk if you need to
        # TODO if 'model_tfidf' doesn't exist.        
        
        # TODO ....
        # save tfidf vectors in matrix market format
        # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
        MmCorpus.serialize('./data/corpus_tfidf.mm', model_tfidf[corpus_bow], progress_cnt=10000)
        
        print 'Applying tf-idf model took %.2f hrs.' % ((time.time() - t0) / 3600)

    # ======== STEP 5: Train LSI on the articles ========
    # Learn an LSI model from the tf-idf vectors.
    if True:
        
        # The number of topics to use.
        num_topics = 300
        
        # Load the tf-idf corpus back from disk.
        corpus_tfidf = MmCorpus('./data/corpus_tfidf.mm')        
        
        # Train LSI
        # TODO - How long does it take?
        print 'Learning LSI model from the tf-idf vectors...'
        t0 = time.time()
        
        # Build the LSI model
        # TODO ...
        model_lsi = LsiModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary)   
    
        print 'Building LSI model took %.2f hrs.' % (time.time() - t0 / 3600)    

        # Write out the LSI model to disk.
        model_lsi.save('./data/lsi.lsi_model')
    
    # ========= STEP 6: Convert articles to LSI representation ========
    # Transform corpus to LSI space and index it
    if True:
        
        print 'Applying LSI model to all vectors...'        
        t0 = time.time()
        
        # Apply the LSI model to all of the tf-idf vectors and write them
        # to disk.
        MmCorpus.serialize('./data/corpus_lsi.mm', model_lsi[corpus_tfidf], progress_cnt=10000)    
        
        print 'Applying LSI model took %.2f hrs.' % ((time.time() - t0) / 3600)
    
    # ======== STEP 7: Create MatrixSimilarity index ========  
        #self.index = similarities.MatrixSimilarity(self.lsi[self.ksearch.corpus_tfidf], num_features=num_topics) 