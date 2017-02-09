# -*- coding: utf-8 -*-

"""
Convert articles from a Wikipedia dump to (sparse) vectors. The input is a
bz2-compressed dump of Wikipedia articles, in XML format.

This script was built on the one provided in gensim: 
`gensim.scripts.make_wikicorpus`

This creates the following files:

* `wiki_wordids.txt.bz2`: mapping between words and their integer ids

"""

from gensim.models import TfidfModel
from gensim.corpora import MmCorpus
from gensim.corpora import Dictionary, WikiCorpus
import time
import sys

   

# ======== main ========
# Main entry point for the script.
# This little check has to do with the multiprocess module (which is used by
# WikiCorpus). Without it, the code will spawn infinite processes and hang!
if __name__ == '__main__':
  
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
        # TODO -- Why text and not other formats?
        wiki.dictionary.save_as_text('./data/dictionary.txt.bz2')
    
    # ======== STEP 2: Convert Articles To Bag-of-words ========    
    # Now that we have our finalized dictionary, we can create bag-of-words
    # representations for the Wikipedia articles. This means taking another
    # pass over the Wikipedia dump!
    if False:
    
        print 'Converting to bag of words...'
        t0 = time.time()
    
        # Generate bag-of-words vectors (term-document frequency matrix) and 
        # write these directly to disk.
        # On my machine, this took 3.53 hrs. 
        MmCorpus.serialize('./data/bow.mm', wiki, progress_cnt=10000)
        
        print 'Conversion to bag-of-words took %.2f hrs.' % ((time.time() - t0) / 3600)
    
        # To clean up some memory, we can delete our original dictionary and 
        # wiki objects, and load back the dictionary directly from the file.
        del dictionary
        del wiki  
        
        # Load the dictionary back from disk.
        # (0.86sec on my machine loading from an SSD)
        dictionary = Dictionary.load_from_text('dictionary.txt.bz2')
    
        # Load the bag-of-words vectors back from disk.
        # (0.8sec on my machine loading from an SSD)
        mm = MmCorpus('./data/bow.mm')    
        
    
    # ======== STEP 3: Learn tf-idf model ========
    # At this point, we're all done with the original Wikipedia text, and we 
    # just have our bag-of-words representation.
    # Now we can look at the word frequencies and document frequencies to 
    # build a tf-idf model which we'll use in the next step.
    if True:
        print 'Learning tf-idf model from data...'
        t0 = time.time()
        
        # Build a Tfidf Model from the bag-of-words dataset.
        tfidf = TfidfModel(mm, id2word=dictionary, normalize=True)
        tfidf.save('./data/tfidf.tfidf_model')

        print 'Building tf-idf model took %.2f min.' % ((time.time() - t0) / 60)

    # ======== STEP 4: Convert articles to tf-idf ======== 
    # We've learned the word statistics and built a tf-idf model, now it's time
    # to apply it and convert the vectors to the tf-idf representation.
    if True:
        print 'Applying tf-idf model to all vectors...'
        t0 = time.time()
        
        # save tfidf vectors in matrix market format
        # ~4h; result file is 15GB! bzip2'ed down to 4.5GB
        MmCorpus.serialize('./data/tfidf.mm', tfidf[mm], progress_cnt=10000)

        