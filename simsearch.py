# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 21:00:58 2016

@author: Chris
"""

from gensim.models import LsiModel
from gensim import similarities
from keysearch import KeySearch
import numpy as np

class SimSearch(object):
    """
    SimSearch allows you to search a collection of documents by providing 
    conceptually similar text as the search query, as opposed to the typical 
    keyword-based approach. This technique is also referred to as semantic 
    search or concept search.    
    """
    
    def __init__(self, key_search):
        """
        Initialize the SimSearch with a KeySearch object, which holds:
            - The dictionary
            - The tf-idf model and corpus
            - The document metadata.
        
        """        
        self.ksearch = key_search
           

    def trainLSI(self, num_topics=100):
        """
        Train the Latent Semantic Indexing model.
        """
        self.num_topics = num_topics        
        # Train LSA
        
        # Look-up the number of features in the tfidf model.
        #self.num_tfidf_features = max(self.corpus_tfidf.dfs) + 1        
        
        self.lsi = LsiModel(self.ksearch.corpus_tfidf, num_topics=self.num_topics, id2word=self.ksearch.dictionary)   
    
        # Transform corpus to LSI space and index it
        self.index = similarities.MatrixSimilarity(self.lsi[self.ksearch.corpus_tfidf], num_features=num_topics) 
    
    
    def findSimilarToVector(self, input_tfidf, topn=10, in_corpus=False):
        """
        Find documents in the corpus similar to the provided document, 
        represented by its tf-idf vector 'input_tfidf'.
        """
        
        # Find the most similar entries to the input tf-idf vector.
        #  1. Project it onto the LSI vector space.
        #  2. Compare the LSI vector to the entire collection.
        sims = self.index[self.lsi[input_tfidf]]        
        
        # Sort the similarities from largest to smallest.
        # 'sims' becomes a list of tuples of the form: 
        #    (doc_id, similarity_value)
        sims = sorted(enumerate(sims), key=lambda item: -item[1])   

        # Select just the top N results.
        # If the input vector exists in the corpus, skip the first one since
        # this will just be the document itself.
        if in_corpus:        
            # Select just the top N results, skipping the first one.
            results = sims[1:1 + topn]    
        else:
            results = sims[0:topn] 
                    
        return results
    
    def findSimilarToVectors(self, input_tfidfs, exclude_ids=[], topn=10):
        """
        Find documents similar to a collection of input vectors.        
        
        Combines the similarity scores from multiple query vectors.
        """
        # Calculate the combined similarities for all input vectors.
        sims_sum = []
        
        for input_vec in input_tfidfs:
            
            # Calculate the similarities between this and all other entries.
            sims = self.index[self.lsi[input_vec]]
        
            # Accumulate the similarities across all input vectors.
            if len(sims_sum) == 0:
                sims_sum = sims
            else:
                sims_sum = np.sum([sims, sims_sum], axis=0)
                    
        # Sort the combined similarities.
        sims_sum = sorted(enumerate(sims_sum), key=lambda item: -item[1])

        # Look through the results until we've gathered 'topn' results.
        results = []      
        for i in range(0, len(sims_sum)):
            
            # Get the doc id for this result.
            doc_id = sims_sum[i][0]    
        
            # If the result is not one of the inputs, and not in the exclude 
            # list, it's a valid result.
            if doc_id not in exclude_ids:
                results.append(sims_sum[i])

            # Break when we've gathered 'topn' results.
            if len(results) == topn:
                break
                
        return results
   
    
    def findSimilarToText(self, text, topn=10):
        """
        Find documents in the corpus similar to the provided input text.

        `text` should be a single string. It will be parsed, tokenized, and
        converted to a tf-idf vector by KeySearch.
        
        Returns the results as a list of tuples in the form:
            (doc_id, similarity_value)
        """
        # Parse the input text and create a tf-idf representation.        
        tfidf_vec = self.ksearch.getTfidfForText(text)
        
        # Pass the call down.        
        return self.findSimilarToVector(tfidf_vec, topn=topn, in_corpus=False)
    
    def findSimilarToFile(self, filename, topn=10):
        """
        Find documents in the corpus similar to the provided text file.
        
        `filename` should be a valid path to a file. The entire file will be
        read, parsed, tokenized, and converted to a vector.
        
        Returns the results as a list of tuples in the form:
            (doc_id, similarity_value)
        """

        # Convert the file to tf-idf.
        input_tfidf = self.ksearch.getTfidfForFile(filename)
    
        # Pass the call down.
        return self.findSimilarToVector(input_tfidf, topn)
    
    def findSimilarToDoc(self, doc_id, topn=10):
        """
        Find documents similar to the specified entry number in the corpus.
        
        This will not return the input document in the results list.
        
        Returns the results as a list of tuples in the form:
            (doc_id, similarity_value)
        """
        
        # Find the most similar entries to 'doc_id'
        #  1. Look up the tf-idf vector for the entry.
        #  2. Project it onto the LSI vector space.
        #  3. Compare the LSI vector to the entire collection.
        tfidf_vec = self.ksearch.corpus_tfidf[doc_id]
        
        # Pass the call down, specifying that the input is a part of the 
        # corpus.
        return self.findSimilarToVector(tfidf_vec, topn=topn, in_corpus=True)    
        
    
    def findMoreOfTag(self, tag, topn=10):
        """
        Find entries in the corpus which are similar to those tagged with 
        'tag'. That is, find more entries in the corpus that we might want to
        tag with 'tag'.

        """
        
        # All tags should be lower case to avoid mistakes.
        tag = tag.lower()        
        
        # I pre-pend a '!' to indicate that a document does not belong under
        # a specific tag (I do this to create negative samples)
        if ('!' + tag) in self.ksearch.tagsToEntries:
            exclude_ids = set(self.ksearch.tagsToEntries['!' + tag])
        else:
            exclude_ids = set()
        
        # Find all documents marked with 'tag'.
        input_ids = self.ksearch.tagsToEntries[tag]
        
        # Calculate the combined similarities for all input vectors.
        sims_sum = []
        
        for i in input_ids:
            # Get the LSI vector for this document.    
            input_vec = self.lsi[self.ksearch.corpus_tfidf[i]]
        
            print '  ' + self.ksearch.titles[i]
        
            # Calculate the similarities between this and all other entries.
            sims = self.index[input_vec]
        
            # Accumulate the similarities across all input vectors.
            if len(sims_sum) == 0:
                sims_sum = sims
            else:
                sims_sum = np.sum([sims, sims_sum], axis=0)
                    
        # Sort the combined similarities.
        sims_sum = sorted(enumerate(sims_sum), key=lambda item: -item[1])    
        
        # Append the input ids to the list of those to exclude from the results
        exclude_ids = exclude_ids + input_ids

        # Lookup the vectors for all of the input docs.        
        input_vecs = [self.ksearch.getTfidfForDoc(doc_id) for doc_id in input_ids]
        
        # Pass the call down.
        return self.findSimilarToVecs(input_vecs, exclude_ids=exclude_ids, topn=topn)
        
    def sparseToDense(self, sparse_vec, length):
        """
        Convert from a sparse vector representation to a dense vector. 
        
        A sparse vector is represented by a list of (index, value) tuples.
        A dense vector is a fixed length array of values.
        """        
        # Create an empty dense vector.
        vec = np.zeros(length)
        
        # Copy over the values into their correct positions.
        for i in range(0, len(sparse_vec)):
            j = sparse_vec[i][0]
            value = sparse_vec[i][1]
            vec[j] = value
        
        return vec


    def getSimilarityByWord(self, vec1_tfidf, vec2_tfidf):
        """
        Calculates the individual contribution of each word in document 1 to
        the total similarity between documents 1 and 2.
        
        Returns a list of tuples in the form:
            (word_id, sim_value)
        """    
        # Get the tf-idf and LSI vectors for the two documents, and convert them to
        # dense representations.        
        vec1_lsi = self.sparseToDense(self.lsi[vec1_tfidf], self.lsi.num_topics)
        vec2_lsi = self.sparseToDense(self.lsi[vec2_tfidf], self.lsi.num_topics)
        vec1_tfidf = self.sparseToDense(vec1_tfidf, self.ksearch.getVocabSize())
        #vec2_tfidf = self.sparseToDense(self.ksearch.corpus_tfidf[id2], self.ksearch.getVocabSize())    

        # Calculate the norms of the two LSI vectors.
        norms = np.linalg.norm(vec1_lsi) * np.linalg.norm(vec2_lsi)    
                
        # Create a vector to hold the similarity contribution of each word.
        word_sims = np.zeros(self.ksearch.getVocabSize())

        # For each word in the vocabulary...
        for word_id in range(0, self.ksearch.getVocabSize()):

            # Get the weights vector for this word. This vector has one weight
            # for each topic
            word_weights = np.asarray(self.lsi.projection.u[word_id, :]).flatten()

            # Calculate the contribution of this word in doc1 to the total similarity.
            word_sims[word_id] = vec1_tfidf[word_id] * np.dot(word_weights, vec2_lsi) / norms;
          
        # print 'Total word contributions:', np.sum(word_sims)  
        return word_sims

    def printWordSims(self, word_sims, topn, min_pos, max_neg):
        """
        Internal function used by `interpretMatch` to display the contributing
        words.
        """
        # TODO - First create the list of results in interpretMatch, then
        #        in this function just do the printing, and adapt the column
        #        width to the maximum word length in the results...
        
        # Build up the table of results to display.        
        tableStr = ''
        for i in range(0, topn):
            pos_word_id, pos_word_val = word_sims[i]
            neg_word_id, neg_word_val = word_sims[-(i + 1)]
            
            pos_word = self.ksearch.dictionary[pos_word_id]
            neg_word = self.ksearch.dictionary[neg_word_id]                       

            # If neither words pass the thresholds, break.
            if ((pos_word_val <= min_pos) and (neg_word_val >= max_neg)):
                break
            
            # Only display the positive word if the value passes the threshold.
            if (pos_word_val > min_pos):
                tableStr += '  %15s  +%.3f' % (pos_word, pos_word_val)
            # Otherwise add empty space.
            else:
                # e.g.,     '          freedom  +0.440'
                tableStr += '                         '
            
            # Only display the negative word if the value passes the threshold.
            if (neg_word_val < max_neg):
                tableStr += '    %15s  %.3f\n' % (neg_word, neg_word_val)
            # Otherwise just end the line.
            else:
                tableStr += '\n'
        
        print(tableStr)

    def interpretMatch(self, vec1_tfidf, vec2_tfidf, topn=10, min_pos=0.1, max_neg=-0.01):
        """
        Displays the words in each document which contribute the most 
        (positively or negatively) to the total similarity between the two
        specified documents.
        
        At most `topn` positive words and `topn` negative words will be
        displayed.
        
        Only words which contribute a positive similarity above `min_pos` are
        shown. Similarly, only words which contribute a negative similarity
        below `max_neg` are shown.
        
        """

        # Calculate the contribution of each word in doc 1 to the similarity.        
        word_sims = self.getSimilarityByWord(vec1_tfidf, vec2_tfidf)
        
        # Sort the similarities, biggest to smallest.    
        word_sims = sorted(enumerate(word_sims), key=lambda item: -item[1])

        print 'Words in doc 1 which contribute most to similarity:'
        self.printWordSims(word_sims, topn, min_pos, max_neg)

        # Calculate the contribution of each word in doc 2 to the similarity.
        word_sims = self.getSimilarityByWord(vec2_tfidf, vec1_tfidf)
        
        # Sort the similarities, biggest to smallest.    
        word_sims = sorted(enumerate(word_sims), key=lambda item: -item[1])

        print 'Words in doc 2 which contribute most to similarity:'
        self.printWordSims(word_sims, topn, min_pos, max_neg)
    

    def getTopWordsInCluster(self, doc_ids, topn=10):
        """
        Returns the most significant words in a specified group of documents.
        
        This is accomplished by summing together the tf-idf vectors for all the
        documents, then sorting the tf-idf values in descending order.
        """
        # Create a vector to hold the sum
        tfidf_sum = np.zeros(self.ksearch.getVocabSize())
        
        for doc_id in doc_ids:
            
            # Get the tf-idf vector for this document, and convert it to
            # its dense representation.
            vec_tfidf = self.ksearch.getTfidfForDoc(doc_id)
            vec_tfidf = self.sparseToDense(vec_tfidf, self.ksearch.getVocabSize())
            
            # Add the tf-idf vector to the sum.
            tfidf_sum += vec_tfidf

        # Sort the per-word tf-idf values, biggest to smallest.    
        word_ids = sorted(enumerate(tfidf_sum), key=lambda item: -item[1])
        
        # Create a list of the top words (as strings)
        top_words = []        
        for i in range(0, topn):
            word_id = word_ids[i][0]
            top_words.append(self.ksearch.dictionary[word_id])
            
        return top_words
        

    def printResultsByTitle(self, results):
        """
        Print the supplied list of search results in the format:
            [similarity]   [document title]
            [similarity]   [document title]
            ...
        """
        print 'Most similar documents:'
        for i in range(0, len(results)):
            # Print the similarity value followed by the entry title.            
            print '  %.2f    %s' % (results[i][1], self.ksearch.titles[results[i][0]])

    def printResultsByLineNumbers(self, results):
        """
        Print the supplied list of search results in the format:
            [similarity]   [source filename]  [line numbers]
            [similarity]   [source filename]  [line numbers]
            ...
        """        
        print 'Most similar documents:'
        for i in range(0, len(results)):
            # Print the similarity value followed by the source file and line
            # numbers.
            line_nums = self.ksearch.getDocLocation(results[i][0])
                
            print '  %.2f    %s  Lines: %d - %d' % (results[i][1], line_nums[0], line_nums[1], line_nums[2])
    
    def printResultsBySourceText(self, results, max_lines=10):
        """
        Print the supplied list of search results with their original source
        text.
        """
        print 'Most similar documents:\n'
        for i in range(0, len(results)):            
            # Print the similarity value followed by the source file and line
            # numbers.            
            line_nums = self.ksearch.getDocLocation(results[i][0])
                
            print '  %.2f    %s  Lines: %d - %d' % (results[i][1], line_nums[0], line_nums[1], line_nums[2])

            # Call down to the KeySearch to print out the doc.
            self.ksearch.printDocSourcePretty(results[i][0], max_lines)
            
            # Separate the results with a line.
            if len(results) > 1:
                print '\n'
                print '--------------------------------------------------------------------------------'
                print '\n'
                
    
    def save(self, save_dir='./'):
        """
        Save this SimSearch object to disk for later use.
        
        This also saves the underlying KeySearch object to disk.
        """

        # Save the LSI model and the LSI index.        
        self.index.save(save_dir + 'index.mm')
        self.lsi.save(save_dir + 'lsi.model')

        # Save the underlying KeySearch as well.        
        self.ksearch.save(save_dir)
        
    @classmethod
    def load(cls, save_dir='./'):
        """
        Load a SimSearch object and it's underlying KeySearch from the 
        specified directory. Returns both objects.
        """
        
        # First create and load the underlying KeySearch.
        ksearch = KeySearch.load(save_dir)
        
        # Create a SimSearch object.
        ssearch = SimSearch(ksearch)
        
        # Load the LSI index.
        ssearch.index = similarities.MatrixSimilarity.load(save_dir + 'index.mm')
        
        # Load the LSI model.
        ssearch.lsi = LsiModel.load(save_dir + 'lsi.model')
        
        return (ksearch, ssearch)
        