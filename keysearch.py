# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 11:31:44 2016

@author: Chris
"""

import textwrap
import pickle
import nltk
from gensim import corpora
from gensim.models import TfidfModel


# I lazily made this a global constant so that I wouldn't have to include
# it in the save and load features.
enc_format='utf-8'

class KeySearch(object):
    """
    KeySearch, which is short for "keyword search" stores a completed gensim 
    tf-idf corpus. Whereas SimSearch stores an LSI model, and only understands
    conceptual relationships between documents, KeySearch actually knows what
    words occur in each document.  

    It has several key functions:
      1. It has functions for converting new text sources (that is, texts not
         already in the corpus) into tf-idf vectors.
      2. It stores the corpus vocabulary in the form of a gensim dictionary.
      3. It supports boolean keyword search (though this is NOT indexed!).
      4. It stores the document metadata:
           - Title
           - Text source file
           - Line numbers in source file
           - Tags
    
    
    Saving & Loading
    ================
    The KeySearch object can be saved to and loaded from a directory
    using `save` and `load`. The typical useage, however is to simply save and
    load the SimSearch object (which also saves the underlying KeySearch).
    
    When saving the KeySearch, only the dictionary, feature vectors, and
    and document metadata are saved. The original text is not saved in any
    form.

    """
    def __init__(self, dictionary, tfidf_model, corpus_tfidf, titles, 
                  tagsToDocs={}, docsToTags={}, files=[], doc_line_nums=[]):
        """
        KeySearch requires a completed gensim corpus, along with some 
        additional metadata
        
        Parameters:
            dictionary - gensim dictionary
            tfidf_model - gensim TfidfModel
            corpus_tfidf - gensim corpora
            titles - List of string titles.
            tagsToDocs - Mapping of tags to doc ids
            docsToTags - List of tags for each doc
            files - Unique files in the corpus
            doc_line_nums - 
        """
        self.dictionary = dictionary
        self.tfidf_model = tfidf_model
        self.corpus_tfidf = corpus_tfidf
        
        self.titles = titles
    
        # Create mappings for the entry tags.
        self.tagsToDocs = tagsToDocs
        self.docsToTags = docsToTags

        self.files = files
        self.doc_line_nums = doc_line_nums
    
    def printTags(self):
        """
        Print all of the tags present in the corpus, plus the number of docs
        tagged with each.        
        """
        print 'All tags in corpus (# of documents):'
        
        # Get all the tags and sort them alphabetically.
        tags = self.tagsToDocs.keys()
        tags.sort()        

        # Print each tag followed by the number of documents.
        for tag in tags:
            print '%20s %3d' % (tag, len(self.tagsToDocs[tag]))       
        
    def getTfidfForText(self, text):
        """
        This function takes new input `text` (not part of the original corpus),
        and processes it into a tf-idf vector.
        
        The input text should be a single string.
        """
        # If the string is not already unicode, decode the string into unicode
        # so the NLTK can handle it.
        if isinstance(text, str):
            try:    
                text = text.decode(enc_format)        
            except:
                print '======== Failed to decode input text! ========'
                print 'Make sure text is encoded in', enc_format
                print 'Input text:'
                print text
                return []
        
        # If the string ends in a newline, remove it.
        text = text.replace('\n', ' ')

        # Convert everything to lowercase, then use NLTK to tokenize.
        tokens = nltk.word_tokenize(text.lower())

        # We don't need to do any special filtering of tokens here (stopwords, 
        # infrequent words, etc.). If a token is not in the dictionary, it is 
        # simply ignored. So the dictionary effectively does the token 
        # filtering for us.

        # Convert the tokenized text into a bag of words representation.
        bow_vec = self.dictionary.doc2bow(tokens) 
        
        # Convert the bag-of-words representation to tf-idf
        return self.tfidf_model[bow_vec]
    
    def getTfidfForFile(self, filename):
        """
        Convert the text in the provided file to a tf-idf vector.
        """
        # Open the file and read all lines.        
        with open(filename) as f:
            text = f.readlines()

        # Combine the lines into a single string.
        text = " ".join(text)

        # Pass the text down.
        return self.getTfidfForText(text)
    
    def getTfidfForDoc(self, doc_id):
        """
        Return the tf-idf vector for the specified document.
        """        
        return self.corpus_tfidf[doc_id]

    def keywordSearch(self, includes=[], excludes=[], docs=[]):
        """
        Performs a boolean keyword search over the corpus.
        
        All words in the dictionary are lower case. This function will convert
        all supplied keywords to lower case.
        
        Parameters:
            includes    A list of words (as strings) that the documents 
                        *must include*.
            excludes    A list of words (as strings) that the documents
                        *must not include*.
            docs        The list of documents to search in, represented by
                        by doc_ids. If this list is empty, the entire corpus
                        is searched.
        """
        
        # If no doc ids were supplied, search the entire corpus.
        if not docs:
            docs = range(0, len(self.corpus_tfidf))
    
        # Convert all the keywords to their IDs.
        # Force them to lower case in the process.
        include_ids = []
        exclude_ids = []
    
        for word in includes:
            # Lookup the ID for the word.            
            word_id = self.getIDForWord(word.lower())            
            
            # Verify the word exists in the dictionary.
            if word_id == -1:
                print 'WARNING: Word \'' + word.lower() + '\'not in dictionary!'
                continue
            
            # Add the word id to the list.
            include_ids.append(word_id)
            
        for word in excludes:
            exclude_ids.append(self.getIDForWord(word.lower()))
        
        results = []
    
        # For each of the documents to search...
        
        for doc_id in docs:
            # Get the sparse tf-idf vector for the next document.
            vec_tfidf = self.corpus_tfidf[doc_id]
            
            # Create a list of the word ids in this document.
            doc_words = [tfidf[0] for tfidf in vec_tfidf]
            
            match = True
            
            # Check for words that must be present.
            for word_id in include_ids:
                if not word_id in doc_words:
                    match = False
                    break
            
            # If we failed the 'includes' test, skip to the next document.
            if not match:
                continue
    
            # Check for words that must not be present.
            for word_id in exclude_ids:
                if word_id in doc_words:
                    match = False
                    break
            
            # If we passed the 'excludes' test, this is a valid result.
            if match:
                results.append(doc_id)
        
        return results
            
    
    def printTopNWords(self, topn=10):
        """
        Print the 'topn' most frequent words in the corpus.
        
        This is useful for checking to see if you have any common, bogus tokens
        that need to be filtered out of the corpus.
        """
        
        # Get the dictionary as a list of tuples.
        # The tuple is (word_id, count)
        word_counts = [(key, value) for (key, value) in self.dictionary.dfs.iteritems()]
        
        # Sort the list by the 'value' of the tuple (incidence count) 
        from operator import itemgetter
        word_counts = sorted(word_counts, key=itemgetter(1))
        
        # Print the most common words.
        # The list is sorted smallest to biggest, so...
        print 'Top', topn, 'most frequent words'
        for i in range(-1, -topn, -1):
            print '  %s   %d' % (self.dictionary[word_counts[i][0]].ljust(10), word_counts[i][1])
    
    def getVocabSize(self):
        """
        Returns the number of unique words in the final vocabulary (after all
        filtering).
        """
        return len(self.dictionary.keys())
        
    def getIDForWord(self, input_word):
        """
        Lookup the ID for a specific word.
        
        Returns -1 if the word isn't in the dictionary.
        """

        # All words in dictionary are lower case.
        input_word = input_word.lower()
        
        # First check if the word exists in the dictionary.
        if not input_word in self.dictionary.values():
            return -1            
        # If it is, look up the ID.    
        else:
            return self.dictionary.token2id[input_word]
               
    def getDocLocation(self, doc_id):
        """
        Return the filename and line numbers that 'doc_id' came from.
        """
        line_nums = self.doc_line_nums[doc_id]        
        filename = self.files[line_nums[0]]
        return filename, line_nums[1], line_nums[2]
    
    def readDocSource(self, doc_id):
        """
        Reads the original source file for the document 'doc_id' and retrieves
        the source lines.
        """
        # Lookup the source for the doc.
        line_nums = self.doc_line_nums[doc_id]        
        
        filename = self.files[line_nums[0]]
        line_start = line_nums[1]
        line_end = line_nums[2]

        results = []        

        # Open the file and read just the specified lines.        
        with open(filename) as fp:
            for i, line in enumerate(fp):
                # 'i' starts at 0 but line numbers start at 1.
                line_num = i + 1
                
                if line_num > line_end:
                    break
                
                if line_num >= line_start:
                    results.append(line)
    
        return results
    
    def printDocSourcePretty(self, doc_id, max_lines=8, indent='    '):
        """
        Prints the original source lines for the document 'doc_id'.
        
        This function leverages the 'textwrap' Python module to limit the 
        print output to 80 columns.        
        """
            
        # Read in the document.
        lines = self.readDocSource(doc_id)
            
        # Limit the result to 'max_lines'.
        truncated = False
        if len(lines) > max_lines:
            truncated = True
            lines = lines[0:max_lines]

        # Convert the list of strings to a single string.
        lines = '\n'.join(lines)

        # Remove indentations in the source text.
        dedented_text = textwrap.dedent(lines).strip()
        
        # Add an ellipsis to the end to show we truncated the doc.
        if truncated:
            dedented_text = dedented_text + ' ...'
        
        # Wrap the text so it prints nicely--within 80 columns.
        # Print the text indented slightly.
        pretty_text = textwrap.fill(dedented_text, initial_indent=indent, subsequent_indent=indent, width=80)
        
        print pretty_text   
    
    def save(self, save_dir='./'):
        """
        Write out the built corpus to a save directory.
        """
        # Store the tag tables.
        pickle.dump((self.tagsToDocs, self.docsToTags), open(save_dir + 'tag-tables.pickle', 'wb'))
        
        # Store the document titles.
        pickle.dump(self.titles, open(save_dir + 'titles.pickle', 'wb'))
        
        # Write out the tfidf model.
        self.tfidf_model.save(save_dir + 'documents.tfidf_model')
        
        # Write out the tfidf corpus.
        corpora.MmCorpus.serialize(save_dir + 'documents_tfidf.mm', self.corpus_tfidf)  

        # Write out the dictionary.
        self.dictionary.save(save_dir + 'documents.dict')
        
        # Save the filenames.
        pickle.dump(self.files, open(save_dir + 'files.pickle', 'wb'))
        
        # Save the file ID and line numbers for each document.
        pickle.dump(self.doc_line_nums, open(save_dir + 'doc_line_nums.pickle', 'wb'))
        
        # Objects that are not saved:
        #  - stop_list - You don't need to filter stop words for new input
        #                text, they simply aren't found in the dictionary.
        #  - frequency - This preliminary word count object is only used for
        #                removing infrequent words. Final word counts are in
        #                the `dictionary` object.
        
    @classmethod
    def load(cls, save_dir='./'):
        """
        Load the corpus from a save directory.
        """
        tables = pickle.load(open(save_dir + 'tag-tables.pickle', 'rb'))
        tagsToDocs = tables[0]
        docsToTags = tables[1]        
        titles = pickle.load(open(save_dir + 'titles.pickle', 'rb'))
        tfidf_model = TfidfModel.load(fname=save_dir + 'documents.tfidf_model')
        corpus_tfidf = corpora.MmCorpus(save_dir + 'documents_tfidf.mm')
        dictionary = corpora.Dictionary.load(fname=save_dir + 'documents.dict')
        files = pickle.load(open(save_dir + 'files.pickle', 'rb'))
        doc_line_nums = pickle.load(open(save_dir + 'doc_line_nums.pickle', 'rb'))
        
        ksearch = KeySearch(dictionary, tfidf_model, 
                            corpus_tfidf, titles, tagsToDocs,
                            docsToTags, files, doc_line_nums) 
        
        return ksearch
            