# wiki-sim-search #
Similarity search on Wikipedia using gensim in Python.

The goals of this project are the following two features:

1. Create LSI vector representations of all the articles in English Wikipedia using a modified version of the make_wikicorpus.py script in gensim.
2. Perform concept searches and other fun text analysis on Wikipedia, also using gensim functionality.

## Generating Vector Representations ##

I started with the [make_wikicorpus.py](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/scripts/make_wikicorpus.py) script from gensim, and the results of my script are nearly identical.

My changes were the following:
* I broke out each of the steps and commented the hell out of them to explain what was going on in each.
* For clarity and simplicity, I removed the "online" mode of operation.
* I modified the script to save out the names of all of the Wikipedia articles as well, so that you could perform searches against the dataset and get the names of the matching articles.
* I added the conversion to LSI step.

### What to expect ###

I pulled down the latest Wikipedia dump on 1/18/17; here are some statistics on it:

<table>
<tr><td>17,180,273</td><td>Total number of articles (without any filtering)</td></tr>
<tr><td>4,198,780</td><td>Number of articles after filtering out "article redirects" and "short stubs"</td></tr>
<tr><td>2,355,066,808</td><td>Total number of tokens in all articles (without any filtering)</td></tr>
<tr><td>2,292,505,314</td><td>Total number of tokens after filtering articles</td></tr>
<tr><td>8,746,676</td><td>Total number of unique words found in all articles (*after* filtering articles)</td></tr>
</table>

Vectorizing all of Wikipedia is a fairly lengthy process, and the data files are large. Here is what you can expect from each step of the process.

These numbers are from running on my desktop PC, which has an Intel Core i7 4770, 16GB of RAM, and an SSD.

<table>
<tr><td>#</td><td>Step</td><td>Time (h:m)</td><td>Output File</td><td>File Size</td></tr>
<tr><td>0</td><td>Download Wikipedia Dump</td><td>--</td><td>enwiki-latest-pages-articles.xml.bz2</td><td>12.6 GB</td></tr>
<tr><td>1</td><td>Parse Wikipedia & Build Dictionary</td><td>3:12</td><td>dictionary.txt.bz2</td><td>769 KB</td></tr>
<tr><td>2</td><td>Convert articles to bag-of-words vectors</td><td>3:32</td><td>bow.mm</td><td>9.44 GB</td></tr>
<tr><td>2a.</td><td>Store article titles</td><td>--</td><td>bow.mm.metadata.cpickle</td><td>152 MB</td></tr>
<tr><td>3</td><td>Learn tf-idf model from document statistics</td><td>0:47</td><td>tfidf.tfidf_model</td><td>4.01 MB</td></tr>
<tr><td>4</td><td>Convert articles to tf-idf</td><td>1:40</td><td>corpus_tfidf.mm</td><td>17.9 GB</td></tr>
<tr><td>5</td><td>Learn LSI model with 300 topics</td><td>2:07</td><td>lsi.lsi_model</td><td>3.46 MB</td></tr>
<tr><td></td><td></td><td></td><td>lsi.lsi_model.projection</td><td>3 KB</td></tr>
<tr><td></td><td></td><td></td><td>lsi.lsi_model.projection.u.npy</td><td>228 MB</td></tr>
<tr><td>6</td><td>Convert articles to LSI</td><td>0:58</td><td>lsi_index.mm</td><td>1 KB</td></tr>
<tr><td></td><td></td><td></td><td>lsi_index.mm.index.npy</td><td>4.69 GB</td></tr>
<tr><td></td><td><strong>TOTALS</strong></td><td><strong>12:16</strong></td><td></td><td><strong>45 GB</strong></td></tr>
</table>

I recommend converting the LSI vectors directly to a MatrixSimilarity class rather than performing the intermediate step of creating and saving an "LSI corpus". If you do, it takes longer and the resulting file is huge:

<table>
<tr><td>6</td><td>Convert articles to LSI and save as MmCorpus</td><td>2:34</td><td>corpus_lsi.mm</td><td>33.2 GB</td></tr>
</table>

The final LSI matrix is pretty huge. We have ~4.2M articles with 300 features, and the features are 32-bit (4-byte) floats. 

To store this matrix in memory, we need (4.2E6 * 300 * 4) / (2^30) = 4.69GB of RAM!

Once the script is done, you can delete bow.mm (9.44 GB), but the rest of the data you'll want to keep for performing searches.

### Running the script ###

Before running the script, download the latest Wikipedia dump here:
https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

Save the dump file in the ./data/ directory of this project.

Then, run `make_wikicorpus.py` to fully parse Wikipedia and generate the LSI index!

The script enables gensim logging, and saves all the logging to `log.txt` in the project directory. I've included an example log.txt in the project. You can open this log while the script is running to get more detailed progress updates.

The script also prints an overview to the console; here is an exmaple output:

```
Parsing Wikipedia to build Dictionary...
    Building dictionary took 3:05
    8746676 unique tokens before pruning.

Converting to bag of words...
    Conversion to bag-of-words took 3:47

Learning tf-idf model from data...
    Building tf-idf model took 0:47
     
Applying tf-idf model to all vectors...
    Applying tf-idf model took 1:40

Learning LSI model from the tf-idf vectors...
    Building LSI model took 2:07

Applying LSI model to all vectors...
    Applying LSI model took 2:00
```

## Concept Searches on Wikipedia ##
Once you have the LSI vectors for Wikipedia, you're ready to perform similarity searches.

### Basic Search Script ###
The script `run_search.py` shows a bare bones approach to performing a similarity search with gensim. 

Here is the example output:

```
Loading Wikipedia LSI index (15-30sec.)...
   Loading LSI vectors took 13.03 seconds

Loading Wikipedia article titles...

Searching for articles similar to 'Topic model':
    Similarity search took 320 ms
    Sorting took 8.45 seconds

Results:
    Topic model
    Online content analysis
    Semantic similarity
    Information retrieval
    Data-oriented parsing
    Concept search
    Object-role modeling
    Software analysis pattern
    Content analysis
    Adaptive hypermedia
```

### Advanced Search with SimSearch ###
For some more bells and whistles, I've pulled over my SimSearch project.

The SimSearch and KeySearch classes (in `simsearch.py` and `keysearch.py`) add a number of features:

* Supply new text as the input to a similarity search.
* Interpret similarity matches by looking at which words contributed most to the similarity.
* Identify top words in clusters of documents.

To see some of these features, look at and run `searchWithSimSearch.py`

#### Example 1 #####
Example 1 searches for articles similar to the article 'Topic model', and also interprets the top match.

Example output:

```
Loading Wikipedia article titles

Loading dictionary...
    Took 0.81 seconds

Loading tf-idf model...
    Took 0.08 seconds

Creating tf-idf corpus object (leaves the vectors on disk)...
    Took 0.82 seconds

Loading LSI model...
    Took 0.73 seconds

Loading Wikipedia LSI index...
    Took 13.21 seconds

Searching for similar articles...
Most similar documents:
  0.90    Online content analysis
  0.90    Semantic similarity
  0.89    Information retrieval
  0.89    Data-oriented parsing
  0.89    Concept search
  0.89    Object-role modeling
  0.89    Software analysis pattern
  0.88    Content analysis
  0.88    Adaptive hypermedia
  0.88    Model-driven architecture

Search and sort took 9.59 seconds

Interpreting the match between 'Topic model' and 'Online content analysis' ...

Words in doc 1 which contribute most to similarity:
             text  +0.065
             data  +0.059
            model  +0.053
           models  +0.043
            topic  +0.034
         modeling  +0.031
         software  +0.028
         analysis  +0.019
           topics  +0.019
       algorithms  +0.014
          digital  +0.014
            words  +0.012
          example  +0.012
         document  +0.011
      information  +0.010
         language  +0.010
           social  +0.009
           matrix  +0.008
         identify  +0.008
         semantic  +0.008

Words in doc 2 which contribute most to similarity:
         analysis  +0.070             trains  -0.001
             text  +0.067
          content  +0.054
          methods  +0.035
        algorithm  +0.029
         research  +0.027
           online  +0.026
           models  +0.026
             data  +0.014
      researchers  +0.014
            words  +0.013
              how  +0.013
    communication  +0.013
           sample  +0.012
           coding  +0.009
         internet  +0.009
              web  +0.009
       categories  +0.008
            human  +0.008
           random  +0.008

Interpreting match took 0.75 seconds
```

#### Example 2 ####
Example 2 demonstrates searching using some new input text as the query. I've included the markdown for a couple of my blog articles as example material for the search.

#### Example 3 ####
Prints the top 10 words associated with each of the topics, and also writes these out to `topic_words.txt`
