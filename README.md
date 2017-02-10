# wiki-sim-search
Similarity search on Wikipedia using gensim in Python.

This project provides two features:

1. Creates vector representations of all the articles in English Wikipedia using a modified version of the make_wikicorpus.py script in gensim.
2. (Coming Soon) Allows you to perform concept searches and other fun text analysis on Wikipedia, also using gensim functionality.

Generating Vector Representations
=================================
I started with the [make_wikicorpus.py](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/scripts/make_wikicorpus.py) script from gensim, and the results of my script are nearly identical.

My changes were the following:
* I broke out each of the steps and commented the hell out of them to explain what was going on in each.
* For clarity and simplicity, I removed the "online" mode of operation.
* I modified the script to save out the names of all of the Wikipedia articles as well, so that you could perform searches against the dataset and get the names of the matching articles.
* I added the conversion to LSI step.

+++What to expect

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
<tr><td>#</td><td>Step</td><td>Time</td><td>Output File</td><td>File Size</td></tr>
<tr><td>1</td><td>Download Wikipedia Dump</td><td>?</td><td>enwiki-latest-pages-articles.xml.bz2</td><td>12.6 GB</td></tr>
<tr><td>2</td><td>Parse Wikipedia & Build Dictionary</td><td>3.2 hrs.</td><td>dictionary.txt.bz2</td><td>769 KB</td></tr>
<tr><td>3</td><td>Convert articles to bag-of-words vectors</td><td>3.53 hrs.</td><td>bow.mm</td><td>9.44 GB</td></tr>
<tr><td>3a.</td><td>Store article titles</td><td>N/A</td><td>bow.mm.metadata.cpickle</td><td>152 MB</td></tr>
<tr><td>4</td><td>Learn tf-idf model from document statistics</td><td>46.66 min.</td><td>tfidf.tfidf_model</td><td>4.01 MB</td></tr>
<tr><td>5</td><td>Convert articles to tf-idf</td><td>???</td><td>corpus_tfidf.mm</td><td>17.9 GB</td></tr>
<tr><td>6</td><td>Learn LSI model with 300 topics</td><td>~4hrs. ??</td><td>lsi.lsi_model</td><td>3.46 MB</td></tr>
<tr><td>7</td><td>Convert articles to LSI</td><td>2.57 hrs.</td><td>corpus_lsi.mm</td><td>33.2 GB</td></tr>
</table>

* TODO - I think you can delete the bow.mm at a minimum...

+++Running the script

* TODO - Point to dump file download.



Concept Searches on Wikipedia
=============================
Coming soon.