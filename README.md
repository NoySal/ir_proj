## Wikipedia Search Engine
Final project in Information retrival Course 372.1.4406 @ BGU-Uni , submitted by [myself](https://github.com/NoySal) and [Nadav](https://github.com/arbelna).

## Introduction
The engine is cosine simlarity based. It uses various inverted indices on wikipedia (full text , title , and anchor text ) as well as page rank, page view and page title dictionaries.
We've examined different retrival models (TF-IDF , BM-25) and combination of indices (title and body) as well as different Corpus preprocess methods (lemmatization and stemming).
for the various analysis Notebooks - [check here]( https://drive.google.com/drive/folders/1T6T9xsArboF5KgMP11zDk06rCq9L7HV5?usp=sharing) .

## Main Components
**search_frontend.py**- as given by course staff. It contains a Flask interface to run on server and accapt HTML request to retrive via 5 functions : 
_Search_ - Our main retrival model.
_Search_body_ - TFIDF basic model, only on body.
_Search_title_ /_search_anchor_ - Two methods to conduct a binary search for terms in the anchor \ title indices.
_get_pageRank_ , _get_Pageview_ - Two methods to return the PR or PV values of a given list of Doc-id's.

Overall , we've done nothing in this file besides connection our functions to it and placing all the pickle loading and few computations to reduce in-query computations.

**Retrivers.py**-  Our main retrivers. Contains the functions that are called from the frontend.
_Corpus_tokenizer_ for query processing that matches the corpus (index). currently only has a default implementation , but during dev. we've tried stemming and lemmatization.
_get_Binary_ - for binary retrival on indices ( for the requested title / anchor text retrival).
_get_pagerank_ - not exciting, return  from dict.
_get_TFIDF_ - Returns values accoring to TF_IDF calculation and cosine similarity. has two PIPE's - 'HW' for the requested search_body implementation , 
 and the other 'opt' for a BM25 retrival with optimizied parameters (k1 ,b) over the wikipedia corpus -tested on supplied test queries.
_text_title_merge_ - A functions that runs BM25 queries on both the title index and text index , and re-sorts the combined results according to retrived TF-IDF scores and weights.
We've found that the weights 0.7 for the body and 0.3 for the title works best on our corpus.
_Weight_Sort_ - that a Dev. function that I creared to implement a re-sort of a retrived docs list using doc's page rank and page view values.
It gave us really good results on the naive BM-25 implementation on the body only,  but with less success on the parameterized BM-25 body and title merge-  so eventuallty it was not included.

**opt_TfIDF.py**- Where the magic happens.  contains 4 functions , their pretty much the same. all models use HeapSort on large doc_lists to reduce sort time.
_get_OPT_Tfidf_ - retrives a list of docs according to tokens , uses sequencial implementation instead of the naive- vectoric model.
_get_BM25_ - same , only using BM25 model and default k1 , b parameters of ( 1.2 , 0.5)
_get_opt_BM25_ - parameters were changed (after optimization - check the notebook in the link above! )  to k1 =3 , b =0.25
_get_opt_BM25_for_joint_ -gets an inverted index and tokens and returns a list of (doc_id , tf-idf score) for further sorting and merging.

**inverted_index_class.py** - class for inverted index object, contains document length, total_terms and posting locs dictionaries. Capble of reading and writing posting locs to and from bins.

