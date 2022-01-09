from inverted_index_gcp import *
import nltk
from nltk.tokenize import word_tokenize
import math
import numpy as np
from numpy.linalg import norm
import pandas as pd
import HWTFIDFPIPE as pipe1
import pickle

from opt_TfIDF import *
import nltk
# should be activated only one time !
#nltk.download('stopwords')

from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))

# FROM GCP FOR THE SEARCH BODY PART
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


def Corpus_Tokenizer(text, method ='norm'):
    """"
    During developing - had a 'stem' and 'lemm' methods.
    """

    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    if method =='norm':
        return [token for token in tokens if token not in all_stopwords]




def get_binary(query, inv_idx):
    """
    Function to binary search in indices (for either anchor or title retrival)
    :param query:  query to tokenize and search
    :param inv_idx: inverted index to binary search in
    :return: list of relevant documents  , sorted by query tokens matches
    """
    #tokens = word_tokenize(query.lower())
    tokens = Corpus_Tokenizer(query)
    out = {}
    for token in tokens:
        try:
            res = read_posting_list(inv_idx, token)
            for doc_id, amount in res:
                try:
                    out[doc_id] += 1
                except:
                    out[doc_id] = 1
        except Exception as e:
            print('error in Anchor/title index occured - ', e)

    return sorted(out, key=out.get, reverse=True)



def text_title_Merge(query, text_idx, text_n_docs, text_avg_doc, title_idx, title_n_docs, title_avg_doc, N=200):
    """
    recieves a query and run it on text and title indices.
    returns a merged list of docs - based on TFIDF score of each index and pre-determined weights.
    param query: string, query to tokenize and search.
    param text_idx : inverted index of text.
    param text_n_docs : int, number of docs in text
    param text_avg_doc : float , average length of doc in text index.
    param title_idx : inverted index of title
    param title_n_docs : int, number of docs in title.
    param title_avg_doc : float , average length of title docs.
    returns : list of doc_id . sorted by combined tf-idf scores and weights.
    """
    tex_w = 0.74  # found via optimization
    tit_w = 0.31  # found via optimization

    q_tokens = Corpus_Tokenizer(query)
    text_retrival = get_opt_BM25_for_joint(q_tokens, text_idx, text_n_docs, text_avg_doc, N=N)
    title_retrival = get_opt_BM25_for_joint(q_tokens, title_idx, title_n_docs, title_avg_doc, N=N)
    doc_dict = {}
    for doc, score in text_retrival:
        doc_dict[doc] = tex_w * score
    for doc, score in title_retrival:
        if doc in doc_dict.keys():
            doc_dict[doc] += tit_w * score
        else:
            doc_dict[doc] = tit_w * score
    q_res = sorted(doc_dict, key=doc_dict.get, reverse=True)

    return q_res

def weightSort(docs, prw , pvw ,pr_scale , pv_scale,tw  ):
  """
  resorts a list of docs according to new weights
  docs : list of docs retrived from vectoric model
  param prw: page rank weight to include
  param pvw : page view wieght to include
  param pr_scale , pv_scale : tuple , scaling for min max normalization (max , min , mean)
  param tw : position at  K weight to consider
  returns :  list of doc_id  , sorted by calculated weights.
  """

  weighted = {}

  pr_max , pr_min , pr_mean = pr_scale
  pv_max , pv_min , pv_mean = pv_scale


  for i , doc_id in enumerate(docs):
      #page rank score for document , normalized by min max
      pr = (pr_dict.get(doc_id , 1) - pr_mean) / (pr_max -  pr_min) # 1 is approximatly PR mean
      #page view score for document , normalized by minmax
      pv = (pv_dict.get(doc_id , 670)  - pr_mean  )  / ( pr_max - pr_min)  # 670 mean value of page view
      #position at k for document
      pk = 1 / (i+1)
      weighted[doc_id] = prw * pr + pvw * pv +tw * pk
  return  sorted(weighted , key = weighted.get , reverse= True)

def get_TFIDF(q_text, index, corpus_docs , avg_dl , N=100, PIPE='HW'):
    """
    Function that retrives top N files matching each query accoding to TFIDF and cosine similarity.
    :param q_text: free text of query
    :param index: inverted index to search in
    :param N: top number of documents to retrive
    :param corpus_docs : int , optimization - number of docs in corpus
    param avg_dl : float, optimization - average document size in corpus
    :param PIPE: differenciate between naive (homework pipe and optimized)
    :return: list of docs id, sorted by rank
    """
    # preprocess according to corpus preprocess
    q_tokens = list(set(Corpus_Tokenizer(q_text)))

    # retrive docs and score

    if PIPE == 'HW':
        # HW expectes queries as ditionary of {id  : tokens }
        #res = pipe1.get_topN_score_for_queries({1: q_tokens}, index, N)[1]
        res = get_OPT_Tfidf(q_tokens, index ,corpus_docs , N)

    if PIPE == 'opt':
        # using optimized tfIDF
        res = get_opt_BM25(q_tokens, index, corpus_docs , avg_dl, N )
        return res

    return res


def get_pagerank(id_lst , pr_dict):
    """
    Function retrieves list of pagerankes suitability to list of docs id
    :param id_lst: list of docs id
    :return: list of pagerankes
    """
    pr_lst = []
    for doc_id in id_lst:
        try:
            pr_lst.append(pr_dict[doc_id])
        except:
            pr_lst.append(-1)
    return pr_lst


if __name__ =="__main__":
    queries = ['what is love' , 'why do men have nipples' , 'what to watch']
    for q in queries:
        print(Corpus_Tokenizer(q , method='lem'))



