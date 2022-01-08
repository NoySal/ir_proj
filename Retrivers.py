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


def Corpus_Tokenizer(query):
    """"
    Temporarely the usual
    """
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return [token for token in tokens if token not in all_stopwords]



def get_binary(query, inv_idx):
    """
    Function to binary search in indices (for either anchor or title retrival)
    :param query:  query to tokenize and search
    :param inv_idx: inverted index to binary search in
    :return: list of relevant documents  , sorted by query tokens matches
    """
    tokens = word_tokenize(query.lower())
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


def PreProc(text, PIPE):
    """
    preprocess pipeline according to corpus preprocess,  for TFIDF inquieries
    :param text:  free text
    :return: tokens
    """
    if PIPE == 'HW':
        return pipe1.IR_Tokenize(text)
    if PIPE == 'opt':
        return Corpus_Tokenizer(text)


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
    q_tokens = list(set(PreProc(q_text, PIPE)))

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
        print(New_Tokenizer(q))



