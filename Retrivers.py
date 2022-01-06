from inverted_index_colab import *
import nltk
from nltk.tokenize import word_tokenize
import math
import numpy as np
from numpy.linalg import norm
import pandas as pd
import HWTFIDFPIPE as pipe1
import pickle

from opt_TfIDF import get_OPT_Tfidf

import nltk
# should be activated only one time !
# nltk.download('stopwords')


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
    Temporarely no corpus tokenizer used
    """
    return pipe1.IR_Tokenize(query)


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


def get_TFIDF(q_text, index, N, PIPE='HW'):
    """
    Function that retrives top N files matching each query accoding to TFIDF and cosine similarity.
    :param q_text: free text of query
    :param index: inverted index to search in
    :param N: top number of documents to retrive
    :param PIPE: differenciate between naive (homework pipe and optimized)
    :return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    # preprocess according to corpus preprocess
    q_tokens = list(set(PreProc(q_text, PIPE)))

    # retrive docs and score

    if PIPE == 'HW':
        # HW expectes queries as ditionary of {id  : tokens }
        res = pipe1.get_topN_score_for_queries({1: q_tokens}, index, N)[1]

    if PIPE == 'opt':
        # using optimized tfIDF
        res = get_OPT_Tfidf(q_tokens, index, N)
        return [(id, title_dict[id]) for id in res]

    return [(id, title_dict[id]) for id, score in res]


def get_pagerank(id_lst):
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


from time import time


def test_tfidf(query='field marshal killed thousand of indians holocaust is among us', N=10):
    index = InvertedIndex().read_index(os.path.dirname(os.path.realpath(__file__)), 'text')
    start = time()
    print('TESTING NAIVE : ')
    print(get_TFIDF(query, index, N, PIPE='HW'))
    print(f'took {time() - start} sec')
    start = time()
    print('TESTING Opt TFIDF : ')
    print(get_TFIDF(query, index, N, PIPE='opt'))
    print(f'took {time() - start} sec')

    return


