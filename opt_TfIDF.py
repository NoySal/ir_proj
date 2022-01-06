
import numpy as np
import pandas as pd
import math
from inverted_index_gcp import *

from numpy.linalg import norm
from HWTFIDFPIPE import get_posting_gen , IR_Tokenize
from heapq import nlargest
from collections import Counter

def get_OPT_Tfidf(q_tokens , index , N=100):
    """
    Tfidf Optimized function , recieves tokens as a list of tokens (with duplicates)
    :param q_tokens:  list of tokens , with duplicatse.
    :param index:  inverted index.
    :return: list of documents : title , sorted by their IDF score with query
    """
    q_size = len(q_tokens)
    corpus_docs = len(index.DL)
    q_tokens = list(Counter(q_tokens).items())
    sim_q = {}
    for q_word , q_count in q_tokens:
        if q_word in index.term_total.keys():
            for doc_id, word_count in read_posting_list(index, q_word):
                tw = word_count * np.log10(corpus_docs / index.df[q_word])
                if doc_id in sim_q.keys():
                    sim_q[doc_id] += q_count * tw
                else:
                    sim_q[doc_id] = q_count * tw

    for doc_id in sim_q.keys():
        sim_q[doc_id] = sim_q[doc_id]*(1/q_size) * (1/index.DL[doc_id])


    if len(sim_q) < 100:
        return sorted(sim_q , key = sim_q.get ,reverse=True)
    else: #use heap sort and return
        top_100 = nlargest(N, sim_q , key = sim_q.get )
        return sorted(top_100 ,reverse=True)

