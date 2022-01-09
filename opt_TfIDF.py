
import numpy as np
import pandas as pd
import math
from inverted_index_gcp import *

from numpy.linalg import norm
from HWTFIDFPIPE import get_posting_gen , IR_Tokenize
from heapq import nlargest
from collections import Counter

def get_OPT_Tfidf(q_tokens , index ,corpus_docs, N=100):
    """
    Tfidf Optimized function , recieves tokens as a list of tokens (with duplicates)
    :param q_tokens:  list of tokens , with duplicatse.
    :param index:  inverted index.
    :param corpus_docs : int , optimization - number of docs in corpus
    :return: list of documents : title , sorted by their IDF score with query
    """
    q_size = len(q_tokens)
    q_tokens = list(Counter(q_tokens).items())
    sim_q = {}
    for q_word , q_count in q_tokens:
        if q_word in index.term_total.keys():
            for doc_id, word_count in read_posting_list(index, q_word):
                if doc_id==0:  #friggin missing values!
                    continue
                tw = word_count * np.log10(corpus_docs / index.df[q_word])
                if doc_id in sim_q.keys():
                    sim_q[doc_id] += q_count * tw
                else:
                    sim_q[doc_id] = q_count * tw

    for doc_id in sim_q.keys():
        sim_q[doc_id] = sim_q[doc_id]*(1/q_size) * (1/index.DL[doc_id])


    if True: #len(sim_q) < 100: ##heap sort makes thing wrong!
        return sorted(sim_q, key=sim_q.get, reverse=True)[:N]
    else: #use heap sort and return
        top_100 = nlargest(N, sim_q , key = sim_q.get )
        return sorted(top_100 ,reverse=True)

def get_BM25(q_tokens , index ,corpus_docs , avg_dl , k =1.5,b =0.75 , N=100):
    """
    BM25 retrival model from inverted index .
    param q_tokens - list of str,  tokens of words processed like corpus.
    param index : inverted index class
    :param corpus_docs : int , optimization - number of docs in corpus
    param avg_dl : float, average document size in corpus
    param k : float, range between [1.2 ,2] optimized to corpus
    param b : float , ~0.75 optimizable
    param N : int , number of best-fit docs to retrive
    :return: list of documents : title , sorted by their IDF score with query
    """
    ####
    # if this method is good , we can create one new posting list , and one new q_idf dictionary.
    # or at least q_idf dictionary for all the words.
    ####
    q_tokens = list(Counter(q_tokens).items())
    sim_q = {}
    for q_word , q_count in q_tokens:
        if q_word in index.term_total.keys():
            q_idf = np.log((1 + corpus_docs) / (index.df[q_word] + 0.5))
            if q_word in index.term_total.keys():
                for doc_id, word_count in read_posting_list(index, q_word):
                    if doc_id == 0:  # friggin missing values!
                        continue
                    tw = word_count * (k + 1) / (word_count + k * (1 - b + b * index.DL[doc_id] / avg_dl))
                    if doc_id in sim_q.keys():
                        sim_q[doc_id] += tw * q_idf
                    else:
                        sim_q[doc_id] = tw * q_idf


    if True: #len(sim_q) < 100: ##heap sort makes thing wrong!
        return sorted(sim_q, key=sim_q.get, reverse=True)[:N]
    else: #use heap sort and return
        top_100 = nlargest(N, sim_q , key = sim_q.get )
        return sorted(top_100 ,reverse=True)

def get_opt_BM25(q_tokens, index, corpus_docs, avg_dl, k=3, b=0.25, N=100):
    """
    BM25 retrival model from inverted index .
    param q_tokens - list of str,  tokens of words processed like corpus.
    param index : inverted index class
    :param corpus_docs : int , optimization - number of docs in corpus
    param avg_dl : float, average document size in corpus
    param k : float, range between [1.2 ,2] optimized to corpus
    param b : float , ~0.75 optimizable
    param N : int , number of best-fit docs to retrive
    :return: list of documents : title , sorted by their IDF score with query
    """
    ####
    # if this method is good , we can create one new posting list , and one new q_idf dictionary.
    # or at least q_idf dictionary for all the words.
    ####
    q_tokens = list(Counter(q_tokens).items())
    sim_q = {}
    for q_word, q_count in q_tokens:
        if q_word in index.term_total.keys():
            q_idf = np.log((1 + corpus_docs) / (index.df[q_word] + 0.5))
            if q_word in index.term_total.keys():
                for doc_id, word_count in read_posting_list(index, q_word):
                    if doc_id == 0:  # friggin missing values!
                        continue
                    tw = word_count * (k + 1) / (word_count + k * (1 - b + b * index.DL[doc_id] / avg_dl))
                    if doc_id in sim_q.keys():
                        sim_q[doc_id] += tw * q_idf
                    else:
                        sim_q[doc_id] = tw * q_idf

    if True: #len(sim_q) < 100: ##heap sort makes thing wrong!
        return sorted(sim_q, key=sim_q.get, reverse=True)[:N]
    else:  # use heap sort and return
        top_100 = nlargest(N, sim_q, key=sim_q.get)
        return sorted(top_100, reverse=True)