
from inverted_index_colab import *
import nltk
from nltk.tokenize import word_tokenize
import math
import numpy as np
from numpy.linalg import norm
import pandas as pd
import HWTFIDFPIPE as pipe1

#should be activated only one time !
#nltk.download('stopwords')


from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))

# FROM GCP FOR THE SEARCH BODY PART
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


PIPE = 'HW'

#reading the title dictionary - MOVE IT TO MAIN AFTER TESTING
with open('title_dic.pkl', 'rb') as f:
    title_dict = pickle.load(f)



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
            for doc_id , amount in res:
                try:
                    out[doc_id]+=1
                except:
                    out[doc_id] = 1
        except Exception as e:
            print('error in Anchor/title index occured - ' , e)

    return [(id , title_dict[id]) for id in sorted(out , key=out.get ,reverse = True)]


def PreProc(text):
    """
    preprocess pipeline according to corpus preprocess,  for TFIDF inquieries
    :param text:  free text
    :return: tokens
    """
    if PIPE == 'HW':
        return pipe1.IR_Tokenize(text)

def get_TFIDF(q_text , index , N):
    """
    Function that retrives top N files matching each query accoding to TFIDF and cosine similarity.
    :param q_text: free text of query
    :param index: inverted index to search in
    :param N: top number of documents to retrive
    :return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """


    #preprocess according to corpus preprocess
    q_tokens = list(set(PreProc(q_text)))

    #retrive docs and score

    if PIPE == 'HW':
        #HW expectes queries as ditionary of {id  : tokens }
        res = pipe1.get_topN_score_for_queries({1 : q_tokens} , index , N)[1]
        return [(id , title_dict[id]) for id ,score in res]



