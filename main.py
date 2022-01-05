from inverted_index_colab import *
from os import path
import nltk
import Retrivers as ret
import re
from time import time

#nltk.download('stopwords')

from nltk.corpus import stopwords


def createIndices():

    mod_path = os.path.dirname(os.path.realpath(__file__))
    print("Creating Indices")
    title_idx = InvertedIndex().read_index(mod_path,'title' )
    anchor_idx = InvertedIndex().read_index(mod_path ,'anchor' )
    text_idx = InvertedIndex().read_index(mod_path, 'text')

    print("Searching Anchor index for query \'political\'")
    print(read_posting_list(anchor_idx, 'political'))
    print("Searching Title index for query \'abraham\'")
    print(read_posting_list(title_idx, 'abraham'))


def test_binary():
    mod_path = os.path.dirname(os.path.realpath(__file__))
    title_idx = InvertedIndex().read_index(mod_path, 'title')
    res = ret.get_binary('Abraham Lincoln', title_idx)
    print(res)

def test_tfidf(query , N=10):
    index = InvertedIndex().read_index(os.path.dirname(os.path.realpath(__file__)), 'text')
    start = time()
    print('TESTING NAIVE : ')
    print(ret.get_TFIDF(query, index , N , PIPE = 'HW'))
    print(f'took {time() - start} sec')
    start = time()
    print('TESTING Opt TFIDF : ')
    print(ret.get_TFIDF(query, index , N , PIPE = 'opt'))
    print(f'took {time() - start} sec')

    return



def test_tokenizer(text):
    print('Original text  :  ')
    print(text)
    print('Tokens   :    ' )
    tokens = ret.IR_Tokenize(text)
    print(tokens)

    return tokens
if __name__ =="__main__":

    mod_path = os.path.dirname(os.path.realpath(__file__))


    print(test_tfidf('field marshal killed thousand of indians holocaust is among us'))
