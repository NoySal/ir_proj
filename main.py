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
    title_idx = InvertedIndex().read_index(path.join(mod_path , 'content/title/') ,'title' )
    anchor_idx = InvertedIndex().read_index(path.join(mod_path , 'content/anchor/') ,'anchor' )
    text_idx = InvertedIndex().read_index(path.join(mod_path, 'content/text/'), 'text')
    print(read_posting_list(anchor_idx, 'political'))
    print(read_posting_list(title_idx, 'abraham'))


def test_binary():
    mod_path = os.path.dirname(os.path.realpath(__file__))
    title_idx = InvertedIndex().read_index(path.join(mod_path, 'content/title/'), 'title')
    res = ret.get_binary('Abraham Lincoln', title_idx)
    print(res)

def test_tokenizer(text):
    print('Original text  :  ')
    print(text)
    print('Tokens   :    ' )
    tokens = ret.IR_Tokenize(text)
    print(tokens)

    return tokens
if __name__ =="__main__":

    mod_path = os.path.dirname(os.path.realpath(__file__))

    print('CHANGE INDICES TO DIFFERENT HASH RANGES FOR EASIER ACCESS ')

