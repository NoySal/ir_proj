
from inverted_index_colab import *
import nltk
from nltk.tokenize import word_tokenize
import math
import numpy as np


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


def get_posting_gen(index):
    """
    This function returning the generator working with posting list.

    Parameters:
    ----------
    index: inverted index
    """
    words, pls = zip(*index.posting_lists_iter())
    return words, pls


def IR_Tokenize(text):
    """
    Function to tokenize like ass 3 without removing stopwords
    :param text:  text to tokenize
    :return:  tokens
    """
    ##tokenizer for body retrivals

    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    return tokens


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

    return sorted(out , key=out.get ,reverse = True)


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.term_total.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q

