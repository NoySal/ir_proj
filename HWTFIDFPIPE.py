import numpy as np
import pandas as pd
import math
from inverted_index_colab import *
from numpy.linalg import norm

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


def get_candidate_documents_and_scores(query_to_search, index, words, pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(index.DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
            normlized_tfidf = [(doc_id, (freq / index.DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """
    #print('DEBUG TFIDF MATRIX GENERATION : ')

    total_vocab_size = len(index.term_total)
    #print("TOTAL VOCAB SIZE : " ,total_vocab_size )
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words,
                                                           pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    #print("CANDIDATE SCORES : " ,candidates_scores)
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    #print("UNIQUE CANDIDTAS : " , unique_candidates)

    D = np.zeros((len(unique_candidates), total_vocab_size))

    D = pd.DataFrame(D)
    #print("DATAFRAME BEFORE SETTING IS : " , D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf
    #print('DEBUG TFIDF MATRIX GENERATION : ENDED ####')
    return D

def cosine_similarity(D, Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE
    #print("##DEBUG COSINE SIM")
    #print("Q IS " , Q)
    return {i: (np.dot(D.loc[i], Q) / (norm(D.loc[i]) * norm(Q))) for i in D.index}


def get_top_n(sim_dict, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]


def get_topN_score_for_queries(queries_to_search, index, N=3):
    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    # YOUR CODE HERE
    words, pls = get_posting_gen(index)
    N_dict = {}
    for qId, qTokens in queries_to_search.items():
        #print("DEBUG GET N : ")
        #print(qId , " tokens : " , qTokens)
        tfidt_df = generate_document_tfidf_matrix(qTokens, index, words, pls)
        #print("TFIDF DF IS : ")
        #print(tfidt_df)
        #print("Query vector is : ")
        qVec = generate_query_tfidf_vector(qTokens, index)
        #print(qVec)
        #print("Scores are : ")
        scores = cosine_similarity(tfidt_df, qVec)
        #print(scores)
        N_dict[qId] = get_top_n(scores, N)

    return N_dict


