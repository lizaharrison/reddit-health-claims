# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : run_tfidf.py
AUTHOR : Eliza Harrison

Runs vocabulary generation and feature extraction for subreddit clustering.
Generates feature vocabulary (for LDA) and TF-IDF matrix (for k-means)

"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import cluster_dir, load_corpus


def vocab_dir(action='set',
              min_df=params.vocab_params_default['min_df'],
              max_df=params.vocab_params_default['max_df'],
              ):
    """
    Creates or sets directory for feature vocabulary

    Parameters
    ----------
    action : str
        Choose whether to create new directory ('new') or set working directory
        to existing directory ('set')
    min_df : int
        Minimum document frequency (minimum number of subreddits in which term must be present)
    max_df : float
        Maximum document frequency (maximum proportion of subreddits in which term must be present)

    Returns
    -------
    directory: str
        Filepath of feature vocabulary directory

    """

    directory = './vocab_{}_{}'.format(min_df,
                                       max_df,
                                       )
    if action == 'new':
        if not os.path.exists(directory):
            os.makedirs(directory)

    os.chdir(directory)

    return directory


def check_empty(tfidf_matrix, corpus):
    """
    Checks for subreddits with no terms remaining after applying document frequency limits

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        Matrix produced following TF-IDF
    corpus : dataframe
        Final subreddit clustering corpus

    Returns
    -------
    empty_subs : series
        Contains subreddits with no features remaining (empty subreddits)

    """

    ids = np.array(tfidf_matrix.sum(axis=1) == 0).ravel()
    empty_subs = corpus.loc[ids]
    if len(empty_subs) > 0:
        print('Empty subreddits following stopword removal: \n{}'.format(empty_subs))

    return empty_subs


def perform_tfidf(corpus,
                  min_df=params.vocab_params_default['min_df'],
                  max_df=params.vocab_params_default['max_df'],
                  ):
    """
    Performs feature extraction using TF-IDF and the specified feature vocabulary

    Parameters
    ----------
    corpus : dataframe
        Final subreddit clustering corpus
    min_df : int
        Minimum document frequency (minimum number of subreddits in which term must be present)
    max_df : float
        Maximum document frequency (maximum proportion of subreddits in which term must be present)

    Returns
    -------
    (count_vectoriser, count_matrix, tfidf_matrix) : tuple
    Contains the CountVectorizer object, term frequency matrix and TF-IDF matrix

    """

    format_str = '_{}_{}'.format(min_df, max_df)
    print('Performing TF-IDF...')

    try:
        count_matrix = load_npz('count_matrix{}.npz'.format(format_str))
        tfidf_matrix = load_npz('tfidf_matrix{}.npz'.format(format_str))
        count_vectoriser = pickle.load(open('count_vect{}.pkl'.format(format_str), 'rb'))

    except FileNotFoundError:
        # Default tokenizer == '(?u)\b\w\w+\b'
        count_vectoriser = CountVectorizer(min_df=min_df,
                                           max_df=max_df,
                                           )
        count_matrix = count_vectoriser.fit_transform(corpus['clean_text'])
        tfidf_transformer = TfidfTransformer()
        tfidf_matrix = tfidf_transformer.fit_transform(count_matrix)

        # Saves key results to file
        format_str = '_{}_{}'.format(min_df, max_df)
        save_npz('count_matrix{}.npz'.format(format_str), count_matrix)
        save_npz('tfidf_matrix{}.npz'.format(format_str), tfidf_matrix)
        pickle.dump(count_vectoriser, open('count_vect{}.pkl'.format(format_str), 'wb'))

    return count_vectoriser, count_matrix, tfidf_matrix


def get_tfidf_info(corpus,
                   count_vectoriser,
                   count_matrix,
                   tfidf_matrix,
                   ):
    """
    Generates key information summary for feature vocabulary

    Parameters
    ----------
    corpus : dataframe
        Final subreddit clustering corpus
    count_vectoriser : vectoriser object
        Sklearn count vectoriser object
    count_matrix : sparse matrix
        Term frequency matrix
    tfidf_matrix : sparse matrix
        TF-IDF matrix

    Returns
    -------
    summary : dict
        Contains key information for feature vocabulary

    """

    # Count vectorizer parameters
    params = count_vectoriser.get_params()
    del params['vocabulary']
    params = {key: str(value) for key, value in params.items()}

    # Vocabulary details
    vocabulary_dict = count_vectoriser.vocabulary_
    vocabulary_list = list(count_vectoriser.get_feature_names())
    stop_words = list(count_vectoriser.stop_words_)

    print('TF-IDF function: \n{}'.format(count_vectoriser))
    print('Features: {}'.format(len(vocabulary_list)))
    print('Stop words: {}'.format(len(stop_words)))

    # Term frequency analysis
    word_freq = pd.DataFrame(count_matrix.sum(axis=0), columns=vocabulary_list).transpose()
    word_freq.columns = ['term_frequency']
    top_words_df = word_freq.nlargest(n=100, columns='term_frequency')

    print('Top 100 features: \n{}'.format(top_words_df))
    top_words = [(word, int(top_words_df.loc[word, 'term_frequency'])) for word in top_words_df.index]

    # Check for empty subreddits
    empty_subs = check_empty(tfidf_matrix, corpus)
    if empty_subs > 0:
        corpus.drop(empty_subs.index, inplace=True)
        print('Empty subreddits removed from corpus: {}'.format(empty_subs))

    # Compile evaluation summary
    summary = {
        'vocab_params': params,
        'n_features': len(vocabulary_list),
        'n_stop_words': len(stop_words),
        'n_empty_subs': len(empty_subs),
    }

    # pickle.dump(vocabulary_dict, open('vocab{}.pkl'.format(format_str), 'wb'))
    # pickle.dump(stop_words, open('stop_words{}.pkl'.format(format_str), 'wb'))

    return summary


def load_tfidf(min_df=params.vocab_params_default['min_df'],
               max_df=params.vocab_params_default['max_df']):
    """
    Loads vectoriser object and TF-IDF matrix from file

    Parameters
    ----------
    min_df : int
        Minimum document frequency (minimum number of subreddits in which term must be present)
    max_df : float
        Maximum document frequency (maximum proportion of subreddits in which term must be present)

    Returns
    -------
    (count_vectoriser, tfidf_matrix) : tuple
        Contains count vectoriser object

    """

    vocab_dir('set',
              min_df,
              max_df,
              )
    format_str = '_{}_{}'.format(min_df, max_df)
    count_vectoriser = pickle.load(open('count_vect{}.pkl'.format(format_str), 'rb'))
    tfidf_matrix = load_npz('tfidf_matrix{}.npz'.format(format_str))

    return count_vectoriser, tfidf_matrix


if __name__ == '__main__':

    # Running TF-IDF with option to test all parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nposts',
                        help='number of submissions sampled per subreddit',
                        type=int,
                        default=100,
                        )
    parser.add_argument('-i', '--sampleid',
                        help='random state for sampling submissions for each subreddit',
                        type=int,
                        default=1,
                        )
    parser.add_argument('-a',
                        '--all',
                        help='run tf-idf for all parameter combinations (default = 50)',
                        action='store_true',
                        )
    args = parser.parse_args()

    sample_params = params.subreddit_cluster_params

    if args.nposts != 100:
        sample_params['n_posts'] = args.nposts
    if args.randomstate != 1:
        sample_params['sample_id'] = args.sampleid

    if args.all:
        summary_file = 'sub_cluster_summary_ALL.json'
        vocab_params = params.vocab_params_all
    else:
        summary_file = 'sub_cluster_summary_DEFAULT.json'
        vocab_params = [params.vocab_params_default]

    subreddit_corpus = load_corpus(**sample_params)
    print(subreddit_corpus)

    full_summary = {}

    print('All TF-IDF parameters to run: \n{}'.format(vocab_params))
    for vocab_dict in vocab_params:
        print('\nTF-IDF parameters for current test: {}'.format(vocab_dict))
        cluster_dir(**sample_params)
        vocab_dir('new', **vocab_dict)
        tfidf_vect, count_x, tfidf_x = perform_tfidf(subreddit_corpus,
                                                     **vocab_dict
                                                     )
        tfidf_summary, top_features = get_tfidf_info(subreddit_corpus,
                                                     tfidf_vect,
                                                     count_x,
                                                     tfidf_x,
                                                     )
        vocab_key = 'vocab_{}_{}'.format(vocab_dict['min_df'],
                                         vocab_dict['max_df']
                                         )
        full_summary[vocab_key] = tfidf_summary

    cluster_dir(**sample_params)
    json.dump(full_summary, open(summary_file, 'w'),
              indent=4,
              )

    print('\nTF-IDF complete.')
