#!/usr/bin/p > thon
# -*- coding: utf-8 -*-

"""
PROGRAM : run_tsvd.py
AUTHOR : Eliza Harrison

Reduces dimensionality of subreddit corpus prior to K-Means clustering using
TSVD.

"""

import argparse
import json
import os
import pickle

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import cluster_dir
from subreddit_clustering.run_tfidf import load_tfidf


def tsvd_dir(action='set',
             n_components=params.tsvd_params_default['n_components']):
    """
    Creates or sets directory for reduced feature matrix and subsequent analyses
    Parameters
    ----------
    action : str
        Choose whether to create new directory ('new') or set working directory
        to existing directory ('set')
    n_components : int
        Number of components (dimensions) for TSVD

    Returns
    -------
    directory: str
        Filepath of directory for TSVD files

    """

    directory = 'tsvd_{}'.format(n_components)
    if os.getcwd().find('tsvd') > -1:
        os.chdir('../')
    if action == 'new':
        if not os.path.exists(directory):
            os.makedirs(directory)

    os.chdir(directory)

    return directory


def load_tsvd(n_components=params.tsvd_params_default['n_components']):
    """
    Loads reduced feature matrix from file

    Parameters
    ----------
    n_components : int
        Number of components (dimensions) for TSVD

    Returns
    -------
    (model, array) : tuple
       Contains trained TSVD model and resulting low-dimensional array with n components

    """

    tsvd_dir('set',
             n_components=n_components,
             )

    array = pickle.load(open('tsvd_array_{}.pkl'.format(n_components), 'rb'))
    model = pickle.load(open('tsvd_model_{}.pkl'.format(n_components), 'rb'))

    return model, array


def perform_tsvd(tfidf_matrix, n_components=params.tsvd_params_default['n_components']):
    """
    Performs dimensionality reduction of TF-IDF matrix via TSVD

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF matrix
    n_components : int
        Number of components (dimensions) for TSVD

    Returns
    -------
    (model, array) : tuple
       Contains trained TSVD model and resulting low-dimensional array with n components

    """

    try:
        model, array = load_tsvd(n_components)
    except FileNotFoundError:
        print('Performing dimensionality reduction using T-SVD...')
        model = TruncatedSVD(n_components=n_components,
                             n_iter=5,
                             random_state=1,
                             )
        normalizer = Normalizer(copy=False)
        pipeline = make_pipeline(model, normalizer)
        array = pipeline.fit_transform(tfidf_matrix)

    pickle.dump(array, open('tsvd_array_{}.pkl'.format(n_components), 'wb'))
    pickle.dump(model, open('tsvd_model_{}.pkl'.format(n_components), 'wb'))

    return model, array


def get_tsvd_info(model):
    """
    Generates key information summary for outputs of TSVD

    Parameters
    ----------
    model : tsvd model
        Trained sklearn TruncatedSVD model

    Returns
    -------
    summary : dict
        Contains key information for TSVD
    """

    summary = {
        'total_explained_var': float(model.explained_variance_.sum()),
        'total_explained_var_ratio': float(model.explained_variance_ratio_.sum()),
    }

    print('Total variance explained: {}'.format(model.explained_variance_.sum()))
    print('Total variance ratio explained: {}\n'.format(model.explained_variance_ratio_.sum()))

    return summary


if __name__ == '__main__':

    # Running T-SVD with option to test all parameters
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
        tsvd_params = params.tsvd_params_all
    else:
        summary_file = 'sub_cluster_summary_DEFAULT.json'
        vocab_params = [params.vocab_params_default]
        tsvd_params = {'n_components': [params.tsvd_params_default['n_components']]}

    cluster_dir(**sample_params)
    print(sample_params)

    print('All TF-IDF parameters: \n{}'.format(vocab_params))

    for vocab_dict in vocab_params:
        full_summary = json.load(open(summary_file, 'r'))
        print('\nTF-IDF parameters for current test: {}'.format(vocab_dict))
        tfidf_vect, tfidf_x = load_tfidf(**vocab_dict)
        vocab_key = 'vocab_{}_{}'.format(vocab_dict['min_df'],
                                         vocab_dict['max_df'],
                                         )

        for n in tsvd_params['n_components']:
            tsvd_dir(action='new',
                     n_components=n,
                     )
            print('T-SVD components: {}'.format(n))

            # Performs T-SVD dimensionality reduction
            tsvd_model, tsvd_r = perform_tsvd(tfidf_x,
                                              n_components=n,
                                              )
            tsvd_summary = get_tsvd_info(tsvd_model)

            # Saves details/results to dictionary
            tsvd_key = 'tsvd_{}'.format(n)
            full_summary[vocab_key][tsvd_key] = tsvd_summary

        # Updates results JSON file
        cluster_dir(**sample_params)
        json.dump(full_summary, open(summary_file, 'w'),
                  indent=4,
                  )

    print('\nT-SVD complete.')
