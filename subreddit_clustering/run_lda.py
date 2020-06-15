#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : run_lda.py
AUTHOR : Eliza Harrison

This program performs Latent Dirichlet Allocation on subreddit corpus.

"""

import argparse
import json
import os
import pickle
from operator import itemgetter

import pandas as pd
from gensim import utils
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from matplotlib import pyplot as plt

from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.generate_subreddit_corpus import load_corpus, cluster_dir
from subreddit_clustering.run_tfidf import load_tfidf


def lda_dir(action='set'):
    """
    Creates or sets directory for LDA
    Parameters
    ----------
    action : str
            Choose whether to create new directory ('new') or set working directory
            to existing directory ('set')

    Returns
    -------
    directory: str
            Filepath of subreddit clustering directory

    """

    directory = './lda'

    if action == 'new':
        if not os.path.exists(directory):
            os.makedirs(directory)

    os.chdir(directory)

    return directory


'''
def load_vocab(min_df=vocab_params_default['min_df'], max_df=vocab_params_default['max_df']):
    file = './vocab_{}_{}.pkl'.format(min_df,
                                      max_df,
                                      )
    vocabulary = pickle.load(open(file, 'rb'))

    return vocabulary
'''


def remove_stopwords(string, stopwords):
    """
    Removes words in corpus which have been excluded from each feature vocabulary (stop words)

    Parameters
    ----------
    string : str
        Text from which to remove stopwords
    stopwords : list
        List of stop words to be removed from text

    Returns
    -------
    string_no_stopwords : str
        String after removal of stopwords

    """

    # Based on Gensim remove_stopwords function modified to use custom vocab
    stopwords = frozenset(stopwords)
    string_unicode = utils.to_unicode(string)
    string_no_stopwords = " ".join(w for w in string_unicode.split() if w not in stopwords)

    return string_no_stopwords


def prep_corpus_lda(tfidf_vectoriser, corpus):
    """

    Parameters
    ----------
    tfidf_vectoriser : sklearn vectoriser object
        TF-IDF vectoriser object
    corpus : dataframe
        Final subreddit clustering dataset

    Returns
    -------
    (id2word, doc2bow, corpus_text) : tuple
        Contains token ID to word mapping, document ID to bag-of-words mapping and series
        in which text for each subreddit is split into list-form.

    """

    print('Preparing BoW representation...')
    corpus_text = corpus['clean_text'].apply(lambda x: list(x.split()))

    try:
        id2word = pickle.load(open('id2word.pkl', 'rb'))
        doc2bow = pickle.load(open('doc2bow.pkl', 'rb'))

    except FileNotFoundError:
        vocab_dict = tfidf_vectoriser.vocabulary_
        stop_words = tfidf_vectoriser.stop_words_
        print('Features in TF-IDF vocabulary: {}'.format(len(vocab_dict)))
        print('Stop words in TF-IDF vocabulary: {}'.format(len(stop_words)))

        print('Generating LDA dictionary...')
        id2word = Dictionary([vocab_dict.keys()])
        vocab_check = list(vocab_dict.keys())
        vocab_check.sort()

        print('Checking vocab dictionary:\n',
              '> First word in original vocab: {}\n'.format(vocab_check[0]),
              '> First word in gensim vocab: {}\n'.format(id2word[0]),
              '> Last word in original vocab: {}\n'.format(vocab_check[len(vocab_dict) - 1]),
              '> Last word in gensim vocab: {}\n'.format(id2word[len(vocab_dict) - 1]),
              )

        doc2bow = [id2word.doc2bow(text) for text in corpus_text.values]
        print('Checking corpus + stopwords: \n',
              '> Number of docs in original corpus: {}\n'.format(len(corpus)),
              '> Number of docs in gensim corpus: {}\n'.format(len(doc2bow)),
              )
        empty_subs = [index for index, sub in enumerate(doc2bow) if len(sub) < 1]
        empty_subs = corpus.loc[empty_subs, 'subreddit']
        print(
            '\nSubreddits with no features remaining after stopword removal: \n{}'.format([sub for sub in empty_subs]))
        short_subs = [index for index, sub in enumerate(doc2bow) if len(sub) < 11]
        short_subs = corpus.loc[short_subs, 'subreddit']
        print(
            '\nSubreddits with <10 features remaining after stopword removal: \n{}'.format([sub for sub in short_subs]))

        pickle.dump(id2word, open('id2word.pkl', 'wb'))
        pickle.dump(doc2bow, open('doc2bow.pkl', 'wb'))

        if len(empty_subs) > 0:
            print('Cannot perform LDA on this vocabulary due to empty subreddits\n',
                  '> Number of subreddits with no vocabulary words (after stopword removal): {}'.format(
                      len(empty_subs)),
                  '> Number of subreddits with < 11 vocabulary words (after stopword removal): {}'.format(
                      len(short_subs)))
            return None, None, None
        else:
            pass

    return id2word, doc2bow, corpus_text


def lda_gensim(id2word,
               doc2bow,
               n_topics=params.lda_params_default['n_topics']):
    """
    Implements gensim LDA algorithm.

    Parameters
    ----------
    id2word
        Maps token IDs to words
    doc2bow
        Maps documents to bag-of-words lists
    n_topics : int
        Total number of topics

    Returns
    -------
    model
        Trained LDA model

    """

    try:
        model = LdaModel.load('lda_model_{}'.format(n_topics))
        # coh_model_umass = CoherenceModel.load('umass_coherence_model_{}'.format(n_topics))
        # coh_model_cv = CoherenceModel.load('cv_coherence_model_{}'.format(n_topics))

    except FileNotFoundError:
        # Trains LDA model and returns key words for each topic
        model = LdaModel(corpus=doc2bow,
                         id2word=id2word,
                         iterations=500,
                         num_topics=n_topics,
                         random_state=1,
                         alpha='auto',
                         eta='auto',
                         )

        model.save('lda_model_{}'.format(n_topics))

        '''
        print('Training coherence models...')
        coh_model_umass = CoherenceModel(model=model,
                                         corpus=doc2bow,
                                         dictionary=id2word,
                                         coherence='u_mass',
                                         )
        coh_model_umass.save('umass_coherence_model_{}'.format(n_topics))
        
        coh_model_cv = CoherenceModel(model=model,
                                      texts=corpus_text.values,
                                      dictionary=id2word,
                                      coherence='c_v',
                                      )
        # coh_model_cv.save('cv_coherence_model_{}'.format(n_topics))
        '''

    return model


def get_sub_topics(model,
                   doc2bow,
                   corpus,
                   n_topics=params.lda_params_default['n_topics']):
    """
    Identifies the top (dominant) topics for each subreddit in the dataset.

    Parameters
    ----------
    model
        Trained LDA model
    doc2bow
        Document to bag-of-words list mapping
    corpus
        Final subreddit clustering corpus
    n_topics
        Total number of topics

    Returns
    -------
    (top_topics_df, terms_df) : tuple
        Contains dataframe with the top topic assignments for all subreddits and dataframe containing the
        top terms for each topic.

    """
    # Gets dominant topic for each subreddit (hard clustering)
    sub_topics_array = [sorted(doc,
                               key=itemgetter(1),
                               reverse=True,
                               )[0] for doc in model.get_document_topics(doc2bow)]
    top_topics_df = pd.DataFrame(sub_topics_array,
                                 columns=['topic_number', 'topic_percentage'])
    top_topics_df = top_topics_df.join(corpus.loc[:, ['subreddit', 'health']],
                                       how='left',
                                       )
    top_topics_df = top_topics_df[['subreddit', 'health', 'topic_number', 'topic_percentage']]
    all_topic_terms = model.show_topics(num_topics=n_topics,
                                        formatted=False,
                                        )
    terms_df = pd.concat([pd.DataFrame(all_topic_terms[i][1],
                                       columns=['terms', 'term_probability'],
                                       index=[i] * len(all_topic_terms[i][1])) for i in range(0, n_topics)])
    terms_df['terms_list'] = terms_df.groupby(terms_df.index)['terms'].apply(lambda x: x.to_list())
    terms_df['term_probabilities'] = terms_df.groupby(terms_df.index)['term_probability'].apply(lambda x: x.to_list())
    terms_df.drop(['terms', 'term_probability'],
                  axis=1,
                  inplace=True,
                  )
    terms_df = terms_df.rename_axis('topic_number').reset_index()
    terms_df = terms_df.drop_duplicates(subset='topic_number',
                                        ).set_index('topic_number')
    top_topics_df = pd.merge(top_topics_df, terms_df, how='left', on='topic_number')
    print('LDA topics data: \n{}'.format(top_topics_df))

    top_health_topics = top_topics_df.loc[top_topics_df['health'] == 1, ['subreddit', 'topic_number']]
    top_health_topics = top_health_topics['topic_number'].value_counts().rename('subreddit_count')
    print('Health-related topics: \n{}'.format(top_health_topics))

    pd.to_pickle(top_topics_df, 'lda_topic_data_{}'.format(n_topics))

    return top_topics_df, terms_df


def perform_lda(id2word,
                doc2bow,
                corpus,
                *n_topics):
    """
    Performs LDA analysis of subreddits.

    Parameters
    ----------
    id2word
        Token ID to word mapping
    doc2bow
        Document to bag-of-words list mapping
    corpus : dataframe
        Final subreddit clustering corpus
    n_topics : int
        Total number of topics

    Returns
    -------
    (summary_all, health_topics_all) : tuple
        Contains dictionary with all results information for LDA analysis, and a dataframe
        detailing the top health topics for each trained LDA model.

    """

    summary_all = []
    sub_topics_all = []
    health_topics_all = []

    print(n_topics)

    for n in n_topics:
        print('\nLDA with {} topics...'.format(n))
        model = lda_gensim(id2word, doc2bow, n)

        # Gets coherence scores for trained LDA model
        # coherence_umass = None
        # coherence_umass = coh_model_umass.get_coherence()
        # print('Coherence score (u_mass): {}'.format(coherence_umass))
        # coherence_cv = coh_model_cv.get_coherence()
        # print('Coherence score (c_v): {}'.format(coherence_cv))

        # Gets document topics and key terms and health topics
        top_topics_df, terms_df = get_sub_topics(model, doc2bow, corpus, n)

        sub_topics = top_topics_df[['subreddit', 'health', 'topic_number']]
        sub_topics.set_index(['subreddit',
                              'health',
                              ], inplace=True,
                             )
        sub_topics.columns = pd.MultiIndex.from_product([sub_topics.columns,
                                                         [n],
                                                         ])
        sub_topics_all.append(sub_topics)

        health_sub_topics = sub_topics.loc[sub_topics.index.get_level_values(level=1) == 1].transpose()
        health_sub_topics = health_sub_topics.droplevel(level=0,
                                                        )
        health_sub_topics = health_sub_topics.droplevel('health',
                                                        axis=1,
                                                        )
        health_sub_topics.index.name = 'topic_label'
        health_topics_all.append(health_sub_topics)

        health_topics_top = top_topics_df.loc[top_topics_df['health'] == 1, 'topic_number'].value_counts().rename(
            'subreddit_count')

        # top_health = health_topics.idxmax()
        # possible_health_subs = top_topictops_df.loc[top_topics_df['topic_number'] == top_health]

        # Saves results to dictionary
        summary = {
            'n_topics': n,
            # 'coherence_score_umass': coherence_umass,
            # 'coherence_score_cv': coherence_cv,
            'top_health_topics': health_topics_top.to_dict(),
            'topic_terms': terms_df['terms_list'].apply(lambda x: ', '.join(x)).to_dict()
        }
        summary_all.append(summary)
    # sub_topics_file = pd.read_pickle('all_lda_topics.pkl')
    # sub_topics_all.append(sub_topics_file)
    sub_topics_all = pd.concat(sub_topics_all,
                               axis=1,
                               )
    sub_topics_all.to_pickle('all_lda_topics.pkl')
    health_topics_all = pd.concat(health_topics_all)
    health_topics_all.to_pickle('health_lda_topics.pkl')

    return summary_all, health_topics_all


def plot_coherence(summary,
                   min_df=params.vocab_params_default['min_df'],
                   max_df=params.vocab_params_default['max_df']):
    '''
    Plots topic coherence scores for LDA models.

    Parameters
    ----------
    summary : dict
        Details results of LDA (includes coherence scores)
    min_df : int
        Minimum document frequency limit for current feature vocabulary
    max_df : int
        Maximum document frequency limit for current feature vocabulary

    Returns
    -------

    '''

    n_topics = []
    umass = []
    cv = []
    key = 'vocab_{}_{}'.format(min_df,
                               max_df,
                               )
    for k in full_summary[key].keys():
        if k.startswith('lda'):
            print(k)
            n_topics.append(k.split('_')[1])
            umass.append(summary[key][k]['coherence_score_umass'])
            cv.append(summary[key][k]['coherence_score_cv'])
    print(n_topics)
    print(umass)
    print(cv)

    coherence_vals = pd.DataFrame([umass, cv],
                                  columns=n_topics,
                                  index=['umass', 'cv'],
                                  ).transpose()
    pd.to_pickle(coherence_vals, 'LDA_coherence_values.pkl')

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot()
    x_axis = list(coherence_vals.index)
    line_umass = ax.plot(x_axis,
                         coherence_vals['umass'],
                         label='umass',
                         color='teal',
                         linewidth=2,
                         )
    line_cv = ax.plot(x_axis,
                      coherence_vals['cv'],
                      label='cv',
                      color='maroon',
                      linewidth=2,
                      )
    plt.title('LDA topic modelling of subreddit  sample (min_df: {}, max_df: {})'.format(min_df, max_df))
    plt.xlabel('Number of topics')
    plt.ylabel('Coherence score')
    plt.legend()
    plt.savefig('LDA_coherence_values')
    plt.show()
    plt.close('all')


if __name__ == '__main__':

    # Running LDA with option to test all parameters
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
        lda_params = params.lda_params_all
    else:
        summary_file = 'sub_cluster_summary_DEFAULT.json'
        vocab_params = [params.vocab_params_default]
        tsvd_params = {'n_components': [params.tsvd_params_default['n_components']]}
        if params.debug_all_topics:
            lda_params = params.lda_params_all
        else:
            lda_params = {'n_topics': [params.lda_params_default['n_topics']]}

    subreddit_corpus = load_corpus(**sample_params)

    lda_df_all = []
    params_labels = ['min_df',
                     'max_df',
                     'tsvd_components',
                     'method',
                     'n_topics/clusters',
                     ]

    print('All TF-IDF parameters: \n{}'.format(vocab_params))
    for vocab_dict in vocab_params:
        full_summary = json.load(open(summary_file, 'r'))
        print('\nTF-IDF parameters for current test: {}'.format(vocab_dict))
        # Loads TF-IDF matrix from file
        count_vect, tfidf_matrix = load_tfidf(**vocab_dict)
        vocab_key = 'vocab_{}_{}'.format(vocab_dict['min_df'],
                                         vocab_dict['max_df'],
                                         )

        # Generates LDA vocabulary and corpus (using TF-IDF vocabulary)
        lda_dir('new')
        lda_vocab, lda_corpus, lda_corpus_text = prep_corpus_lda(count_vect, subreddit_corpus)

        if lda_vocab is None:
            pass

        else:
            lda_summary, lda_health = perform_lda(lda_vocab,
                                                  lda_corpus,
                                                  subreddit_corpus,
                                                  *lda_params['n_topics'],
                                                  )
            lda_key = 'lda'
            params_values = [[vocab_dict['min_df'],
                              vocab_dict['max_df'],
                              'n/a',
                              'lda',
                              x,
                              ] for x in lda_health.index]
            params_df = pd.DataFrame(params_values,
                                     columns=params_labels,
                                     )
            lda_df = pd.DataFrame(lda_summary)
            lda_df = pd.merge(params_df,
                              lda_df,
                              left_on='n_topics/clusters',
                              right_on='n_topics',
                              )
            lda_df_all.append(lda_df)
            full_summary[vocab_key][lda_key] = lda_summary
        '''
        # Plot coherence values for all values of n_topics
        plot_coherence(full_summary,
                       **vocab_dict,
                       )
        '''
        # Updates results JSON file
        cluster_dir(**sample_params)
        json.dump(full_summary, open(summary_file, 'w'), indent=4)

    # Saves LDA data to file
    # lda_df_all.append(lda_df_file)
    lda_df_all = pd.concat(lda_df_all)
    print(lda_df_all)
    lda_df_all.to_pickle('lda_df.pkl')
    lda_df_all.to_csv('lda_df.csv',
                      sep=';',
                      )

    print('LDA Complete.')
