#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : generate_subreddit_corpus.py
AUTHOR : Eliza Harrison

This program generates the subreddit corpus for clustering.

"""

import argparse
import html
import os
import pickle
import re

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

import config
import database_functions as db
from subreddit_clustering import subreddit_clustering_params as params
from subreddit_clustering.compile_seeding_set import load_health_subs


def cluster_dir(action='set',
                n_posts=params.subreddit_cluster_params['n_posts'],
                sample_id=params.subreddit_cluster_params['sample_id'],
                ):
    """
    Creates or sets directory for subreddit clustering
    Parameters
    ----------
    action : str
        Choose whether to create new directory ('new') or set working directory
        to existing directory ('set')
    n_posts : int
        The number of submissions sampled per subreddit
    sample_id : int
        The random state number used to randomly sample submissions for each subreddit

    Returns
    -------
    directory: str
        Filepath of subreddit clustering directory

    """

    directory = '{}/subreddit_clustering_{}_{}'.format(config.project_dir,
                                                       n_posts,
                                                       sample_id,
                                                       )
    if action == 'new':
        if not os.path.exists(directory):
            os.makedirs(directory)

    os.chdir(directory)

    return directory


def sample_files(n_posts=params.subreddit_cluster_params['n_posts'],
                 sample_id=params.subreddit_cluster_params['sample_id'], ):
    """

    Parameters
    ----------
    n_posts : int
        The number of submissions sampled per subreddit
    sample_id : int
        The random state number used to randomly sample submissions for each subreddit

    Returns
    -------
    (raw_corpus_file, clean_corpus_file) : tuple
        Strings corresponding to the filenames of the subreddit clustering dataset before (...RAW)
        and after pre-processing (...CLEAN)

    """

    raw_corpus_file = 'subreddit_clustering_dataset_RAW_{}_{}.pkl'.format(n_posts,
                                                                          sample_id,
                                                                          )
    clean_corpus_file = 'subreddit_clustering_dataset_CLEAN_{}_{}.pkl'.format(n_posts,
                                                                              sample_id,
                                                                              )

    return raw_corpus_file, clean_corpus_file


def load_posts_from_pkl():
    """
    Loads subreddit name and post fullname (ID) data from pickle files on disk.
    Each file contains all posts for a particular source file.

    Returns
    -------
    all_posts : DataFrame
        Contains all post fullnames for each subreddit

    """

    # Loads post data from individual pickle files
    all_posts = []
    path = config.post_data_by_source_file
    pkl_files = os.listdir(path)
    pkl_files.sort(reverse=True)
    print('Loading post IDs from pickle files...')
    for pkl_file in pkl_files:
        print(pkl_file)
        data = pd.read_pickle('{}/{}'.format(path, pkl_file))
        all_posts.append(data)

    all_posts = pd.concat(all_posts)

    return all_posts


def gen_random_sample(engine,
                      n_posts=params.subreddit_cluster_params['n_posts'],
                      sample_id=params.subreddit_cluster_params['sample_id'],
                      ):
    """
    Downloads all subreddits with >= n posts in the last year.
    Randomly selects n posts per subreddit.

    Parameters
    ----------
    engine : SQLAlchemy engine
        Connection to database in which Reddit data is stored
    n_posts : int
        The number of submissions sampled per subreddit
    sample_id : int
        The random state number used to randomly sample submissions for each subreddit

    Returns
    -------
    post_ids : dataframe
    dataframe containing subreddits with n randomly sampled submission (post) fullnames,
    columns ['subreddit', 'fullname']

    """

    print('Generating random sample of {} posts per subreddit...'.format(n_posts))
    sample_filename = 'post_ids_{}_{}.pkl'.format(n_posts,
                                                  sample_id,
                                                  )

    try:
        post_ids = pd.read_pickle(sample_filename)

    except (OSError, IOError, FileNotFoundError) as e:
        # Downloads list of subreddits with n+ posts per year
        sql = ("SELECT "
               "display_name as subreddit, "
               "url as sub_url, "
               "post_count "
               "FROM all_subreddits "
               "WHERE post_count >= {}".format(n_posts))
        print('Downloading subreddits with at least {} submissions per year from database...'.format(n_posts))
        subs = pd.read_sql(sql,
                           engine,
                           )
        subs.set_index('subreddit', inplace=True)
        print('{} subreddits in sample dataset'.format(len(subs)))
        print(subs)

        # Retrieves IDs for all post in each subreddit
        all_post_ids = load_posts_from_pkl()
        # all_post_ids = all_post_ids[['subreddit', 'fullname']]

        # Deletes those posts made in subreddits with less than n posts in the last year
        all_post_ids = all_post_ids.loc[all_post_ids['subreddit'].isin(subs.index)]

        # Randomly select n posts per subreddit
        print('Randomly sampling {} submissions per subreddit...'.format(n_posts))
        post_ids = all_post_ids.groupby('subreddit').apply(lambda x: x.sample(n=n_posts,
                                                                              random_state=sample_id,
                                                                              ))
        post_ids = post_ids.join(subs['sub_url'], how='left')
        post_ids = post_ids[['subreddit', 'sub_url', 'fullname']]
        post_ids.index = post_ids.index.droplevel(1)
        post_ids.drop('subreddit',
                      axis=1,
                      inplace=True,
                      )
        post_ids.to_pickle(sample_filename)
    print(post_ids)
    print('{} submissions in sample dataset'.format(len(post_ids)))

    return post_ids


def get_corpus_text(engine, post_ids, raw_corpus_file):
    """
    Downloads title and selftext from database for each post in random sample
    
    Parameters
    ----------
    engine : SQLAlchemy engine
        Connection to database in which Reddit data is stored
    post_ids : dataframe
        Contains columns ['subreddit', 'fullname'] where fullname is ID for each submission (post)
        in the randomly sampled set
    raw_corpus_file : str
        Filename for subreddit sample dataset prior to pre-processing

    Returns
    -------
    subreddit_text: dataframe
        Grouped by subreddit with post title and selftext concatenated
        into single string

    """
 
    print('Downloading title and selftext of posts in random sample from database...')
    sql = (
        "SELECT "
        "subreddit, "
        "fullname, "
        "CONCAT(title, ', ', selftext) as concat_text "
        "FROM all_posts "
        "WHERE fullname IN ('{}') ".format("', '".join(post_ids['fullname']))
    )
    post_text = pd.read_sql(sql,
                            engine,
                            )

    '''
    print('Concatenating titles and selftext of each post...')
    post_text['concat_text'] = post_text[['title', 'selftext']].apply(lambda x: ' '.join(x).strip(),
                                                                      axis=1,
                                                                      )
    '''

    print('Concatenating posts for each subreddit into single string...')
    subreddit_text = post_text.groupby(['subreddit'])['concat_text'].apply(lambda x: ', '.join(x).strip())
    post_id_list = post_text.groupby(['subreddit'])['fullname'].apply(lambda x: x.tolist())
    subreddit_text = pd.concat([subreddit_text,
                          post_id_list,
                          ],
                         axis=1,
                         ).reset_index()
    subreddit_text.columns = ['subreddit', 'text', 'posts']
    print('Raw subreddit corpus data: \n{}'.format(subreddit_text))
    print(subreddit_text.columns)

    print('Saving raw sample text to pickle file...')
    pd.to_pickle(subreddit_text, raw_corpus_file)

    return subreddit_text


def text_level_cleaning(raw_sub_corpus):
    """
    Cleans the text strings for each subreddit.
    Removes URLs (starting with https and www), HTML entities, punctuation, double whitespace.
    
    Parameters
    ----------
    raw_sub_corpus : dataframe
        Raw (unprocessed) dataset for subreddit clustering

    Returns
    -------
    sub_sample_df : dataframe
        Contains additional column ('clean_text') containing processed text representing each subreddit

    """
    
    # Encoding to ASCII
    # sub_sample_df['clean_text'] = sub_sample_df['text'].apply(lambda x: str(x.encode('ascii', 'ignore')))
    # print('Removing emoji characters...')
    # sub_sample_df['clean_text'] = sub_sample_df['clean_text'].apply(lambda x: emoji.get_emoji_regexp().sub(r'', x))

    print('Converting to lowercase...')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['text'].apply(lambda x: x.lower())

    print('Removing newlines...')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: x.replace('\n', ' '))

    print('Removing tabs...')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: x.replace('\t', ' '))

    print('Removing URLs...')
    http_regex = re.compile(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: re.sub(http_regex, ' ', x))
    www_regex = re.compile(r'www(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: re.sub(www_regex, ' ', x))

    print('Removing HTML entities...')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: html.unescape(x))

    print('Removing numeric-only words...')
    r_numbers = re.compile(r'\b[0-9]+\b')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: re.sub(r_numbers, ' ', x))

    print('Applying Scikit-Learn tokenizer regex to remove punctuation...')
    r_sklearn = re.compile('[^\b\w\w+\b]')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: re.sub(r_sklearn, ' ', x))

    print('Removing double whitespace...\n')
    r_whitespace = re.compile(r' +')
    raw_sub_corpus['clean_text'] = raw_sub_corpus['clean_text'].apply(lambda x: re.sub(r_whitespace, ' ', x))

    print(raw_sub_corpus['clean_text'])
    raw_sub_corpus.drop('text', axis=1, inplace=True)

    return raw_sub_corpus


def subreddit_level_cleaning(clean_sub_corpus):
    """
    Cleans the sample of subreddits to identify and exclude those with too few words for analysis.
    Calculates summary statistics for subreddit dataset.

    Parameters
    ----------
    clean_sub_corpus : dataframe
        Subreddit dataset after processing of text representing each subreddit

    Returns
    -------
    final_sub_corpus : dataframe
        Subreddit dataset after removal of subreddits with insufficient text for efficient clustering

    """

    word_counts = clean_sub_corpus[['subreddit', 'clean_text']]
    word_counts['word_list'] = word_counts['clean_text'].apply(lambda x: list(x.split()))
    word_counts['total_word_count'] = word_counts['word_list'].apply(lambda x: len(x))
    word_counts['distinct_word_list'] = word_counts['clean_text'].apply(lambda x: list(dict.fromkeys(x.split())))
    word_counts['distinct_word_count'] = word_counts['distinct_word_list'].apply(lambda x: len(x))

    shortest_sub = word_counts.loc[word_counts['total_word_count'] == word_counts['total_word_count'].min(),
                                   ['subreddit', 'total_word_count']]
    longest_sub = word_counts.loc[word_counts['total_word_count'] == word_counts['total_word_count'].max(),
                                  ['subreddit', 'total_word_count']]
    least_distinct_words = word_counts.loc[word_counts['total_word_count'] == word_counts['total_word_count'].min(),
                                           ['subreddit', 'total_word_count']]
    most_distinct_words = word_counts.loc[word_counts['total_word_count'] == word_counts['total_word_count'].max(),
                                          ['subreddit', 'total_word_count']]
    shortest_total = word_counts.loc[word_counts['total_word_count'] < 100]
    shortest_distinct = word_counts.loc[word_counts['distinct_word_count'] < 100]

    print('Analysis of subreddit words: \n'
          '> Length of SHORTEST subreddit: \n{}\n'.format(shortest_sub),
          '> Length of LONGEST subreddit: \n{}\n'.format(longest_sub),
          '> Subreddit with LEAST distinct words: \n{}\n'.format(least_distinct_words),
          '> Subreddit with MOST distinct words: \n{}\n'.format(most_distinct_words),
          '> Subreddits with < 100 total words: \n{}\n'.format(shortest_total.loc[:, ['subreddit',
                                                                                      'clean_text',
                                                                                      'total_word_count',
                                                                                      ]].sort_values(
              'total_word_count')),
          '> Subreddits with < 100 distinct words): \n{}\n'.format(shortest_distinct.loc[:, ['subreddit',
                                                                                             'clean_text',
                                                                                             'distinct_word_count',
                                                                                             ]].sort_values(
              'distinct_word_count')),
          )
    print('Total word count summary (ORIGINAL CORPUS): \n{}'.format(word_counts['total_word_count'].describe()))
    print('Distinct word count summary (ORIGINAL CORPUS): \n{}'.format(word_counts['distinct_word_count'].describe()))

    excluded_subs = word_counts.loc[word_counts['distinct_word_count'] < 100]
    print(excluded_subs)

    word_counts_reduced = word_counts.loc[~word_counts.index.isin(excluded_subs.index)]
    final_sub_corpus = clean_sub_corpus.loc[~clean_sub_corpus.index.isin(excluded_subs.index)].reset_index(drop=True)

    print('Total word count summary (FINAL CORPUS): \n{}'.format(word_counts_reduced['total_word_count'].describe()))
    print('Distinct word count summary (FINAL CORPUS): \n{}'.format(
        word_counts_reduced['distinct_word_count'].describe()))

    health = clean_sub_corpus.loc[clean_sub_corpus['health'] == 1, 'subreddit']
    word_counts_health = word_counts_reduced.loc[word_counts_reduced['subreddit'].isin(health)]

    word_counts.drop('clean_text', axis=1, inplace=True)
    # word_counts.to_pickle('corpus_w_word_counts.pkl')

    return final_sub_corpus


def gen_full_vocab(corpus_text,
                   n_posts=params.subreddit_cluster_params['n_posts'],
                   sample_id=params.subreddit_cluster_params['sample_id'],
                   ):
    """
    Uses sklearn's CountVectorizer to extract all unigrams (terms) in the final subreddit corpus.
    Uses minimum document frequency of 2 to exclude terms present in only one subreddit

    Parameters
    ----------
    corpus_text : series
        Series containing strings representing each subreddit in the final corpus for clustering
    n_posts : int
        The number of submissions sampled per subreddit
    sample_id : int
        The random state number used to randomly sample submissions for each subreddit

    Returns
    -------
    vocabulary_full : list
        List of all unique unigrams in the subreddit corpus

    """

    vectorizer = CountVectorizer(lowercase=True,
                                 min_df=2,
                                 max_df=0.6,
                                 )
    print('Generating count vector matrix (full vocab)...')
    x = vectorizer.fit_transform(corpus_text)
    vocabulary_full = pd.Series(vectorizer.get_feature_names())
    print('{} distinct terms in corpus (no processing or document frequency limits)'.format(len(vocabulary_full)))

    vocab_file = 'subreddit_vocab_full_{}_{}.pkl'.format(n_posts,
                                                         sample_id,
                                                         )
    pickle.dump(vocabulary_full, open(vocab_file, 'wb'))

    return vocabulary_full


def corpus_health_subs(corpus):
    """
    Labels subreddits in the corpus based on whether they are 'known health' subreddits
        0 = Not in seeding sample of known health subreddits
        1 = In the seeding sample of known health subreddits

    Parameters
    ----------
    corpus : dataframe
        Final subreddit corpus

    Returns
    -------
    corpus_w_health : dataframe
        Contains additional column ('health') indicating which subreddits are in the seeding sample

    """

    # Loads known health subreddits (seeding set) from file
    health_subs = load_health_subs()
    print([sub for sub in health_subs['display_name']])

    health_df = pd.DataFrame(corpus.loc[corpus['subreddit'].isin(health_subs['display_name']), 'subreddit'])
    health_df['health'] = 1
    non_health_df = pd.DataFrame(corpus.loc[~corpus['subreddit'].isin(health_subs['display_name']), 'subreddit'])
    non_health_df['health'] = 0
    corpus_w_health = pd.concat([health_df, non_health_df], axis=0).sort_index()
    corpus_w_health = corpus.merge(corpus_w_health, how='left', on='subreddit')

    print(corpus_w_health.loc[corpus_w_health['health'] == 1])
    print('Corpus indices of known health-related subreddits: \n{}'.format(health_df['subreddit']))
    print('Corpus indices of all other subreddits: \n{}'.format(non_health_df['subreddit']))

    return corpus_w_health


def load_corpus(n_posts=params.subreddit_cluster_params['n_posts'],
                sample_id=params.subreddit_cluster_params['sample_id']):
    """

    Parameters
    ----------
    n_posts : int
        The number of submissions sampled per subreddit
    sample_id : int
        The random state number used to randomly sample submissions for each subreddit

    Returns
    -------

    """

    print('Subreddit sample parameters: \n',
          '> Number of posts per subreddit: {}\n'.format(n_posts),
          '> Sample ID: {}\n'.format(sample_id),
          )
    cluster_dir('set', n_posts, sample_id)
    raw_corpus_file, clean_corpus_file = sample_files(n_posts, sample_id)

    corpus = pd.read_pickle(clean_corpus_file)

    return corpus


def health_sub_to_db(corpus_w_health, tablename):
    """
    Uploads health subreddits (seeding sample) to database

    Parameters
    ----------
    corpus_w_health : dataframe
        Final subreddit corpus with column 'health' indicating whether subreddit is in seeding set
    tablename : str
        Name of database table

    Returns
    -------

    """

    health_sub_names = corpus_w_health.loc[corpus_w_health['health'] == 1, 'subreddit']
    print(health_sub_names)

    sql = ("SELECT "
           "* "
           "FROM dbreddit.public.all_subreddits "
           "WHERE display_name IN ('{}') ".format("', '".join(health_sub_names))
           )

    # Connects to dbreddit database using SSH tunnel
    tunnel, engine = db.db_connect()

    health_sub_data = pd.read_sql(sql, engine)
    db.df_to_db(health_sub_data, tablename, engine, if_exists='replace')

    # Closes database connection
    db.db_close_conn(tunnel, engine)

    return health_sub_data


def health_post_to_db(corpus_w_health, tablename):
    """
        Uploads submissions posted to health subreddits (seeding sample) to database

        Parameters
        ----------
        corpus_w_health : dataframe
            Final subreddit corpus with column 'health' indicating whether subreddit is in seeding set
        tablename : str
            Name of database table

        Returns
        -------

        """

    health_sub_names = corpus_w_health.loc[corpus_w_health['health'] == 1, 'subreddit']
    sql = ("SELECT "
           "* "
           "FROM dbreddit.public.all_posts "
           "WHERE subreddit IN ('{}') ".format("', '".join(health_sub_names))
           )

    # Connects to dbreddit database using SSH tunnel
    tunnel, engine = db.db_connect()

    health_sub_data = pd.read_sql(sql, engine)
    db.df_to_db(health_sub_data, tablename, engine, if_exists='replace')

    # Closes database connection
    db.db_close_conn(tunnel, engine)


if __name__ == '__main__':
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
    args = parser.parse_args()

    sample_params = params.subreddit_cluster_params

    if args.nposts != 100:
        sample_params['n_posts'] = args.nposts
    if args.randomstate != 1:
        sample_params['sample_id'] = args.sampleid

    n_posts = params.subreddit_cluster_params['n_posts']
    sample_id = params.subreddit_cluster_params['sample_id']

    # Generates subreddit corpus from randomly sampled set of posts
    cluster_dir('new',
                n_posts,
                sample_id,
                )
    raw_corpus_file, clean_corpus_file = sample_files(n_posts, sample_id)

    print('Number of posts randomly selected for each subreddit: {}'.format(n_posts))
    print('Version of dataset (random state value for sampling algorithm): {}'.format(sample_id))

    try:
        corpus_w_health = pd.read_pickle(clean_corpus_file)
    except (FileNotFoundError, EOFError):
        try:
            raw_corpus = pd.read_pickle(raw_corpus_file)

        except (FileNotFoundError, EOFError):
            # Connects to dbreddit database using SSH tunnel
            tunnel, engine = db.db_connect()

            # Randomly selects n posts from those subreddits with n+ posts in the last year
            sample_post_ids = gen_random_sample(engine,
                                                n_posts=n_posts,
                                                sample_id=sample_id,
                                                )

            # Generates subreddit corpus
            raw_corpus = get_corpus_text(engine, sample_post_ids, raw_corpus_file)

            # Closes database connection
            db.db_close_conn(tunnel, engine)

            # Cleans text for each subreddit
            print(raw_corpus)

        clean_corpus = text_level_cleaning(raw_corpus)

        # Cleans subreddits (based on word counts)
        clean_corpus = subreddit_level_cleaning(clean_corpus)

        # Assigns health label to known health subs
        corpus_w_health = corpus_health_subs(clean_corpus)

        # Saves cleaned sample_50_1 sample to database and pickle file
        print('Saving sample to pickle file...')
        print('\n')
        pd.to_pickle(corpus_w_health, clean_corpus_file)

        vocab_full = gen_full_vocab(clean_corpus['clean_text'])

    # Upload details of seeding sample to database
    all_health_sub_data = health_sub_to_db(corpus_w_health,
                                           'health_subreddits_{}_{}'.format(n_posts,
                                                                            n_posts,
                                                                            ))

    # Upload details of submissions posted to subreddits in the seeding sample to database
    health_post_to_db(corpus_w_health,
                      'health_posts_{}_{}'.format(n_posts,
                                                  sample_id,
                                                  ))

    print('Subreddit sample generation complete.')
