# !/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : generate_thread_dataset.py
AUTHOR : Eliza Harrison

This program processes the labelled thread dataset in preparation for the
classification of threads containing health claims.

"""

import html
import os
import re

import nltk
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords, wordnet as wn
from scipy.stats import fisher_exact
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from wordcloud import WordCloud

import database_functions as db

'''
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
'''

sentence_loc_dict = {
    'subreddit': [0, 'sss'],
    'post_title': [1, 'ttt'],
    'post_description': [2, 'ddd'],
    'comment_1': [3, 'ccc'],
    'comment_2': [4, 'ccc'],
    'comment_3': [5, 'ccc'],
    'comment_4': [6, 'ccc'],
    'comment_5': [7, 'ccc'],
}

colour_map = LinearSegmentedColormap.from_list('pink_blue', ['#43081e',
                                                             '#6c1d33',
                                                             '#983248',
                                                             '#b95857',
                                                             '#d67e67',
                                                             '#f4d5b1',
                                                             # '#ffffe3',
                                                             '#c3e2d2',
                                                             '#89c5be',
                                                             '#5ca6a5',
                                                             '#3f8087',
                                                             '#2c6568',
                                                             '#1c484a',
                                                             '#0e2b2d'],
                                               N=78)

label_votes_sql = ("SELECT * FROM agreed_yes_thread "
                   "UNION "
                   "SELECT * FROM agreed_no_thread "
                   "UNION "
                   "SELECT * FROM majority_yes_thread "
                   "UNION "
                   "SELECT * FROM majority_no_thread "
                   )

test_qs_sql = "SELECT * FROM all_results_thread WHERE send_to_mturk = 2 "
worker_scores_sql = "SELECT * FROM worker_validation "

ngram_names = ('UNIGRAM',
               'BIGRAM',
               'TRIGRAM',
               )


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def stopwords_and_lemmatize(sentence):
    stop_words = stopwords.words('english')
    lemma = WordNetLemmatizer()

    nltk_tagged = nltk.pos_tag([word for word in word_tokenize(sentence)
                                if word.lower() not in stop_words])
    wn_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wn_tagged:
        if tag is None:
            lemmatized_sentence.append(word)
        else:
            lemmatized_sentence.append(lemma.lemmatize(word, tag))

    lemmatized_sentence = [word for word in lemmatized_sentence if word not in stop_words]

    return ' '.join(lemmatized_sentence)


def clean_text(sentences_df):
    print('Original corpus text: \n{}\n'.format(sentences_df['sentence_text']))

    print('Removing URLs...')
    r_http = re.compile(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    sentences_df['clean_sentence_text'] = sentences_df['sentence_text'].apply(lambda x: re.sub(r_http, ' ', x))
    r_www = re.compile(r'www(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    sentences_df['clean_sentence_text'] = sentences_df['clean_sentence_text'].apply(lambda x: re.sub(r_www, ' ', x))

    print('Removing HTML entities...')
    sentences_df['clean_sentence_text'] = sentences_df['clean_sentence_text'].apply(lambda x: html.unescape(x))

    print('Removing numeric-only words...')
    r_numbers = re.compile(r'\b[0-9]+\b')
    sentences_df['clean_sentence_text'] = sentences_df['clean_sentence_text'].apply(lambda x: re.sub(r_numbers, ' ', x))

    print('Removing /r/ from subreddit name...')
    r_regex = re.compile(r'\/r\/')
    sentences_df['clean_sentence_text'] = sentences_df['clean_sentence_text'].apply(lambda x: re.sub(r_regex, '', x))

    '''
    print('Removing single-letter terms...')
    r_single = re.compile(r'\b\w\b\s')
    sentences_df['clean_sentence_text'] = sentences_df['clean_sentence_text'].apply(lambda x: re.sub(r_single, ' ', x))
    '''

    print('Applying Scikit-Learn tokenizer regex to remove punctuation...')
    r_sklearn = re.compile('[^\b\w\w+\b]')
    sentences_df['clean_sentence_text'] = sentences_df['clean_sentence_text'].apply(lambda x: re.sub(r_sklearn, ' ', x))

    print('POS tagging and lemmatizing text...')
    sentences_df['lemmatized_text'] = sentences_df['clean_sentence_text'].apply(lambda x: stopwords_and_lemmatize(x))

    print('Removing double whitespace...')
    r_whitespace = re.compile(r' +')
    sentences_df['lemmatized_text'] = sentences_df['lemmatized_text'].apply(
        lambda x: re.sub(r_whitespace, ' ', x))

    print('\nCleaned and lemmatized corpus text: \n{}\n'.format(sentences_df['lemmatized_text']))

    return sentences_df


def combine_sentences(sentence_df_clean):
    sentence_df_clean.sort_values(['health_claim',
                                   'subreddit',
                                   'post_fullname',
                                   'sentence_location',
                                   'sentence_order',
                                   ],
                                  inplace=True,
                                  )
    sentence_df_clean.reset_index(drop=True,
                                  inplace=True,
                                  )

    prefix_mapping = dict(sentence_loc_dict.values())
    sentence_df_clean['pos_prefix'] = sentence_df_clean['sentence_location'].map(prefix_mapping)
    sentence_df_clean['text_w_pos'] = sentence_df_clean.apply(
        lambda x: ' '.join([x['pos_prefix'] + term for term in x['lemmatized_text'].split(' ')]),
        axis=1,
    )

    grouped = sentence_df_clean.groupby(['post_fullname', 'sentence_location'])
    sentences_concat = grouped['lemmatized_text', 'text_w_pos'].agg(' '.join).reset_index()

    for_merge = sentence_df_clean[['post_fullname', 'sentence_location'] +
                                  [col for col in sentence_df_clean.columns
                                   if col not in sentences_concat.columns]]
    for_merge.drop_duplicates(['post_fullname', 'sentence_location'],
                              inplace=True,
                              )
    final_df = sentences_concat.merge(for_merge,
                                      how='left',
                                      on=['post_fullname',
                                          'sentence_location',
                                          ])
    final_df.rename({'lemmatized_text': 'text',
                     'sentence_location': 'text_location'},
                    inplace=True,
                    axis=1,
                    )
    final_df = final_df[['health_claim',
                         'send_to_mturk',
                         'subreddit',
                         'post_fullname',
                         'comment_fullname',
                         'text_location',
                         'text',
                         'text_w_pos',
                         ]].reset_index()

    return final_df


def combine_threads(location_df):
    grouped = location_df.groupby(['post_fullname'])
    thread_concat = grouped['text', 'text_w_pos'].agg(' '.join).reset_index()
    for_merge = location_df[['post_fullname'] + [col for col in location_df.columns
                                                 if col not in thread_concat.columns]]
    for_merge.drop_duplicates('post_fullname',
                              inplace=True,
                              )
    final_df = thread_concat.merge(for_merge,
                                   how='left',
                                   on='post_fullname',
                                   )
    final_df = final_df[['health_claim',
                         'send_to_mturk',
                         'subreddit',
                         'post_fullname',
                         'text',
                         'text_w_pos',
                         ]].reset_index(drop=True)
    return final_df


def gen_ngrams(thread_data,
               min_df=2,
               ngrams=(1, 3),
               positional=False, ):
    print('Minimum document frequency: {}'.format(min_df))
    print('n-grams: {}'.format(ngrams))
    if positional:
        X_col = 'text_w_pos'
    else:
        X_col = 'text'

    Y_col = 'health_claim'

    # Total ngram frequency (frequency vectorization)
    freq_vect = CountVectorizer(ngram_range=ngrams,
                                min_df=min_df,
                                )
    ngram_freq_matrix = freq_vect.fit_transform(thread_data[X_col])
    print('Frequency vocabulary shape: {}'.format(ngram_freq_matrix.shape))

    # Number of documents in which ngram appears (binary vectorization)
    binary_vect = CountVectorizer(ngram_range=ngrams,
                                  min_df=min_df,
                                  binary=True,
                                  )
    ngram_binary_matrix = binary_vect.fit_transform(thread_data[X_col])
    print('Binary vocabulary shape: {}'.format(ngram_freq_matrix.shape))

    # Get vocabulary and transform matrices into DataFrame
    vocab_dict = dict(zip(list(freq_vect.vocabulary_.values()),
                          list(freq_vect.vocabulary_.keys())))
    vocabulary_df = pd.DataFrame(pd.Series(vocab_dict).sort_index(),
                                 columns=['ngram'],
                                 )
    vocabulary_df['n'] = vocabulary_df['ngram'].apply(lambda x: len(x.split(' ')))

    ngram_freq_df = pd.DataFrame(ngram_freq_matrix.toarray())
    ngram_binary_df = pd.DataFrame(ngram_binary_matrix.toarray())
    for df in [ngram_freq_df, ngram_binary_df]:
        df[Y_col] = thread_data[Y_col]

    # Calculate ngram frequencies for classes
    unique_ngrams = len(vocabulary_df)
    ngram_class_freq = ngram_freq_df.groupby(Y_col).sum().transpose()
    ngram_class_binary = ngram_binary_df.groupby(Y_col).sum().transpose()
    for c in [0, 1]:
        vocabulary_df['{}_total_freq'.format(c)] = ngram_class_freq[c]
        vocabulary_df['{}_doc_freq'.format(c)] = ngram_class_binary[c]
    vocabulary_df['total_freq'] = vocabulary_df['0_total_freq'] + vocabulary_df['1_total_freq']
    vocabulary_df['doc_freq'] = vocabulary_df['0_doc_freq'] + vocabulary_df['1_doc_freq']
    vocabulary_df.sort_values('total_freq',
                              ascending=False,
                              inplace=True,
                              )

    return vocabulary_df


def contingency_table(x, total_pos, total_neg):
    pos_in = x['1_doc_freq']
    pos_out = total_pos - x['1_doc_freq']
    neg_in = x['0_doc_freq']
    neg_out = total_neg - x['0_doc_freq']

    table = [[pos_in, pos_out],
             [neg_in, neg_out]]

    return table


def fishers_exact_test(vocabulary_df, thread_data):
    """
                                1
                        pos_in    pos_not_in
            neg_in      
    0
            neg_not_in

    [['pos_in', 'pos_not_in'],
     ['neg_in', 'neg_not_in']]

    pos_in = x['1_doc_freq']
    pos_out = total_pos - pos_in
    neg_in = x[0_doc_freq]
    neg_out = total_neg - neg_in

    :param vocabulary_df:
    :param thread_data:
    :return:
    """

    # Performs fisher exact test to determine the probability of each n-gram occuring in each class
    total_pos = len(thread_data.loc[thread_data['health_claim'] == 1])
    total_neg = len(thread_data.loc[thread_data['health_claim'] == 0])
    vocabulary_df['p_value'] = vocabulary_df.apply(lambda x: float(fisher_exact(contingency_table(x,
                                                                                                  total_pos,
                                                                                                  total_neg, ))[1]),
                                                   axis=1, )

    return vocabulary_df


def predict_sig_ngrams(vocabulary_df, p_value=0.05):
    sig_ngrams_positive = []
    sig_ngrams_negative = []
    for n, gram in enumerate(ngram_names, 1):
        print('\n', n, gram)
        sig_ngrams = vocabulary_df.loc[(vocabulary_df['p_value'] < p_value) &
                                       (vocabulary_df['n'] == n)]
        sig_ngrams.sort_values('p_value',
                               inplace=True, )
        print('Significantly discriminative {}: \n {}'.format(gram,
                                                              sig_ngrams, ))
        predict_pos_class = sig_ngrams.loc[sig_ngrams['1_doc_freq'] > sig_ngrams['0_doc_freq']]
        predict_neg_class = sig_ngrams.loc[sig_ngrams['0_doc_freq'] > sig_ngrams['1_doc_freq']]
        sig_ngrams_positive.append(predict_pos_class)
        sig_ngrams_negative.append(predict_neg_class)
        '''
        if len(predict_pos_class) > 0:
            sig_ngrams.to_pickle('./significant_{}grams.pkl'.format(n))
        '''
    sig_ngrams_positive = pd.concat(sig_ngrams_positive)
    print('N-Grams predicting positive class (p < {}): \n{}'.format(p_value,
                                                                    sig_ngrams_positive, ))
    sig_ngrams_negative = pd.concat(sig_ngrams_negative)
    print('N-Grams predicting negative class (p < {}): \n{}'.format(p_value,
                                                                    sig_ngrams_negative, ))

    return sig_ngrams_positive, sig_ngrams_negative


def uni_bigrams_cloud(ngrams, random_state=12):
    names = ('Unigrams',
             'Bigrams', )
    random_states = (123,
                     1234, )
    min_font_size = (10,
                     8, )
    for n, grams in enumerate(ngrams, 1):
        fig, ax = plt.subplots()
        # ax = fig.add_subplot(2, 1, n)
        ax.title.set_text(names[n - 1])
        cloud = WordCloud(background_color='white',
                          width=600,
                          height=300,
                          max_words=len(grams),
                          colormap=colour_map,
                          random_state=random_states[n - 1],
                          relative_scaling=0.1,
                          max_font_size=80,
                          min_font_size=min_font_size[n - 1],
                          ).generate_from_frequencies(grams.to_dict()['p_value'])
        ax.imshow(cloud)
        ax.axis('off', interpolation='bilinear')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.rcParams.update({'font.size': 10})
        plt.savefig('sig_ngrams_cloud_{}.png'.format(n), bbox='tight')
        plt.show()


def unique_pos_ngrams_cloud(unique_ngrams, random_state=12343):
    fig = plt.figure(figsize=(8, 4))
    cloud = WordCloud(background_color='white',
                      width=800,
                      height=400,
                      max_words=len(unique_ngrams),
                      colormap=colour_map,
                      random_state=random_state,
                      relative_scaling=1,
                      max_font_size=80,
                      ).generate_from_frequencies(unique_ngrams.to_dict()['1_doc_freq'])
    plt.imshow(cloud)
    plt.axis('off', interpolation='bilinear')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    plt.rcParams.update({'font.size': 10})
    plt.savefig('health_claim_only_ngram_cloud.png', bbox='tight')
    plt.show()


def health_claims_tsne(vocab, thread_data):
    tfidf = TfidfVectorizer(min_df=2, )
    feature_matrix = tfidf.fit_transform(thread_data['text'])
    tsne = TSNE(n_components=2, metric='cosine', random_state=123)
    tsne_array = tsne.fit_transform(feature_matrix)

    neg = thread_data.loc[thread_data['health_claim'] == 0]
    pos = thread_data.loc[thread_data['health_claim'] == 1]
    neg_array = tsne_array[neg.index]
    pos_array = tsne_array[pos.index]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    ax.scatter(neg_array[:, 0],
               neg_array[:, 1],
               lw=0,
               s=30,
               c='#C92D39',  # Light grey
               )
    ax.scatter(pos_array[:, 0],
               pos_array[:, 1],
               lw=0,
               s=30,
               c='#19967D',  # Light grey
               )
    ax.axis('tight')
    ax.axis('off')
    # plt.savefig(filename)
    plt.show()
    plt.close('all')


if __name__ == '__main__':
    os.chdir('./health_claims_classification')

    # DOWNLOAD LABELLED DATASET #
    ssh, sql_engine = db.db_connect()

    print('Downloading posts labelled for presence of health claims...')
    text_by_sentence = pd.read_sql(label_votes_sql, sql_engine)
    print('Complete')

    # TEXT PRE-PROCESSING #
    # Cleaning text (removing URLs, HTMLs, stopwords and lemmatizing)
    text_by_sentence_clean = clean_text(text_by_sentence)

    # Text broken up by location in the thread (post title separate from post description etc.)
    text_by_location = combine_sentences(text_by_sentence_clean)
    # text_by_location.to_pickle('/health_claim_classification/text_by_location.pkl')

    # Text for entire threads concatenated into single string
    text_by_thread = combine_threads(text_by_location)
    text_by_thread.sort_values('health_claim',
                               inplace=True,
                               )
    text_by_thread.reset_index(inplace=True,
                               drop=True,
                               )

    # Ensure all test question are labelled 1 (health claim)
    test_threads = text_by_thread.loc[text_by_thread['send_to_mturk'] == 2]
    text_by_thread.loc[test_threads.index, 'health_claim'] = 1
    print(text_by_thread.loc[text_by_thread['send_to_mturk'] == 2])

    # Drop test questions (used to weight worker responses) from text_by_thread
    text_by_thread_no_test = text_by_thread.drop(text_by_thread.loc[text_by_thread['send_to_mturk'] == 2].index)

    # Summary of term distributions between classes
    health_claim = text_by_thread_no_test.loc[text_by_thread_no_test['health_claim'] == 1]
    no_health_claim = text_by_thread_no_test.loc[text_by_thread_no_test['health_claim'] == 0]
    print('Threads w/ health claim: {} (+ 20 test threads)'.format(len(health_claim)))
    print('Threads w/o health claim: {}'.format(len(no_health_claim)))
    print('Percentage of threads with health claim: {}'.format(
        round((len(health_claim) / (len(health_claim) + len(no_health_claim))) * 100, 3)))
    print('\nCleaned & lemmatized corpus text (by thread): \n{}'.format(text_by_thread_no_test['text']))

    text_by_thread['total_words'] = text_by_thread['text'].apply(lambda x: len(x.split(' ')))
    text_by_thread['distinct_words'] = text_by_thread['text'].apply(lambda x: len(set(x.split(' '))))
    total_word_summary = text_by_thread.groupby('health_claim')['total_words'].describe()
    distinct_word_summary = text_by_thread.groupby('health_claim')['distinct_words'].describe()
    print('\nTotal words statistical summary: \n{}'.format(total_word_summary))
    print('Distinct words statistical summary: \n{}'.format(distinct_word_summary))

    text_by_thread_no_test.to_pickle('./text_by_thread_no_test.pkl')
    text_by_thread.to_pickle('./text_by_thread.pkl')

    text_by_thread_no_test = pd.read_pickle('./text_by_thread_no_test.pkl')
    text_by_thread = pd.read_pickle('./text_by_thread.pkl')

    # PREDICTIVE N-GRAMS ANALYSIS #
    print('\nN-GRAMS ONLY...')
    vocab_df = gen_ngrams(text_by_thread,
                          positional=False, )
    vocab_w_p = fishers_exact_test(vocab_df, text_by_thread)
    sig_pos_grams, sig_neg_grams = predict_sig_ngrams(vocab_df, 0.01)
    all_sig_ngrams = pd.concat([sig_pos_grams, sig_neg_grams], axis=0)
    all_sig_ngrams.to_csv('significant_ngrams.csv')

    # Visualise significant uni- and bi-grams
    sig_posgrams_4_vis = sig_pos_grams.loc[sig_pos_grams['n'].isin([1, 2]), ['ngram',
                                                                             'n',
                                                                             'p_value',
                                                                             '1_doc_freq', ]].set_index('ngram')
    sig_posgrams_4_vis['p_value'] = 1 - sig_posgrams_4_vis['p_value']
    uni_bigrams = (sig_posgrams_4_vis.loc[sig_posgrams_4_vis['n'] == 1],
                   sig_posgrams_4_vis.loc[sig_posgrams_4_vis['n'] == 2])
    uni_bigrams_cloud(uni_bigrams,
                      random_state=12345, )

    # Visualise n-grams found only in threads with a health claim
    only_pos_grams = all_sig_ngrams.loc[all_sig_ngrams['0_doc_freq'] == 0, ['n', 'ngram', '1_doc_freq']].set_index('ngram')
    unique_pos_ngrams_cloud(only_pos_grams, 1234)


    # WORKER ANALYSIS #
    # Worker responses to test questions
    test_qs = pd.read_sql(test_qs_sql, sql_engine)
    worker_results = pd.read_sql(worker_scores_sql, sql_engine)

    test_qs = test_qs.merge(text_by_thread[['post_fullname', 'text']],
                            how='left',
                            left_on='post_fullname',
                            right_on='post_fullname',
                            )
    test_qs.to_pickle('worker_test_q_answers.pkl')

    # SUBREDDIT ANALYSIS #
    threads_per_sub = text_by_thread_no_test['subreddit'].value_counts()
    health_claims_per_sub = text_by_thread_no_test.groupby('health_claim')['subreddit']
    print(health_claims_per_sub.describe())
    neg_subs = health_claims_per_sub.value_counts().loc[0].value_counts()
    pos_subs = health_claims_per_sub.value_counts().loc[1].value_counts()

    # Close database connection
    db.db_close_conn(ssh, sql_engine)
    print('Complete.')
