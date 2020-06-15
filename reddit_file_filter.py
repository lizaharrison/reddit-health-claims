#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : reddit_file_filter.py
AUTHOR : Eliza Harrison

Filters the subreddit, submission and comment files to extract records
eligible for subreddit clustering and the classification of threads containing
health claims.

"""


import argparse
import os
import subprocess
import time

import pandas as pd

import config
import database_functions as db


def get_files(filetype='zst'):
    """
    Gets list of snapshot filenames.

    Parameters
    ----------
    filetype : str
        Whether to load files of type .zst ('zst') or .xz ('xz')

    Returns
    -------
    text_files : list
        List of .zst or .xz available in specified directory.

    """
    os.chdir('./{}'.format(filetype))
    text_files = [file for file in os.listdir() if (file not in ['.DS_Store',
                                                                 'Completed',
                                                                 ]) and (not file.endswith('.{}'.format(filetype)))]
    print('Existing uncompressed files: {}'.format(text_files))

    if filetype == 'zst':
        files = [file for file in os.listdir() if file.endswith('zst') and file[:-4] not in text_files]
    elif filetype == 'xz':
        files = [file for file in os.listdir() if file.endswith('xz') and file[:-3] not in text_files]

    files.sort(reverse=True)
    text_files.extend(files)
    print('All files: {}'.format(files))

    return text_files


def uncompress_file(file, filetype='zst'):
    """
    Uncompresses snapshot files

    Parameters
    ----------
    file : str
        Name of file to uncompress
    filetype : str
        Specifies compression type ('zst', 'xz')

    Returns
    -------
    uncomp_filename : str
        Name of uncompressed file

    """

    print('Current working directory: {}'.format(os.getcwd()))
    '''
    filepath = '{}/Comments/{}/{}'.format(config.raw_data_files,
                                          filetype,
                                          file,
                                          )
    print('Path to current file: {}'.format(filepath))
    '''
    print('Uncompressing file...')

    if filetype == 'zst':
        # Uncompress .zst file before filtering
        command = ['unzstd']

    elif filetype == 'xz':
        # Uncompress .xz file before filtering
        command = ['unxz', '-k']

    command.append(file)
    subprocess.call(command)
    uncomp_filename = file.replace('.{}'.format(filetype), '')
    print('Filename for uncompressed file: {}'.format(uncomp_filename))
    print('Complete')

    return uncomp_filename


def compress_file(uncomp_filename, filetype='zst'):
    """
    Compress file

    Parameters
    ----------
    uncomp_filename : str
        Name of uncompressed file
    filetype : str
        Specifies compression type ('zst', 'xz')

    Returns
    -------

    """
    filepath = './{}/{}/{}'.format(config.raw_data_files,
                                   filetype,
                                   uncomp_filename,
                                   )
    print('Compressing file to .{}...'.format(filetype))

    if filetype == 'zst':
        # Uncompress .zst file before filtering
        command = ['zstd']

    elif filetype == 'xz':
        # Uncompress .xz file before filtering
        command = ['xz', '-z']

    command.append(uncomp_filename)
    subprocess.call(command)
    print('Complete')


def json_to_df(infile, chunksize):
    """

    Parameters
    ----------
    infile : str
        Name of JSON file
    chunksize : int
        Size of chunks read into memory

    Returns
    -------
    chunked_df
        Iterator object containing data in chunks of size specified by chunksize
        parameter
    """

    chunked_df = pd.read_json(infile,
                              lines=True,
                              chunksize=chunksize,
                              )
    return chunked_df


def subreddit_filter(engine):
    """
    Reads Reddit subreddit JSON files, filters records according to pre-defined inclusion criteria
    and uploads required fields to existing database table.

    Parameters
    ----------
    engine : SQLAlchemy engine
        Connection to database to which subreddit data will be written

    Returns
    -------

    """

    fields = ['id',
              'name',
              'display_name',
              'title',
              'url',
              'subscribers',
              'subreddit_type',
              'description',
              'lang',
              ]

    # Filepath for subreddit data file
    filepath = config.raw_data_files + '/Subreddits'
    os.chdir(filepath)
    file = 'subreddits.ndjson'

    print('File: {}'.format(file))
    start = time.time()
    chunks = json_to_df(file,
                        chunksize=100000,
                        )
    # print('JSON file loaded')

    for i, chunk in enumerate(chunks):
        # Selects subreddits meeting filter criteria
        # print('Filtering chunk ' + str(i) + '...')
        filtered_records = chunk.loc[(chunk['over18'] == 0) &
                                     (chunk['lang'].isin(['en', None])) &
                                     (chunk['subreddit_type'].isin(['public', 'restricted'])) &
                                     (chunk['subscribers'] > 9),
                                     ]

        if len(filtered_records) > 0:
            print(str(len(filtered_records))
                  + ' subreddits meeting inclusion criteria')
            final_data = filtered_records[fields]
            # Writes data to existing database table
            db.df_to_db(final_data, 'all_subreddits_w_title', engine, if_exists='append')
            print('Data written to database table')

    end = time.time()
    print('File %s processing time: %s' % (file,
                                           str(time.strftime('%H:%M:%S',
                                                             time.gmtime(end - start)))))


def submission_filter(engine, subreddits, submission_tablename, filetype='zst'):
    """
    Reads Reddit submission JSON files, filters records according to pre-defined inclusion criteria
    and uploads required fields to existing database table.

    Parameters
    ----------
    engine : SQLAlchemy engine
        Connection to database to which submission data will be written
    subreddits : series
        Names of all subreddits meeting initial inclusion criteria
    submission_tablename : str
        Name of table to upload submission data to
    filetype : str
        Whether to load data from .zst or .xz files

    Returns
    -------


    """

    fields = ['subreddit_id',
              'subreddit',
              'id',
              'fullname',
              'permalink',
              'score',
              'title',
              'selftext',
              'url',
              'author',
              'author_fullname',
              'created_utc',
              'source_file',
              ]

    files = get_files(filetype)

    # Iteratively read post files
    # Filter subreddits according to inclusion criteria
    # Save subreddits meeting inclusion criteria to existing database table
    for file_before in files:
        start = time.time()
        print('File: {}'.format(file_before))

        # Uncompresses file ready for filtering
        if file_before.endswith(filetype):
            file_after = uncompress_file(file_before, filetype)
        else:
            file_after = file_before

        # Loads text file into iterator object in chunks of specified length
        # print('Loading file...')
        chunks = json_to_df(file_after,
                            chunksize=100000,
                            )
        # print('JSON file loaded')

        for i, chunk in enumerate(chunks):
            # Selects posts meeting filter criteria
            # print('Filtering chunk ' + str(i) + '...')
            filtered_records = chunk.loc[(chunk['author'] != '[deleted]') &
                                         (chunk['over_18'] == 0) &
                                         (chunk['subreddit'].isin(subreddits['display_name'])),
                                         (chunk['title'].str.lower != '[deleted by user]') &
                                         (chunk['title'].str.lower != '[removed by user]') &
                                         (chunk['title'].str.lower != '[deleted]') &
                                         (chunk['title'].str.lower != '[removed]') &
                                         (chunk['title]'].str.lower != '(removed)') &
                                         (chunk['title]'].str.lower != 'removed') &
                                         (chunk['title]'].str.lower != 'deleted') &
                                         (chunk['selftext'].str.lower != '[deleted]') &
                                         (chunk['selftext'].str.lower != '[removed]') &
                                         (chunk['selftext'].str.lower.str.lower() != 'deleted') &
                                         (chunk['selftext'].str.lower.str.lower() != 'removed')
                                         ]

            if len(filtered_records) > 0:
                print(str(len(filtered_records))
                      + ' posts meeting inclusion criteria')
                # Adds new fields for posts
                filtered_records['fullname'] = filtered_records.apply(lambda row: 't3_' + row.id,
                                                                      axis=1,
                                                                      )
                filtered_records['source_file'] = file_after
                # Removes unrequired fields
                final_data = filtered_records[fields]

                # Writes data to existing database table
                db.df_to_db(final_data, submission_tablename, engine)
                # print('Data written to database table')

        # Re-compress file
        # compress_file(uncomp_filename, filetype)

        end = time.time()
        print('File %s processing time: %s' % (file_after,
                                               str(time.strftime('%H:%M:%S',
                                                                 time.gmtime(end - start)))))

        # Delete uncompressed file after filtering complete
        subprocess.call(['rm', file_after])
        # Moves completed file to 'Completed' folder
        subprocess.call(['mv', file_before, './Completed/'])


def comment_filter(engine, submissions, comments_tablename, filetype='zst'):
    """
    Reads Reddit comment JSON files, filters records according to pre-defined inclusion critera
    and uploads required fields to existing database table.
    
    Parameters
    ----------
    engine : SQLAlchemy engine
        Connection to database to which submission data will be written
    submissions : series
        All submissions (identified using fullnames e.g. t3_xxxxx) for which associated comments
        should be extracted from JSON files
    comments_tablename : str
        Name of table to upload comment data to
    filetype : str
        Whether to load data from .zst or .xz files

    Returns
    -------

    """

    fields = ['subreddit_id',
              'subreddit',
              'link_id',
              'id',
              'parent_id',
              'permalink',
              'score',
              'body',
              'source_file',
              ]

    files = get_files(filetype)

    # Iteratively read post files
    # Filter subreddits according to inclusion criteria
    # Save subreddits meeting inclusion criteria to existing database table
    for file_before in files:
        start = time.time()
        print('File: {}'.format(file_before))

        # Uncompresses file ready for filtering
        if file_before.endswith(filetype):
            file_after = uncompress_file(file_before, filetype)
        else:
            file_after = file_before

        # Loads text file into iterator object in chunks of specified length
        # print('Loading file...')
        chunks = json_to_df(file_after,
                            chunksize=500000,
                            )

        for i, chunk in enumerate(chunks):
            # Selects posts meeting filter criteria
            # print('Filtering chunk ' + str(i) + '...')
            filtered_records = chunk.loc[(chunk['link_id'].isin(submissions['fullname'])) &
                                         (chunk['parent_id'] == chunk['link_id']) &
                                         (chunk['body'].str.lower != '[deleted by user]') &
                                         (chunk['body'].str.lower != '[removed]') &
                                         (chunk['body'].str.lower != '(removed)') &
                                         (chunk['body'].str.lower != '[deleted]') &
                                         (chunk['body'].str.lower != '[removed]') &
                                         (chunk['body'].str.lower() != 'deleted') &
                                         (chunk['body'].str.lower() != 'removed')
                                         ]

            if len(filtered_records) > 0:
                print(str(len(filtered_records))
                      + ' comments meeting inclusion criteria')
                # Adds new fields for comments
                filtered_records['source_file'] = file_after

                # Removes unrequired fields
                final_data = filtered_records[fields]

                # Writes data to existing database table
                try:
                    db.df_to_db(final_data, comments_tablename, engine)
                    # print('Data written to database table')
                except ValueError as e:
                    print(e)
                    print(final_data)
                    final_data.to_pickle('ValueError.pkl')
                    final_data = final_data.apply(lambda col: col.str.replace(
                        '\x00', '') if col.dtype == object else col)
                    db.df_to_db(final_data, comments_tablename, engine)

        # Re-compress file
        compress_file(file_after, filetype)

        end = time.time()
        print('File %s processing time: %s' % (file_after,
                                               str(time.strftime('%H:%M:%S',
                                                                 time.gmtime(end - start)))))

        # Delete uncompressed file after filtering complete
        subprocess.call(['rm', file_after])
        # Moves completed file to 'Completed' folder
        subprocess.call(['mv', file_before, './Completed/'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--sub',
                        help='filter subreddits from files',
                        action='store_true',
                        )
    parser.add_argument('-p', '--post',
                        help='filter posts from files',
                        action='store_true',
                        )
    parser.add_argument('-c', '--comm',
                        help='filter comments from files',
                        action='store_true',
                        )
    parser.add_argument('-l', '--health',
                        help='filter only posts and comments from health subreddits',
                        action='store_true',
                        )
    parser.add_argument('-f', '--file',
                        help='nominate filetype for filtering (zst/xz)',
                        choices=['zst', 'xz'],
                        default='zst',
                        )
    parser.add_argument('-m', '--method',
                        help='',
                        type=str,
                        default='k',
                        choices=('k', 'l'),
                        )
    args = parser.parse_args()

    # Connects to dbreddit database using SSH tunnel
    ssh, sql_engine = db.db_connect()

    file_type = args.file
    print('File type: {}'.format(file_type))

    if args.method == 'k':
        method = 'kmeans'
    elif args.method == 'l':
        method = 'lda'

    if args.sub:
        # SUBREDDIT FILTER
        # Changes working directory to folder containing subreddit file
        path = config.raw_data_files + '/Subreddits'
        os.chdir(path)

        # Executes subreddit filter
        subreddit_filter(sql_engine)

    if args.post:
        # POST FILTER
        if args.health:
            sql = 'SELECT display_name FROM dbreddit.public.health_subreddits_100_1_{}'.format(method)
            post_table = 'health_posts'
            print('Downloading health subreddits from database...')
        else:
            sql = 'SELECT display_name FROM dbreddit.public.all_subreddits'
            post_table = 'all_posts'
            print('Downloading all subreddits from database...')

        # Downloads list of subreddits meeting inclusion criteria
        subreddits = pd.read_sql(sql,
                                 sql_engine,
                                 )
        print('Download complete')

        # Changes working directory to folder containing post files
        path = config.raw_data_files + '/Posts'
        os.chdir(path)

        # Executes post filter
        submission_filter(sql_engine, subreddits, post_table)

    if args.comm:
        # COMMENT FILTER
        if args.health:
            sql = "SELECT fullname FROM dbreddit.public.health_posts_100_1"
            comments_table = 'health_comments_100_1_{}'.format(method)

        else:
            sql = 'SELECT fullname FROM dbreddit.public.all_posts'
            comments_table = 'all_comments'

        # Loads list of posts for which to filter comments
        posts = pd.read_sql(sql,
                            sql_engine,
                            )
        print('Posts: \n{}'.format(posts))

        # Changes file path to folder containing comment files
        path = config.raw_data_files + '/Comments'
        os.chdir(path)
        print('Folder containing comments data files: {}'.format(os.getcwd()))
        # file_type = input('Please enter the file type ("zst" or "txt"): ')
        # Executes comment filter
        comment_filter(sql_engine,
                       posts,
                       comments_table,
                       filetype=file_type,
                       )

    # Closes connection to dbreddit database
    db.db_close_conn(ssh, sql_engine)

    print('Complete.')
