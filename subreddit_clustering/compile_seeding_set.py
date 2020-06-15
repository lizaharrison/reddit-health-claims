#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : compile_seeding_set.py
AUTHOR : Eliza Harrison

Retrieves and compiles a seeding set of subreddits known to relate to the
discussion of health topics and which can be used to identify clusters of similarly
health-related subreddits.

"""

import re

import pandas as pd

import config
import database_functions as db


def load_health_subs():
    """
    Loads health-related subreddits (seeding set) from file.

    Returns
    -------
    health_subs : dataframe
        Contains all the health-related subreddits that make up the seeding set for
        subreddit clustering.

    """
    print('\nLoading known health-related subreddits from pickle file...')
    filepath = '{}/r_Health_related_subs.pkl'.format(config.project_dir)
    health_subs = pd.read_pickle(filepath)

    return health_subs


if __name__ == '__main__':

    # URL for r/Health information = 'https://www.reddit.com/r/health/about.json?'
    r_health = pd.read_json('{}/r_Health_about.json'.format(config.project_dir))
    r_health_desc = r_health.loc['description', 'data']
    r_health_desc = r_health_desc[r_health_desc.find('**Other health related subreddits**'):]
    print(r_health_desc)

    sub_name_regex = re.compile(r'\[[a-zA-Z0-9\'_ ]*\]')
    sub_url_regex = re.compile(r'/[r]/[a-zA-Z0-9_]*')
    sub_names = pd.Series(re.findall(sub_name_regex, r_health_desc)).str.replace('[', '').str.replace(']', '')
    sub_urls = pd.Series(re.findall(sub_url_regex, r_health_desc))
    sub_urls.drop_duplicates(keep='first', inplace=True)
    sub_urls = sub_urls.apply(lambda x: x + '/')
    print(sub_names)
    print(sub_urls)

    ssh, sql_engine = db.db_connect()
    sql = ("SELECT "
           "display_name, "
           "url "
           "FROM all_subreddits"
           )
    all_subs = pd.read_sql(sql, sql_engine)
    db.db_close_conn(ssh, sql_engine)

    exclude = ['Green',
               'VentureBiotech', ]

    health_related_subs = all_subs.loc[(all_subs['url'].str.lower().isin(sub_urls.str.lower())) &
                                       (~all_subs['display_name'].isin(exclude))]
    r_health = all_subs.loc[all_subs['display_name'] == 'Health']
    health_related_subs = pd.concat([health_related_subs, r_health])

    health_related_subs.sort_values('display_name',
                                    inplace=True, )
    health_related_subs.reset_index(inplace=True,
                                    drop=True, )
    print(health_related_subs)

    health_related_subs.to_pickle('r_Health_related_subs.pkl')

    print('\nGeneration of seeding set of known-health subreddits complete.')
