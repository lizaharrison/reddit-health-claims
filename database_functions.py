#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
PROGRAM : database_functions.py
AUTHOR : Eliza Harrison

Functions used to connect, read and write to secure database in which all
Reddit data used for this project is stored.

"""

import psycopg2
from sqlalchemy import create_engine
from sshtunnel import SSHTunnelForwarder

import config


def ssh_tunnel():
    """
    Creates ssh tunnel to database server using credentials stored in config.py file
    :return:
    """
    tunnel = SSHTunnelForwarder(
        config.dbreddit['ssh_host_ip'],
        remote_bind_address=('localhost', 5432),
        # local_bind_address=('localhost', 5432),
        ssh_username=config.dbreddit['user'],
        ssh_password=config.dbreddit['password'],
    )
    # Start the SSH tunnel
    print(tunnel)
    tunnel.start()
    return tunnel


def sql_alch_engine(tunnel):
    """
    Connects to dbreddit database using SQLAlchemy engine
    Parameters
    ----------
    tunnel
        SSH tunnel to database server

    Returns
    -------
    engine
        SQLAlchemy engine

    """

    port = str(tunnel.local_bind_port)

    # Create a database connection using sqlalchemy
    connection_addr = ('postgresql://'
                       + config.dbreddit['user']
                       + ':'
                       + config.dbreddit['password']
                       + '@localhost:'
                       + port
                       + '/'
                       + config.dbreddit['dbname'])
    try:
        engine = create_engine(connection_addr)
        return engine
    except Exception as e:
        print(e)


def db_connect():
    """
    Connects to database via SSH tunnel and SQLAlchemy engine

    Returns
    -------
    tunnel, engine : tuple
        SSH tunnel and SQLAlchemy engine
    """

    # Connects to dbreddit database using SSH tunnel adn SQLAlchemy engine
    tunnel = ssh_tunnel()
    engine = sql_alch_engine(tunnel)
    print('\n')

    return tunnel, engine


def psycopg2_connect(tunnel):
    """
    Connects to database via SSH tunnel and psycopg2 connection

    Parameters
    ----------
    tunnel
        SSH tunnel to database server

    Returns
    -------
    connection
        psycopg2 connection to database

    """

    # Create a database connection
    connection = psycopg2.connect(
        database=config.dbreddit['dbname'],
        user=config.dbreddit['user'],
        password=config.dbreddit['password'],
        host=tunnel.local_bind_host,
        port=tunnel.local_bind_port,
    )
    connection.set_session(autocommit=True)
    print('Database connection successful')
    return connection


def df_to_db(dataframe, tablename, engine,
             index=False, index_label=None, if_exists='append',
             chunksize=100000):
    """
    Uploads dataframe to database table.

    Parameters
    ----------
    dataframe : dataframe
        Dataframe to write to database table
    tablename : str
        Name of table to write to
    engine : SQLAlchemy engine
        Connection to database
    index : bool
        Whether to include index as column in database table
    index_label : str
        Name of index column (if index=True)
    if_exists : str
        Whether to append dataframe to existing table ('append') or replace
        contents of existing table ('replace')
    chunksize : int
        Number of rows to upload to database in a single chunk

    Returns
    -------

    """
    dataframe.to_sql(tablename,
                     con=engine,
                     index=index,
                     index_label=index_label,
                     if_exists=if_exists,
                     chunksize=chunksize
                     )


def db_close_conn(tunnel, engine):
    """
    Closes connection to database

    Parameters
    ----------
    tunnel : SSHTunnel
        SSH tunnel
    engine : SQLAlchemy engine
        Connection to database

    Returns
    -------

    """
    engine.dispose()
    tunnel.close()


def check_table(table_name, engine):
    """
    Check whether database table exists.

    Parameters
    ----------
    table_name : str
        Name of table
    engine
        SQLAlchemy connection to database

    Returns
    -------
    bool

    """
    sql = ("SELECT "
           "* "
           "FROM information_schema.tables "
           "WHERE table_name = '{}'".format(table_name)
           )
    result = engine.execute(sql)

    if len(result.fetchall()) > 0:
        return True
    else:
        return False
