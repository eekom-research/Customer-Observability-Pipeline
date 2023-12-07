from db import Model, Session, engine
from models import Tweet, ProcessedTweet, Company
from sqlalchemy import select, or_, func


def get_training_raw_tweets(query_limit=10):
    companies_info_query = select(Company.username)
    with Session() as session:
        companies_username = session.scalars(companies_info_query).all()

    like_conditions = [func.lower(Tweet.text).like(f'%{username}%') for username in companies_username]
    #subquery = select(ProcessedTweet.id)
    #subquery_condition = Tweet.id.notin_(subquery)
    query = select(Tweet).where(or_(*like_conditions)).limit(query_limit)

    # Fetch raw tweets
    with Session() as session:
        raw_tweets = session.scalars(query).all()

    return raw_tweets


def get_raw_tweets(query_limit=10):
    companies_info_query = select(Company.username)
    with Session() as session:
        companies_username = session.scalars(companies_info_query).all()

    like_conditions = [func.lower(Tweet.text).like(f'%{username}%') for username in companies_username]
    subquery = select(ProcessedTweet.id)
    subquery_condition = Tweet.id.notin_(subquery)
    query = select(Tweet).where(or_(*like_conditions)).where(subquery_condition).limit(query_limit)

    # Fetch raw tweets
    with Session() as session:
        raw_tweets = session.scalars(query).all()

    return raw_tweets


def store_processed_tweets(filtered_df):
    data = filtered_df.to_dict(orient='records')
    with Session() as session:
        for item in data:
            processed_tweet = ProcessedTweet(**item)
            try:
                session.add(processed_tweet)
                session.commit()
            except Exception as e:
                session.rollback()
                print(e)
