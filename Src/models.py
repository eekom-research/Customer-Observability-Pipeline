from typing import Optional
from sqlalchemy import String, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db import Model


# tweet table
class Tweet(Model):
    __tablename__ = 'tweets'

    id: Mapped[str] = mapped_column(primary_key=True)
    text: Mapped[str]
    profile_id: Mapped[str]
    stats_likes: Mapped[Optional[int]]
    stats_retweets: Mapped[Optional[int]]
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.id'), index=True)
    date: Mapped[DateTime] = mapped_column(DateTime(timezone=True))

    # company: Mapped['Company'] = relationship(back_populates='tweets')

    def __repr__(self):
        return f'Tweet({self.id}, "{self.text}")'


# processed tweet table
class ProcessedTweet(Model):
    __tablename__ = 'processed_tweets'

    id: Mapped[str] = mapped_column(primary_key=True)
    text: Mapped[str]
    sk_full_topic: Mapped[Optional[str]]
    sk_topic: Mapped[Optional[str]]
    gensim_topic: Mapped[Optional[str]]
    company_id: Mapped[int] = mapped_column(ForeignKey('companies.id'), index=True)
    date: Mapped[DateTime] = mapped_column(DateTime(timezone=True))
    sentiment: Mapped[Optional[float]] = mapped_column(default=None)

    # company: Mapped['Company'] = relationship(back_populates='tweets')

    def __repr__(self):
        return f'ProcessedTweet({self.id}, {self.text}, {self.sk_topic}, {self.sk_topic})'


# company table
class Company(Model):
    __tablename__ = 'companies'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    nickname: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    country: Mapped[Optional[str]]
    industry: Mapped[Optional[str]]
    username: Mapped[Optional[str]]

    def __repr__(self):
        return f'Company({self.id}, {self.name})'


# company table
class TestCompany(Model):
    __tablename__ = 'test_companies'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    nickname: Mapped[str] = mapped_column(String(64), index=True, unique=True)
    country: Mapped[Optional[str]]
    industry: Mapped[Optional[str]]
    username: Mapped[Optional[str]]

    def __repr__(self):
        return f'TestCompany({self.id}, {self.name})'
