"""
    The purpose of this file is to provide the functions to do the following:
    - provide sentiment score for selected data
"""

# import necessary libraries
import pandas as pd
from flair.data import Sentence
from flair.models import TextClassifier
from textblob import TextBlob
from transformers import pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# sentiment interpretation: Textblob
def interpret_sentiment(sentiment, neg_val, neut_val, pos_val, neg_lim, pos_lim):
    if sentiment <=-neg_lim:
        return neg_val
    elif sentiment >= pos_lim:
        return pos_val
    else:
        return neut_val

# Apply Textblob sentiment to dataframe
def apply_textblob(df):

    # get columns for polarity and subjectivity scores
    df['textblob_polarity'] = df.iloc[:, -1].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_subjectivity'] = df.iloc[:, -2].apply(lambda x: TextBlob(x).sentiment.subjectivity)

    # implement sentiment categories for polarity and sentiment
    df['textblob_polarity_sentiment']= df['textblob_polarity'].apply(lambda x: interpret_sentiment(x,"Negative", "Neutral", "Positive", -0.25, 0.25))
    df['textblob_subjectivity_sentiment'] = df['textblob_subjectivity'].apply(lambda x: interpret_sentiment(x, "Subjective", "Neutral", "Objective", 0.3, 0.6))

    return df


# Calculate VADER score
def apply_vader(df):
    analyzer = SentimentIntensityAnalyzer()

    # calculate scores
    df[["neg", "neu", "pos", "vader_score"]] = df.iloc[:, -1].apply(lambda x: pd.Series(analyzer.polarity_scores(x)))

    # analyze sentiment using compound scores
    df['vader_sentiment'] =  df['vader_score'].apply(lambda x: interpret_sentiment(x,"Negative", "Neutral", "Positive", -0.25, 0.25))

    return df


# Calculate BERT Sentiment Score
def apply_roberta(df):
    sentiment_analysis = pipeline("sentiment-analysis", model = 'roberta-base')

    # calculate sentiment score and label
    df['roberta_sentiment'] = df.iloc[:, -1].apply(lambda x: sentiment_analysis(x)[0]['label'])
    df['roberta_score'] = df.iloc[:, -2].apply(lambda x: sentiment_analysis(x)[0]['score'])

    return df

# Calculate Flair Sentiment Score
def apply_flair(df):
    classifier = TextClassifier.load('sentiment')

    # calculate sentiment score and label
    df["flair_sentiment"] = df.iloc[:, -1].apply(lambda x: classifier.predict(Sentence(x))[0].value)
    df["flair_score"] = df.iloc[:, -2].apply(lambda x: classifier.predict(Sentence(x))[0].score)

    return df