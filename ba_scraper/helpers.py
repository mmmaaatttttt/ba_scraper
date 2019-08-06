from nltk.sentiment.vader import SentimentIntensityAnalyzer
from string import punctuation

analyzer = SentimentIntensityAnalyzer()


def lower_and_remove_punc(sent):
    table = str.maketrans('', '', punctuation)
    return sent.lower().translate(table)
