from string import punctuation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from profanity_check import predict_prob, predict
from helpers import truncate

analyzer = SentimentIntensityAnalyzer()


class Sentence:
    """
    Class for a sentence in a podcast transcript.

    Attributes:
        words (str): The words in the sentence.
        sentiment (float): VADER compound sentiment score
            (see https://github.com/cjhutto/vaderSentiment#about-the-scoring)
        is_profane (bool): is the sentence profane?
        probability_profane (float): with what probability is the sentence profane?
            (see https://pypi.org/project/profanity-check/)
    """

    def __init__(self, words):
        self.words = words
        self.sentiment = analyzer.polarity_scores(self.words)["compound"]
        self.is_profane = bool(predict([self.words])[0])
        self.probability_profane = predict_prob([self.words])[0]

    def __repr__(self):
        return f"<Sentence words='{truncate(self.words)}' sentiment={self.sentiment}>"

    def lower_and_remove_punc(self):
        """lower case the words in the sentence and remove all punctuation."""
        table = str.maketrans('', '', punctuation)
        return self.words.lower().translate(table)

    def tokenize(self):
        """convert the words in the sentence to tokens for NLP."""
        return word_tokenize(self.lower_and_remove_punc())
