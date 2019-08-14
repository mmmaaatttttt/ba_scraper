from nltk.tokenize import sent_tokenize
from models import Sentence
from helpers import truncate


class Line:
    """
    Class for a sentence in a podcast transcript.

    Attributes:
        speaker (str): The person who says the line.
        words (str): The words in the line.
        sentences (list): A list of sentence instances derived from the words in the line.
    """

    def __init__(self, speaker, words):
        self.speaker = speaker
        self.words = words
        self.sentences = [Sentence(sent) for sent in sent_tokenize(self.words)]

    def __repr__(self):
        """repr for an instance of line.

        >>> sent = Line("Chris", "Hello caller.")
        >>> sent
        <Chris: Hello caller.>

        """
        return f"<{self.speaker}: {truncate(self.words, 100)}>"

    def sentiments(self):
        """Extract the sentiment scores for each sentence in the line."""
        return [sent.sentiment for sent in self.sentences]

    def profanity_probs(self):
        """Return probabilities that each sentence within a line is profane."""
        return [sent.probability_profane for sent in self.sentences]

    def profanity_count(self):
        """Return a count of the number of offensive sentences in a line."""
        return sum(sent.is_profane for sent in self.sentences)

    def word_count(self):
        """Count the words inside of a line."""
        return len(self.words.replace("...", "").split())

    def sentence_count(self):
        """Count the sentences inside of a line."""
        return len(self.sentences)
