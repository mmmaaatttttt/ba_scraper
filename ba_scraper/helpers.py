from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import FreqDist, ngrams
from string import punctuation

analyzer = SentimentIntensityAnalyzer()


def lower_and_remove_punc(sent):
    table = str.maketrans('', '', punctuation)
    return sent.lower().translate(table)


def speaker_features(line):
    words = line.words
    cleaned_words = lower_and_remove_punc(words).lower()
    cleaned_words_list = cleaned_words.split()
    features = {}
    word_freq = FreqDist(cleaned_words_list)
    most_freq_word = word_freq.most_common()[0][0]
    word_set = set(cleaned_words_list)
    bigram_set = set(
        [" ".join(ngram) for ngram in ngrams(cleaned_words_list, 2)])
    features[f"most_common_word={most_freq_word}"] = most_freq_word
    features["first_word"] = cleaned_words_list[0]
    for word in word_set:
        features[f"contains_word={word}"] = True
    for bigram in bigram_set:
        features[f"contains_bigram={bigram}"] = True
    return features


def truncate(s, num_chars=30):
    """Truncate a string to the specified number of characters.

    If truncation occurs, include an ellipsis at the end.

    >>> truncate("Hello world", 5)
    'He...'

    >>> truncate("Hello world")
    'Hello world'

    >>> truncate("Hello world", 11)
    'Hello world'

    >>> truncate("Hello world", 10)
    'Hello w...'

    """
    if len(s) <= num_chars:
        return s

    return s[:num_chars - 3] + "..."
