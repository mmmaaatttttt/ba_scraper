from random import shuffle
from nltk import FreqDist, ngrams, Text, NaiveBayesClassifier
from nltk.classify import accuracy
from ba_scraper.models.Conversation import Conversation


class ConversationList:
    """
    Class for a collection of podcast transcripts.

    Attributes:
        conversations (list): A list of conversation instances.
    """

    def __init__(self, filepaths):
        self.conversations = [Conversation(fpath) for fpath in filepaths]

    def all_lines(self, speaker):
        """Get all lines from speaker across all conversations."""
        return [
            line for convo in self.conversations for line in convo.lines
            if line.speaker == speaker
        ]

    def ngram_freq(self, speaker, token_count=1):
        """Return a FreqDist of ngrams of length token_count for speaker."""
        freq = FreqDist()
        for line in self.all_lines(speaker):
            for sent in line.sentences:
                freq.update(" ".join(ngram)
                            for ngram in ngrams(sent.tokenize(), token_count))
        return freq

    def most_common_phrases(self,
                            speaker,
                            phrase_length,
                            min_count_per_convo=1):
        """Return a list of common phrases said by speaker of token length equal to phrase_length,
        provided the phrase appears at least an average of min_count_per_convo number of times.
        Defaults to an average of at least once per conversation."""
        all_phrases = self.ngram_freq(speaker, phrase_length)
        common_phrases = [
            item for item in all_phrases.items()
            if item[1] / len(self.conversations) >= min_count_per_convo
        ]
        common_phrases.sort(key=lambda t: t[1], reverse=True)
        return common_phrases

    def collocation_list(self, speaker):
        all_lines_by_speaker = self.all_lines(speaker)
        cleaned_string = " ".join(sent.lower_and_remove_punc()
                                  for line in all_lines_by_speaker
                                  for sent in line.sentences)
        return Text(cleaned_string.split(" ")).collocation_list()

    def classifier_summary(self, speakers, test_size=500):
        labeled_lines = []
        for speaker in speakers:
            labeled_lines.extend(self.all_lines(speaker))

        all_words = [
            word for line in labeled_lines for sent in line.sentences
            for word in sent.lower_and_remove_punc().strip().split()
        ]

        all_bigrams = [
            " ".join(ngram) for line in labeled_lines
            for sent in line.sentences
            for ngram in ngrams(sent.lower_and_remove_punc().strip(), 2)
        ]

        word_freq = FreqDist(all_words)
        most_common_words = list(word_freq)[:2000]

        bigram_freq = FreqDist(all_bigrams)
        most_common_bigrams = list(bigram_freq)[:2000]

        def speaker_features(line):
            cleaned_words_list = [
                word for sent in line.sentences
                for word in sent.lower_and_remove_punc().strip().split()
            ]
            features = {}
            word_freq = FreqDist(cleaned_words_list)
            most_freq_word = word_freq.most_common()[0][0]
            word_set = set(cleaned_words_list)
            bigram_set = set(
                [" ".join(ngram) for ngram in ngrams(cleaned_words_list, 2)])
            avg_sentiment = sum(sent for sent in line.sentiments()) / len(line.sentences)
            features[f"most_common_word={most_freq_word}"] = most_freq_word
            features["first_word"] = cleaned_words_list[0]
            features["has_profanity"] = line.profanity_count() > 0
            features["sentiment_very_negative"] = avg_sentiment < -0.5
            features["sentiment_negative"] = -0.5 < avg_sentiment < 0.05
            features["sentiment_neutral"] = -0.05 < avg_sentiment < 0.05
            features["sentiment_positive"] = 0.05 < avg_sentiment < 0.5
            features["sentiment_very_positive"] = 0.5 < avg_sentiment
            features["long_line"] = line.word_count() > 50
            features["num_repeated_words"] = len(
                [val for val in word_freq.values() if val > 1])
            for word in most_common_words:
                features[f"contains({word})"] = (word in word_set)
            for bigram in most_common_bigrams:
                features[f"contains_bigram({bigram})"] = (bigram in bigram_set)
            features["asks_question"] = ("?" in line.words)
            features["contains_nyc"] = ("new york city" in word_set)
            return features

        shuffle(labeled_lines)
        featuresets = [(speaker_features(line), line.speaker.lower())
                       for line in labeled_lines]
        train_set, test_set = featuresets[test_size:], featuresets[:test_size]
        classifier = NaiveBayesClassifier.train(train_set)
        classifier.show_most_informative_features(50)
        print(accuracy(classifier, test_set))

    def write_lines_to_file(self, speaker):
        """Write all words from speaker to a file. Useful for textgenrnn module."""
        all_words = [line.words for line in self.all_lines(speaker)]
        with open(f"{speaker}.txt", "w") as file:
            file.write("\n".join(all_words))
