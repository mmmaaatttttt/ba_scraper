from random import shuffle
from nltk import FreqDist, ngrams, Text, NaiveBayesClassifier
from nltk.classify import accuracy
from models import Conversation


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

    # write a method to get sentiment scores between a certain range

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
        def speaker_features(line):
            cleaned_words_list = [
                sent.lower_and_remove_punc().strip() for sent in line.sentences
            ]
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

        labeled_lines = []
        for speaker in speakers:
            labeled_lines.extend(self.all_lines(speaker))
        shuffle(labeled_lines)
        featuresets = [(speaker_features(line), line.speaker.lower())
                       for line in labeled_lines]
        train_set, test_set = featuresets[test_size:], featuresets[:test_size]
        classifier = NaiveBayesClassifier.train(train_set)
        classifier.show_most_informative_features(20)
        print(accuracy(classifier, test_set))

    def write_lines_to_file(self, speaker):
        """Write all words from speaker to a file. Useful for textgenrnn module."""
        all_words = [line.words for line in self.all_lines(speaker)]
        with open(f"{speaker}.txt", "w") as file:
            file.write("\n".join(all_words))
