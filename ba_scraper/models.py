import os
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import FreqDist, ngrams, Text
from numpy import mean, var
from ba_scraper.helpers import analyzer, lower_and_remove_punc


class Line:
    def __init__(self, speaker, words):
        self.speaker = speaker
        self.words = words

    def __repr__(self):
        words_summary = self.words[:97] if len(
            self.words) <= 97 else f"{self.words[:97]}..."
        return f"<{self.speaker}: {words_summary}>"

    def avg_sentiment(self):
        """Return the average compound sentiment for all sentences in the line."""
        sentiments = self.sentiments()
        return mean([sent['compound'] for sent in sentiments])

    def sentiments(self):
        """Perform sentiment analysis on all sentences in the words.
        Returns a list of analyses, one for each sentence.
        """
        sentences = sent_tokenize(self.words)
        return [analyzer.polarity_scores(sent) for sent in sentences]

    def word_count(self):
        """Count the words inside of a line."""
        return len(self.words.replace("...", "").split())


class Conversation:
    def __init__(self, filepath):
        with open(os.path.join("episodes", filepath)) as file:
            text_lines = file.readlines()
            self.id = int(text_lines[0].split(" ")[1])
            self.title = text_lines[1].strip()
            self.date = text_lines[2].strip()
            self.lines = []
            curr_speaker = text_lines[3].split(":")[0]
            curr_words = ''
            for line_idx, line in enumerate(text_lines[3:]):
                colon_idx = line.find(":")
                words = line[colon_idx + 1:].strip()
                speaker = line[:colon_idx]
                # if the same speaker has multiple consecutive lines,
                # just consolidate them
                if speaker == curr_speaker:
                    curr_words += '\n' + words
                # Otherwise complete the line and swap to the other speaker
                else:
                    self.lines.append(Line(curr_speaker, curr_words))
                    curr_speaker = speaker
                    curr_words = words
                # if it's the last line, always append
                if line_idx == len(text_lines) - 4:
                    self.lines.append(Line(curr_speaker, curr_words))

    def __repr__(self):
        return f"<Episode {self.id}: {self.title} ({self.date})>"

    def lines_by(self, speaker):
        """Return a list of all lines spoken by the given speaker."""

        return [line.words for line in self.lines if line.speaker == speaker]

    def line_count(self, speaker=None):
        """Return a count of the number of lines in the conversation.
        Can optionally pass a speaker to filter the count by.
        """
        if speaker:
            return len(self.lines_by(speaker))
        return len(self.lines)

    def sentiment_by_line(self, speaker=None):
        """Return a list of sentiment analyses, one for each line in the conversation.
        Optionally filter the sentiments by speaker"""
        if speaker:
            return [
                line.sentiments() for line in self.lines
                if line.speaker == speaker
            ]
        return [line.sentiments() for line in self.lines]

    def sentiment_stats(self, speaker=None):
        """Return the mean and variance for compound sentiment in the conversation,
        optionally filtered by speaker."""
        all_sentiments = [
            sent_dict for sent in self.sentiment_by_line(speaker)
            for sent_dict in sent
        ]
        return {
            "compound_average":
            mean([sent["compound"] for sent in all_sentiments]),
            "compound_variance":
            var([sent["compound"] for sent in all_sentiments])
        }

    def speakers(self):
        """Return a set of the names of the speakers in the conversation."""
        return set(line.speaker for line in self.lines)

    def word_count(self, speaker=None):
        """Return a count of the number of words in the conversation.
        Can optionally pass a speaker to filter the count by.
        """
        if speaker:
            return sum(line.word_count() for line in self.lines
                       if line.speaker == speaker)
        return sum(line.word_count() for line in self.lines)

    def summarize(self):
        total_wc = self.word_count()
        speaker_info = [{
            "name": speaker,
            "word_count": self.word_count(speaker),
            "sentiment": self.sentiment_stats(speaker),
        } for speaker in self.speakers()]
        print(
            self,
            f"Line count: {self.line_count()}",
            *(f"{sp['name']} word count: {sp['word_count']} ({sp['word_count'] / total_wc:.2%})"
              for sp in speaker_info),
            f"Total word count: {total_wc}",
            *(f"{sp['name']} compound average: {sp['sentiment']['compound_average']}"
              for sp in speaker_info),
            *(f"{sp['name']} compound variance: {sp['sentiment']['compound_variance']}"
              for sp in speaker_info),
            sep='\n',
            end='\n\n-----------\n\n')


class ConversationList:
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
            for sent in sent_tokenize(line.words):
                cleaned_tokens = word_tokenize(lower_and_remove_punc(sent))
                freq.update(" ".join(ngram)
                            for ngram in ngrams(cleaned_tokens, token_count))
        return freq

    def most_common_phrases(self, speaker, phrase_length, min_count_per_convo=1):
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
        cleaned_string = lower_and_remove_punc(" ".join([line.words for line in all_lines_by_speaker]))
        return Text(cleaned_string.split(" ")).collocation_list()
