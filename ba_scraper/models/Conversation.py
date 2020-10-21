import os
import json
from numpy import mean, var
from ba_scraper.models.Line import Line


class Conversation:
    """
    Class for a conversation of a podcast transcript.

    Attributes:
        id (int): Episode number
        title (str): Episode title
        date (str): Episode release date
        lines (list): A list of line instances derived from the transcript.
    """

    def __init__(self, filepath):
        # load data from transcripts
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
        return [line for line in self.lines if line.speaker == speaker]

    def profanity_prob_by_line(self, speaker=None):
        """Return a list of profanity probabilities, one for each sentence in the conversation.
        Optionally filter the statements by speaker."""
        if speaker:
            return [
                line.profanity_probs() for line in self.lines
                if line.speaker == speaker
            ]
        return [line.profanity_probs() for line in self.lines]

    def profanity_stats(self, speaker=None):
        """Return the mean and variance for profanity probabilities in the conversation,
        optionally filtered by speaker."""
        all_probs = [
            prob for prob_list in self.profanity_prob_by_line(speaker)
            for prob in prob_list
        ]

        if speaker:
            profane_sentences = sum(
                line.profanity_count() for line in self.lines_by(speaker))
            all_sentences = sum(
                line.sentence_count() for line in self.lines_by(speaker))
        else:
            profane_sentences = sum(
                line.profanity_count() for line in self.lines)
            all_sentences = sum(line.sentence_count() for line in self.lines)

        return {
            "profanity_prob_average": mean(all_probs),
            "profanity_prob_variance": var(all_probs),
            "profane_sentence_count": profane_sentences,
            "all_sentence_count": all_sentences
        }

    def sentiment_by_line(self, speaker=None):
        """Return a list of sentiment analyses, one for each line in the conversation.
        Optionally filter the sentiments by speaker."""
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
            "compound_average": mean([sent["compound"] for sent in all_sentiments]),
            "compound_variance": var([sent["compound"] for sent in all_sentiments])
        }

    def speakers(self):
        """Return a set of the names of the speakers in the conversation."""
        return set(line.speaker for line in self.lines)

    def profanity_count(self, speaker=None):
        """Return a count of the number of sentences with profanity in the conversation.
        Can optionally pass a speaker to filter the count by."""
        if speaker:
            return sum(line.profanity_count() for line in self.lines
                       if line.speaker == speaker)
        return sum(line.profanity_count() for line in self.lines)

    def word_count(self, speaker=None):
        """Return a count of the number of words in the conversation.
        Can optionally pass a speaker to filter the count by."""
        if speaker:
            return sum(line.word_count() for line in self.lines
                       if line.speaker == speaker)
        return sum(line.word_count() for line in self.lines)

    def all_sentiment_json(self):
        return json.dumps({
            "id": self.id,
            "title": self.title,
            "date": self.date,
            "sentiment_counts":
            [[line.speaker, mean(line.sentiments())] for line in self.lines]
        })

    def sentiment_count_json(self, min_sentiment=-1, max_sentiment=1):
        return json.dumps({
            "id": self.id,
            "title": self.title,
            "date": self.date,
            "sentiment_counts": {
                "Chris":
                len([
                    sentence for line in self.lines
                    for sentence in line.sentences if line.speaker == "Chris"
                    and min_sentiment < sentence.sentiment < max_sentiment
                ]),
                "Caller":
                len([
                    sentence for line in self.lines
                    for sentence in line.sentences if line.speaker == "Caller"
                    and min_sentiment < sentence.sentiment < max_sentiment
                ]),
                "min_sentiment":
                min_sentiment,
                "max_sentiment":
                max_sentiment
            }
        })

    def word_count_summary_json(self):
        return json.dumps({
            "id": self.id,
            "title": self.title,
            "date": self.date,
            "word_counts": {
                "Chris": self.word_count("Chris"),
                "Caller": self.word_count("Caller")
            }
        })
    
    def profanity_count_summary_json(self):
        return json.dumps({
            "id": self.id,
            "title": self.title,
            "date": self.date,
            "profanity_counts": {
                "Chris": self.profanity_count("Chris"),
                "Caller": self.profanity_count("Caller")
            }
        })
