import os


class Line:
    def __init__(self, speaker, words):
        self.speaker = speaker
        self.words = words

    def __repr__(self):
        words_summary = self.words[:97] if len(
            self.words) <= 97 else f"{self.words[:97]}..."
        return f"<{self.speaker}: {words_summary}>"

    def word_count(self):
        """Count the words inside of a line (ignoring punctuation)."""
        return len(self.words.split())


class Conversation:
    def __init__(self, filepath):
        with open(os.path.join("src", "episodes", filepath)) as file:
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
