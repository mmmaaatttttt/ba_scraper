# Beatiful Anonymous Scraper

Python scraper for analyzing episodes of [Beautiful / Anonymous](https://www.earwolf.com/show/beautiful-anonymous/).

Original transcripts can be found [here](https://drive.google.com/drive/folders/1Tygl0MsbI-b7dqTHEyXs0r-m9qR9wbwd?fbclid=IwAR06rY-TGgR-fDDVBcEGLE2e7KnXKfF5gzbnyEtvMruLA-2FFlz3UY7W3tg).

This tool will perform analysis on episodes of Beautiful Anonymous. (More info here later.)

### Setup

```sh
pip install -r requirements.txt

# to run the main file:
python3 -m ba_scraper
```

### Class setup

`Sentence` - this is the level at which all the sentiment analysis and profanity checking occurs.

`Line` - one for each line in the transcript. Assigned a speaker along with a collection of `Sentence` instances.

`Conversation` - basically a transcript transformed to be suitable for analysis. Contains a list of `Line` instances.

`ConversationList` - a collection of all available `Conversation` instances, one for each episode.

### Relevant dependencies

- Sentiment Analysis: [nltk](https://www.nltk.org/#), specifically VADER sentiment analysis. See [here](https://github.com/cjhutto/vaderSentiment) for details.
- Profanity checking: [profanity-check](https://github.com/vzhou842/profanity-check)
- Teaching a computer to talk like Chris Gethard: [textgenrnn](https://github.com/minimaxir/textgenrnn)