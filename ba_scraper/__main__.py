import os
from ba_scraper.models import ConversationList

if __name__ == '__main__':
    transcript_names = os.listdir(os.path.join('episodes'))
    convo_list = ConversationList(transcript_names)
    for convo in convo_list.conversations:
        print(convo.summary_json())

# predictors for Chris vs guest (ML, all ep)
# most common words / bigrams for speakers (esp chris) (all ep)
# 1 - 5 word phrases
# ignore stop words for 1
# sorry sally - 4 times for each
# word clouds? https://vprusso.github.io/blog/2018/natural-language-processing-python-3/
# include link, but don't make any, they look lame.

# sources
# https://medium.com/@sharonwoo/sentiment-analysis-with-nltk-422e0f794b8
# https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# https://github.com/cjhutto/vaderSentiment#about-the-scoring
# section title: who's line is it anyway??