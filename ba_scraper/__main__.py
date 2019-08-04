import os
from ba_scraper.models import ConversationList

if __name__ == '__main__':
    transcript_names = os.listdir(os.path.join('episodes'))
    convo_list = ConversationList(transcript_names)
    for convo in convo_list.conversations:
        convo.summarize()

# generate random text in Chris voice (ML, all ep)
# predictors for Chris vs guest (ML, all ep)
# most common words / bigrams for speakers (esp chris) (all ep)
    # word clouds? https://vprusso.github.io/blog/2018/natural-language-processing-python-3/

# sources
# https://medium.com/@sharonwoo/sentiment-analysis-with-nltk-422e0f794b8
# https://www.analyticsvidhya.com/blog/2018/03/text-generation-using-python-nlp/
# https://github.com/cjhutto/vaderSentiment#about-the-scoring
