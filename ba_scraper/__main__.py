import os
from ba_scraper.models.ConversationList import ConversationList

if __name__ == '__main__':
    transcript_names = os.listdir(os.path.join('episodes'))
    convo_list = ConversationList(transcript_names)
    for convo in convo_list.conversations:
        # print(convo.word_count_summary_json())
        print(convo.sentiment_count_json(-1, -0.5))
        print(convo.sentiment_count_json(-0.5, -0.05))
        print(convo.sentiment_count_json(-0.05, 0.05))
        print(convo.sentiment_count_json(0.05, 0.5))
        print(convo.sentiment_count_json(0.5, 1))
