import os
from ba_scraper.models.ConversationList import ConversationList

if __name__ == '__main__':
    transcript_names = os.listdir(os.path.join('episodes'))
    convo_list = ConversationList(transcript_names)
    for convo in convo_list.conversations:
        print(convo.word_count_summary_json())
