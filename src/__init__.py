import os
from models import Conversation

if __name__ == '__main__':
    transcript_names = os.listdir(os.path.join('src', 'episodes'))
    conversations = [Conversation(fpath) for fpath in transcript_names]
    for convo in conversations:
        convo.summarize()
