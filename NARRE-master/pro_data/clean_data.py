import os
import json
import pandas as pd
import pickle
# import dill as pickle
import numpy as np
import codecs
import time
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


# TPS_DIR = '../data/music'
# TP_file = os.path.join(TPS_DIR, 'Digital_Music_5.json')

TPS_DIR = '../data/yelp'
TP_file = os.path.join(TPS_DIR, 'yelp_academic_dataset_review.json')

f = codecs.open(TP_file, encoding='utf-8')
users_id=[]
items_id=[]
ratings=[]
reviews=[]
np.random.seed(2017)

new_lines = []
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
i = 0
total_sum = 3874548
for line in f:
    i = i + 1
    if i % 100000 == 0:
        print("{:.2f} %".format(i/total_sum * 100))
    js = json.loads(line)
    # print(js)
    text = js['text']
    text = clean_str(text)
    js['text'] = text
    new_lines.append(json.dumps(js))

print(i)
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)

with codecs.open(os.path.join(TPS_DIR, 'yelp_academic_dataset_review_tokenized.json'), 'w', encoding='utf-8') as f:
    for line in new_lines:
        f.write(line + '\n')

time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
