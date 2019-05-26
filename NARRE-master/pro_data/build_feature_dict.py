import os
import json
import pandas as pd
import pickle
import numpy as np
import codecs
import time
import re
import ast
TPS_DIR = '../data/yelp13res_filtered'
TP_file = "bert_0.9168_aspect.txt"
TP_file2 = "bert_0.9034.pt_bert_0.9168_polarity.txt"
f_aspect_dict = os.path.join(TPS_DIR, 'attribute.json')
f_polarity_dict = os.path.join(TPS_DIR, 'polarity.json')

f = codecs.open(os.path.join(TPS_DIR, TP_file), encoding='utf-8')
f2 = codecs.open(os.path.join(TPS_DIR, TP_file2), encoding='utf-8')

aspect_dict = json.load(open(f_aspect_dict))
polarity_dict = json.load(open(f_polarity_dict))


def build_aspect_feature():
    feature_dict = {}
    count = 0
    for line1, line2 in zip(f, f2):
        if count % 100000 == 0:
            print(count)
        count += 1
        splits = line1.strip().split('\t')
        review_id1 = splits[0]
        text_length1 = len(splits[1].split())
        aspects = splits[2].split('|')
        aspects = [aspect_dict[a] for a in aspects]
        vector = ast.literal_eval(splits[3])
        predicted_aspect_result = {'aspects': aspects, 'vector_a': vector}

        splits2 = line2.strip().split('\t')
        review_id2 = splits[0]
        text_length2 = len(splits[1].split())
        predicted_polarity_result = []
        for result in splits2[2:]:
            aspect, polarity, vector = result.split('#')
            vector = ast.literal_eval(vector)
            aspect = aspect_dict[aspect]
            polarity = polarity_dict[polarity]
            predicted_polarity_result.append({'aspect': aspect, 'polarity': polarity, 'vector_p': vector})
        assert review_id1 == review_id2
        assert text_length1 == text_length2

        if review_id1 in feature_dict:
            feature_dict[review_id1].append({'text_length':text_length1, 'predicted_a':predicted_aspect_result,
                                             'predicted_p':predicted_polarity_result})
        else:
            feature_dict[review_id1] = [{'text_length': text_length1, 'predicted_a': predicted_aspect_result,
                                         'predicted_p': predicted_polarity_result}]
    print(feature_dict['8s-JTklvbLgnijqH9AT6tw'])
    pickle.dump(feature_dict, open(os.path.join(TPS_DIR, TP_file2+'_feature_dict.pkl'), 'wb'))


def build_polarity_feature():
    feature_dict = {}
    count = 0
    for line in f:
        if count % 100000 == 0:
            print(count)
        count += 1
        splits = line.strip().split('\t')
        review_id = splits[0]
        text_length = len(splits[1].split())
        predicted_result = []
        for result in splits[2:]:
            aspect, polarity, vector = result.split('#')
            vector = ast.literal_eval(vector)
            aspect = aspect_dict[aspect]
            polarity = polarity_dict[polarity]
            predicted_result.append({'aspect': aspect, 'polarity': polarity, 'vector': vector})
        if review_id in feature_dict:
            feature_dict[review_id].append({'text_length':text_length, 'predicted':predicted_result})
        else:
            feature_dict[review_id] = [{'text_length':text_length, 'predicted':predicted_result}]
    print(feature_dict['8s-JTklvbLgnijqH9AT6tw'])
    pickle.dump(feature_dict, open(os.path.join(TPS_DIR, TP_file+'_feature_dict.pkl'), 'wb'))



if __name__ == '__main__':
    # build_polarity_feature()
    build_aspect_feature()