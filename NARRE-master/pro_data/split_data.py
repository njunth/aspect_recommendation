'''
Data pre process

@author:
Chong Chen (cstchenc@163.com)

@ created:
25/8/2017
@references:
'''
import os
import json
import pandas as pd
import pickle
import numpy as np
import codecs
import time
import re
TPS_DIR = '../data/yelp'
TP_file = os.path.join(TPS_DIR, 'yelp_academic_dataset_review.json')

f = codecs.open(TP_file, encoding='utf-8')
users_id = []
items_id = []
ratings = []
reviews_id = []
reviews = []
np.random.seed(2017)

line_count = 0
for line in f:
    if line_count % 1000000 == 0:
        print(line_count)
    line_count += 1
    # line_count += 1
    # print(line)

    js = json.loads(line)
    if str(js['user_id'])=='unknown':
        print("unknown")
        continue
    if str(js['business_id'])=='unknown':
        print("unknown2")
        continue

    reviews.append(js['text'])
    users_id.append(str(js['user_id']))
    items_id.append(str(js['business_id']))
    ratings.append(str(js['stars']))
    reviews_id.append(str(js['review_id']))

f.close()

data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews),
                   'review_id': pd.Series(reviews_id)})[['user_id','item_id','ratings','reviews','review_id']]

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

MIN_USER_COUNT = 10
MIN_SONG_COUNT = 10
def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT):
    # Only keep the triplets for songs which were listened to by at least min_sc users.
    itemcount = get_count(tp, 'item_id')

    tp = tp[tp['item_id'].isin(itemcount.index[itemcount >= min_sc])]

    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    usercount = get_count(tp, 'user_id')

    tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and songcount after filtering
    # usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
    return tp, usercount, itemcount
#
#
data,usercount, itemcount = filter_triplets(data)
usercount, itemcount, reviewcount = get_count(data, 'user_id'), get_count(data, 'item_id'), get_count(data, 'review_id')

print(data.shape[0])
print(usercount.shape[0])
print(itemcount.shape[0])
print(reviewcount.shape[0])
# exit(-1)

unique_uid = usercount.index
unique_sid = itemcount.index
unique_rid = reviewcount.index
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))
review2id = dict((rid, i) for (i, rid) in enumerate(unique_rid))

print("len(user2id):", len(user2id))
print("len(item2id):", len(item2id))


# def numerize(tp):
#     uid = list(map(lambda x: user2id[x], tp['user_id']))
#     sid = list(map(lambda x: item2id[x], tp['item_id']))
#     # rid = list(map(lambda x: review2id[x], tp['review_id']))
#     tp['user_id'] = uid
#     tp['item_id'] = sid
#     # tp['review_id'] = rid
#     return tp


# data = numerize(data)
# tp_rating = data[['user_id','item_id','ratings','reviews','review_id']]


n_ratings = data.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

data2=data[test_idx]
data_train=data[~test_idx]

n_ratings = data2.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

data_test = data2[test_idx]
data_valid = data2[~test_idx]

header = ['user_id','item_id','ratings','reviews','review_id']
data_train.to_json(os.path.join(TPS_DIR, 'raw_train.json'), orient='records', lines=True)
data_valid.to_json(os.path.join(TPS_DIR, 'raw_valid.json'), orient='records', lines=True)
data_test.to_json(os.path.join(TPS_DIR, 'raw_test.json'), orient='records', lines=True)