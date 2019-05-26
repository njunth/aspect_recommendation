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
# TP_file = os.path.join(TPS_DIR, 'yelp_academic_dataset_review.json')

np.random.seed(2017)

def load_data(TP_file):
    f = codecs.open(TP_file, encoding='utf-8')
    users_id = []
    items_id = []
    ratings = []
    reviews_id = []
    reviews = []
    for line in f:
        # print(line)

        js=json.loads(line)
        if str(js['user_id'])=='unknown':
            print("unknown")
            continue
        if str(js['item_id'])=='unknown':
            print("unknown2")
            continue

        reviews.append(js['reviews'])
        users_id.append(str(js['user_id']))
        items_id.append(str(js['item_id']))
        ratings.append(str(js['ratings']))
        reviews_id.append(str(js['review_id']))

    f.close()

    data=pd.DataFrame({'user_id': pd.Series(users_id),
                       'item_id': pd.Series(items_id),
                       'ratings': pd.Series(ratings),
                       'reviews': pd.Series(reviews),
                       'review_id': pd.Series(reviews_id)})[['user_id','item_id','ratings','reviews','review_id']]
    return data


data_train = load_data(os.path.join(TPS_DIR, 'raw_train.json'))
data_valid = load_data(os.path.join(TPS_DIR, 'raw_valid.json'))
data_test = load_data(os.path.join(TPS_DIR, 'raw_test.json'))

data = data_train.append([data_valid, data_test])


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count
#
#
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


def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    # rid = list(map(lambda x: review2id[x], tp['review_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    # tp['review_id'] = rid
    return tp


# data = numerize(data)
data_train = numerize(data_train)
data_valid = numerize(data_valid)
data_test = numerize(data_test)

tp_train = data_train[['user_id','item_id','ratings']]
tp_valid = data_valid[['user_id','item_id','ratings']]
tp_test = data_test[['user_id','item_id','ratings']]


tp_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'test.csv'), index=False,header=None)

para = {}
para['user_num'] = len(user2id)
para['item_num'] = len(item2id)
output = open(os.path.join(TPS_DIR, 'data_2.para'), 'wb')

# Pickle dictionary using protocol 0.
pickle.dump(para, output)
exit(-1)

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}

user_features={}
item_features={}
time_stamp = time.asctime().replace(':', '_').split()

with open(os.path.join(TPS_DIR, "bert_0.9034.pt_bert_0.9168_polarity.txt_feature_dict.pkl"), 'rb') as f_pf:
    polarity_feature_dict = pickle.load(f_pf)

print(time_stamp)
for i in data_train.values:
    text = i[3]
    review_id = i[4]
    if review_id in polarity_feature_dict:
        u_feature = {'iid': i[1], 'rid': review_id, 'features': polarity_feature_dict[review_id], 'rating': i[2]}
    else:
        u_feature = {'iid': i[1], 'rid': review_id, 'features': None, 'rating': i[2]}
        print(review_id)
    if review_id in polarity_feature_dict:
        i_feature = {'uid':i[0], 'rid':review_id, 'features': polarity_feature_dict[review_id], 'rating': i[2]}
    else:
        i_feature = {'uid': i[0], 'rid': review_id, 'features': None, 'rating': i[2]}
        print(review_id)
    if i[0] in user_reviews:
        user_reviews[i[0]].append(text)
        user_rid[i[0]].append(i[1])
        user_features[i[0]].append(u_feature)
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[text]
        user_features[i[0]] = [u_feature]
    if i[1] in item_reviews:
        item_reviews[i[1]].append(text)
        item_rid[i[1]].append(i[0])
        item_features[i[1]].append(i_feature)
    else:
        item_reviews[i[1]] = [text]
        item_rid[i[1]]=[i[0]]
        item_features[i[1]] = [i_feature]

print("data done")
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
for i in data_valid.values:
    # print(i)
    if i[0] in user_reviews:
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=[""]
        user_features[i[0]] = None
    if i[1] in item_reviews:
        l=1
    else:
        item_reviews[i[1]] = [""]  #???
        item_rid[i[1]]=[0]
        item_features[i[1]] = None

for i in data_test.values:
    # print(i)
    if i[0] in user_reviews:
        l = 1
    else:
        user_rid[i[0]] = [0]
        user_reviews[i[0]] = [""]
        user_features[i[0]] = None
    if i[1] in item_reviews:
        l = 1
    else:
        item_reviews[i[1]] = [""]  # ???
        item_rid[i[1]] = [0]
        item_features[i[1]] = None

# print(item_reviews[11])
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))

pickle.dump(user_features, open(os.path.join(TPS_DIR, 'user_features'), 'wb'))
pickle.dump(item_features, open(os.path.join(TPS_DIR, 'item_features'), 'wb'))

time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)

usercount, itemcount = get_count(data_train, 'user_id'), get_count(data_train, 'item_id')


print(np.sort(np.array(usercount.values)))

print(np.sort(np.array(itemcount.values)))
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
