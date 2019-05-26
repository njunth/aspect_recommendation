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
TPS_DIR = '../data/yelp13res'
TP_file = os.path.join(TPS_DIR, 'yelp13_review_res.json')

f = codecs.open(TP_file, encoding='utf-8')
users_id = []
items_id = []
ratings = []
reviews_id = []
reviews = []
np.random.seed(2017)

for line in f:
    # print(line)

    js=json.loads(line)
    if str(js['user_id'])=='unknown':
        print("unknown")
        continue
    if str(js['business_id'])=='unknown':
        print("unknown2")
        continue

    reviews.append(js['text'])
    users_id.append(str(js['user_id'])+',')
    items_id.append(str(js['business_id'])+',')
    ratings.append(str(js['stars']))
    reviews_id.append(str(js['review_id']))

data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews),
                   'review_id': pd.Series(reviews_id)})[['user_id','item_id','ratings','reviews','review_id']]

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

# MIN_USER_COUNT = 1
# MIN_SONG_COUNT = 1
# def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT):
#     # Only keep the triplets for songs which were listened to by at least min_sc users.
#     songcount = get_count(tp, 'item_id')
#
#     tp = tp[tp['item_id'].isin(songcount.index[songcount >= min_sc])]
#
#     # Only keep the triplets for users who listened to at least min_uc songs
#     # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
#     usercount = get_count(tp, 'user_id')
#
#     tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]
#
#     # Update both usercount and songcount after filtering
#     # usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
#     return tp, usercount, songcount
#
#
# data,usercount, itemcount=filter_triplets(data)
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


data = numerize(data)
tp_rating = data[['user_id','item_id','ratings']]


n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train= tp_rating[~test_idx]
data2=data[test_idx]
data=data[~test_idx]

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]
tp_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'test.csv'), index=False,header=None)

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
for i in data.values:
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
for i in data2.values:
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

usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')


print(np.sort(np.array(usercount.values)))

print(np.sort(np.array(itemcount.values)))
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
