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
TPS_DIR = '../data/yelp13res_split/filtered'
# TP_file = os.path.join(TPS_DIR, 'raw_train_asp_pol.json')
TP_file = os.path.join(TPS_DIR, 'yelp13_review_asp_pol_res.json')

f = codecs.open(TP_file, encoding='utf-8')
users_id = []
items_id = []
ratings = []
reviews = []
aspects = []
polarity = []
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
    aspects.append(js['aspect'])
    polarity.append(js['polarity'])
    # print type(js['aspect'])


data=pd.DataFrame({'user_id':pd.Series(users_id),
                   'item_id':pd.Series(items_id),
                   'ratings':pd.Series(ratings),
                   'reviews':pd.Series(reviews),
                   'aspects':pd.Series(aspects),
                   'polarity':pd.Series(polarity)})[['user_id','item_id','ratings','reviews','aspects','polarity']]

def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

MIN_USER_COUNT = 1
MIN_SONG_COUNT = 1
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
#     usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
#     return tp, usercount, songcount



# data,usercount, itemcount=filter_triplets(data)
usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')

print(data.shape[0])
print(usercount.shape[0])
print(itemcount.shape[0])

unique_uid = usercount.index
unique_sid = itemcount.index
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))

print("len(user2id):", len(user2id))
print("len(item2id):", len(item2id))

def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp

def data_build(data_file):
    # TP_file_s = os.path.join(TPS_DIR, data_file)

    f = codecs.open( data_file, encoding='utf-8' )
    users_id_s = []
    items_id_s = []
    ratings_s = []
    reviews_s = []
    aspects_s = []
    polarity_s = []

    for line in f:
        # print(line)

        js = json.loads( line )
        if str( js['user_id'] ) == 'unknown':
            print("unknown")
            continue
        if str( js['item_id'] ) == 'unknown':
            print("unknown2")
            continue

        reviews_s.append( js['reviews'] )
        users_id_s.append( str( js['user_id'] ) + ',' )
        items_id_s.append( str( js['item_id'] ) + ',' )
        ratings_s.append( str( js['ratings'] ) )
        aspects_s.append( js['aspect'] )
        polarity_s.append( js['polarity'] )
        # print type(js['aspect'])

    data_s = pd.DataFrame( {'user_id': pd.Series( users_id_s ),
                          'item_id': pd.Series( items_id_s ),
                          'ratings': pd.Series( ratings_s ),
                          'reviews': pd.Series( reviews_s ),
                          'aspects': pd.Series( aspects_s ),
                          'polarity': pd.Series( polarity_s )} )[['user_id', 'item_id', 'ratings', 'reviews', 'aspects', 'polarity']]
    data_s = numerize(data_s)
    return data_s

train_data = data_build(os.path.join(TPS_DIR, 'raw_train_filtered5_asp_pol.json'))
tp_train = train_data[['user_id','item_id','ratings','aspects','polarity']]
valid_data = data_build(os.path.join(TPS_DIR, 'raw_valid_filtered5_asp_pol.json'))
tp_valid = valid_data[['user_id','item_id','ratings','aspects','polarity']]
test_data = data_build(os.path.join(TPS_DIR, 'raw_test_filtered5_asp_pol.json'))
tp_test = test_data[['user_id','item_id','ratings','aspects','polarity']]

# data=numerize(data)
# tp_rating=data[['user_id','item_id','ratings','aspects','polarity']]
#
#
# n_ratings = tp_rating.shape[0]
# test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
# test_idx = np.zeros(n_ratings, dtype=bool)
# test_idx[test] = True
#
# tp_1 = tp_rating[test_idx]
# tp_train= tp_rating[~test_idx]
# data2=data[test_idx]
# data=data[~test_idx]
#
# n_ratings = tp_1.shape[0]
# test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)
#
# test_idx = np.zeros(n_ratings, dtype=bool)
# test_idx[test] = True
#
# tp_test = tp_1[test_idx]
# tp_valid = tp_1[~test_idx]

tp_train.to_csv(os.path.join(TPS_DIR, 'train.csv'), index=False,header=None)
tp_valid.to_csv(os.path.join(TPS_DIR, 'valid.csv'), index=False,header=None)
tp_test.to_csv(os.path.join(TPS_DIR, 'test.csv'), index=False,header=None)

user_reviews={}
item_reviews={}
user_aspects={}
item_aspects={}
user_polarity={}
item_polarity={}
user_rid={}
item_rid={}
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
for i in train_data.values:
    text = i[3]
    # temp = i[4][2:-2].split(',')
    # asp = [int(n) for n in temp]
    # temp = i[5][2:-2].split( ',' )
    # pol = [int( n ) for n in temp]`
    if i[0] in user_reviews:
        user_reviews[i[0]].append(text)
        user_rid[i[0]].append(i[1])
        user_aspects[i[0]].append(i[4])
        user_polarity[i[0]].append(i[5])
    else:
        user_rid[i[0]]=[i[1]]
        user_reviews[i[0]]=[text]
        user_aspects[i[0]]=[i[4]]
        user_polarity[i[0]]=[i[5]]
    if i[1] in item_reviews:
        item_reviews[i[1]].append(text)
        item_rid[i[1]].append(i[0])
        item_aspects[i[1]].append( i[4] )
        item_polarity[i[1]].append( i[5] )
    else:
        item_reviews[i[1]] = [text]
        item_rid[i[1]]=[i[0]]
        item_aspects[i[1]] = [i[4]]
        item_polarity[i[1]] = [i[5]]

print("train_data done")
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
# print len(valid_data)
# data2 = valid_data.extend(test_data)
# print len(data2)
for i in valid_data.values:
    # print i
    if i[0] == 8077:
        print 1111
    if i[0] in user_reviews:
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=[""]
        user_aspects[i[0]]=[i[4]]
        user_polarity[i[0]] = [i[5]]
    if i[1] in item_reviews:
        l=1
    else:
        item_reviews[i[1]] = [""]  #???
        item_rid[i[1]]=[0]
        item_aspects[i[1]] = [i[4]]
        item_polarity[i[1]] = [i[5]]
for i in test_data.values:
    # print i
    if i[0] == 8077:
        print 1111
    if i[0] in user_reviews:
        l=1
    else:
        user_rid[i[0]]=[0]
        user_reviews[i[0]]=[""]
        user_aspects[i[0]]=[i[4]]
        user_polarity[i[0]] = [i[5]]
    if i[1] in item_reviews:
        l=1
    else:
        item_reviews[i[1]] = [""]  #???
        item_rid[i[1]]=[0]
        item_aspects[i[1]] = [i[4]]
        item_polarity[i[1]] = [i[5]]

print(item_reviews[11])
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
pickle.dump(user_reviews, open(os.path.join(TPS_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TPS_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TPS_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TPS_DIR, 'item_rid'), 'wb'))
pickle.dump(user_aspects, open(os.path.join(TPS_DIR, 'user_aspects'), 'wb'))
pickle.dump(item_aspects, open(os.path.join(TPS_DIR, 'item_aspects'), 'wb'))
pickle.dump(user_polarity, open(os.path.join(TPS_DIR, 'user_polarity'), 'wb'))
pickle.dump(item_polarity, open(os.path.join(TPS_DIR, 'item_polarity'), 'wb'))

time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)

usercount, itemcount = get_count(train_data, 'user_id'), get_count(train_data, 'item_id')


print(np.sort(np.array(usercount.values)))

print(np.sort(np.array(itemcount.values)))
time_stamp = time.asctime().replace(':', '_').split()
print(time_stamp)
