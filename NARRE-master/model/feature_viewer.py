import os
import json
import pandas as pd
import pickle
import numpy as np
import codecs
import time
import re
TPS_DIR = '../data/yelp13res_filtered'
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

# values = data[data['user_id'] == '67QGJdqABqaSEsQR4esrsA,'].values
# for value in values:
#     print(value)
# exit(-1)


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count

MIN_USER_COUNT = 5
MIN_SONG_COUNT = 5
def filter_triplets(tp, min_uc=MIN_USER_COUNT, min_sc=MIN_SONG_COUNT):
    # Only keep the triplets for songs which were listened to by at least min_sc users.
    songcount = get_count(tp, 'item_id')

    tp = tp[tp['item_id'].isin(songcount.index[songcount >= min_sc])]

    # Only keep the triplets for users who listened to at least min_uc songs
    # After doing this, some of the songs will have less than min_uc users, but should only be a small proportion
    usercount = get_count(tp, 'user_id')

    tp = tp[tp['user_id'].isin(usercount.index[usercount >= min_uc])]

    # Update both usercount and songcount after filtering
    # usercount, songcount = get_count(tp, 'user_id'), get_count(tp, 'item_id')
    return tp, usercount, songcount


data,usercount, itemcount=filter_triplets(data)
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

user_reviews={}
item_reviews={}
user_rid={}
item_rid={}

user_features={}
item_features={}
time_stamp = time.asctime().replace(':', '_').split()

with open(os.path.join(TPS_DIR, "bert_0.9034.pt_bert_0.9168_polarity.txt_feature_dict.pkl"), 'rb') as f_pf:
    polarity_feature_dict = pickle.load(f_pf)

for i in data.values:
    text = i[3]
    review_id = i[4]
    if review_id in polarity_feature_dict:
        u_feature = {'iid': i[1], 'rid': review_id, 'features': polarity_feature_dict[review_id], 'text': text, 'rating': i[2]}
    else:
        u_feature = {'iid': i[1], 'rid': review_id, 'features': None, 'text': text, 'rating': i[2]}
        print(review_id)
    if review_id in polarity_feature_dict:
        i_feature = {'uid':i[0], 'rid':review_id, 'features': polarity_feature_dict[review_id], 'text': text, 'rating': i[2]}
    else:
        i_feature = {'uid': i[0], 'rid': review_id, 'features': None, 'text': text, 'rating': i[2]}
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


def watch():
    instance = data.sample(1).values[0]
    user_id = instance[0]
    item_id = instance[1]
    print(instance)
    print("-----------------user features:-----------------")
    print(user_id)
    for feature in user_features[user_id]:
        print(feature['iid'], end='\t')
        print(feature['rid'], end='\t')
        print(feature['rating'])
        features = feature['features']
        print([feature['text']])
        for feat in features:
            predict_a = ['a:{}#p:{}'.format(p["aspect"], p['polarity']) for p in feat['predicted_p']]
            print(predict_a, end='  ')
        print()
        # print(feature)
    print("-----------------item features:-----------------")
    print(item_id)
    for feature in item_features[item_id]:
        print(feature['uid'], end='\t')
        print(feature['rid'], end='\t')
        print(feature['rating'])
        features = feature['features']
        print([feature['text']])
        for feat in features:
            predict_a = ['a:{}#p:{}'.format(p["aspect"], p['polarity']) for p in feat['predicted_p']]
            print(predict_a, end='  ')
        print()


def watch_feature():
    user_id = '67QGJdqABqaSEsQR4esrsA,'
    item_id = 'cBpJIOrVXotDI0XAZH_k0g,'
    # user_id = 'hTKFGpi3ltCV4B-XDFRT-A,'
    # item_id = 'd3MxUXS1b6U2P_gGuCO1-A,'
    build_user_feature(user_features[user_id])
    print()
    build_item_feature(item_features[item_id])


def build_user_feature(reviews_polarity_feature, aspect_size=5):
    pos_fields = {'0': [], '1': [], '2': [], '3': [], '4': []}
    neg_fields = {'0': [], '1': [], '2': [], '3': [], '4': []}
    count_tensor = np.zeros((aspect_size, 3))
    if reviews_polarity_feature is None:
        return np.zeros(aspect_size)
    else:
        for review in reviews_polarity_feature:
            uid = review['iid']
            rid = review['rid']
            rating = int(review['rating'])
            sentence_features = review['features']
            if sentence_features is None:
                continue
            for s_feature in sentence_features:
                text_length = s_feature['text_length']
                predicted_a = s_feature['predicted_a']
                predicted_p = s_feature['predicted_p']
                for apv in predicted_p:
                    aspect = str(apv['aspect'])
                    polarity = apv['polarity']
                    polarity_vector = apv['vector_p']
                    if polarity == 2:
                        pos_fields[aspect].append(rating)
                    elif polarity == 0:
                        neg_fields[aspect].append(rating)
    pos_mean = np.zeros(aspect_size)
    neg_mean = np.zeros(aspect_size)
    for key in pos_fields:
        print(key + ':' + str(np.mean(pos_fields[key])) + ':' + str(pos_fields[key]))
        mean_rating = np.mean(pos_fields[key]) if pos_fields[key] != [] else 3
        pos_mean[int(key)] = mean_rating
    for key in neg_fields:
        print(key + ':' + str(np.mean(neg_fields[key])) + ':' + str(neg_fields[key]))
        mean_rating = np.mean(neg_fields[key]) if neg_fields[key] != [] else 3
        neg_mean[int(key)] = mean_rating
    aspect_sensitive = np.exp(pos_mean - neg_mean)
    print(aspect_sensitive)
    return aspect_sensitive


def build_item_feature(reviews_polarity_feature, aspect_size=5):
    # pos_fields = {'0': [], '1': [], '2': [], '3': [], '4': []}
    neg_fields = {'0': [], '1': [], '2': [], '3': [], '4': []}
    count_tensor = np.zeros((aspect_size, 3))
    if reviews_polarity_feature is None:
        return np.zeros(aspect_size)
    else:
        for review in reviews_polarity_feature:
            uid = review['uid']
            rid = review['rid']
            rating = int(review['rating'])
            sentence_features = review['features']
            if sentence_features is None:
                continue
            for s_feature in sentence_features:
                text_length = s_feature['text_length']
                predicted_a = s_feature['predicted_a']
                predicted_p = s_feature['predicted_p']
                for apv in predicted_p:
                    aspect = str(apv['aspect'])
                    polarity = int(apv['polarity'])
                    polarity_vector = apv['vector_p']
                    # pos_fields[aspect].append(polarity)
                    if polarity == 2:
                        score = np.exp(rating - 3)*1
                        neg_fields[aspect].append(score)
                    elif polarity == 0:
                        score = np.exp(3- rating)*-1
                        neg_fields[aspect].append(score)
    # for key in pos_fields:
    #     print(key + ':' + str(sum(pos_fields[key])/len(pos_fields[key])) + ':' + str(pos_fields[key]))
    aspect_characteristics = np.zeros(aspect_size)
    for key in neg_fields:
        print(key + ':' + str(sum(neg_fields[key])/len(neg_fields[key])) + ':' + str(neg_fields[key]))
        mean_rating = np.mean(neg_fields[key]) if neg_fields[key] != [] else 3
        aspect_characteristics[int(key)] = mean_rating
    print(aspect_characteristics)
    return aspect_characteristics


watch_feature()
# for i in range(10):
#     input('Press enter to continue: ')
#     watch()