import numpy as np
import re
import itertools
from collections import Counter

import tensorflow as tf
import csv
import pickle
# import cPickle as pickle
import os
import time
import sys

TPS_DIR = '../data/yelp13res_split/filtered'

tf.flags.DEFINE_string("valid_data", "%s/test.csv" % TPS_DIR, " Data for validation")
tf.flags.DEFINE_string("test_data", "%s/test.csv" % TPS_DIR, "Data for testing")
tf.flags.DEFINE_string("train_data", "%s/train.csv" % TPS_DIR, "Data for training")
tf.flags.DEFINE_string("user_review", "%s/user_review" % TPS_DIR, "User's reviews")
tf.flags.DEFINE_string("item_review", "%s/item_review" % TPS_DIR, "Item's reviews")
tf.flags.DEFINE_string("user_review_id", "%s/user_rid" % TPS_DIR, "user_review_id")
tf.flags.DEFINE_string("item_review_id", "%s/item_rid" % TPS_DIR, "item_review_id")
tf.flags.DEFINE_string("user_aspects", "%s/user_aspects" % TPS_DIR, "User's aspects")
tf.flags.DEFINE_string("item_aspects", "%s/item_aspects" % TPS_DIR, "Item's aspects")
tf.flags.DEFINE_string("user_polarity", "%s/user_polarity" % TPS_DIR, "user_polarity")
tf.flags.DEFINE_string("item_polarity", "%s/item_polarity" % TPS_DIR, "item_polarity")

tf.flags.DEFINE_string("stopwords", "../data/stopwords", "stopwords")

# tf.flags.DEFINE_string("valid_data", "../data/yelp/yelp_valid.csv", " Data for validation")
# tf.flags.DEFINE_string("test_data", "../data/yelp/yelp_test.csv", "Data for testing")
# tf.flags.DEFINE_string("train_data", "../data/yelp/yelp_train.csv", "Data for training")
# tf.flags.DEFINE_string("user_review", "../data/yelp/user_review", "User's reviews")
# tf.flags.DEFINE_string("item_review", "../data/yelp/item_review", "Item's reviews")
# tf.flags.DEFINE_string("user_review_id", "../data/yelp/user_rid", "user_review_id")
# tf.flags.DEFINE_string("item_review_id", "../data/yelp/item_rid", "item_review_id")
# tf.flags.DEFINE_string("stopwords", "../data/stopwords", "stopwords")


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


def pad_sentences(u_text, u_len, u2_len, user_aspects, user_polarity, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len
    review_len = u2_len
    print review_num, review_len

    u_text2 = {}
    user_aspects2 = {}
    user_polarity2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        u_aspects = user_aspects[i]
        padded_u_aspects = []
        u_polarity = user_polarity[i]
        padded_u_polarity = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                padded_u_aspects.append(u_aspects[ri])
                padded_u_polarity.append(u_polarity[ri])
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:
                pad_vec = [0 for n in range( 5 )]
                padded_u_aspects.append(pad_vec)
                padded_u_polarity.append(pad_vec)
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train
        user_aspects2[i] = padded_u_aspects
        user_polarity2[i] = padded_u_polarity
    return u_text2, user_aspects2, user_polarity2


def pad_reviewid(u_train, u_valid, u_len, num):
    pad_u_train = []

    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_train.append(x)
    pad_u_valid = []

    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        pad_u_valid.append(x)
    return pad_u_train, pad_u_valid


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    l = len(u_text)
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array([[vocabulary_u[word] for word in words] for words in u_reviews])
        u_text2[i] = u
    l = len(i_text)
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array([[vocabulary_i[word] for word in words] for words in i_reviews])
        i_text2[j] = i
    return u_text2, i_text2


def load_data(train_data, valid_data, user_review, item_review, user_rid, item_rid, user_aspects,
                        item_aspects, user_polarity, item_polarity, stopwords):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    user_aspects, item_aspects, user_polarity, item_polarity, u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num \
        , reid_user_train, reid_item_train, reid_user_valid, reid_item_valid = \
        load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, user_aspects,
                        item_aspects, user_polarity, item_polarity, stopwords)
    print("load data done")
    u_text, user_aspects, user_polarity = pad_sentences(u_text, u_len, u2_len, user_aspects, user_polarity)
    reid_user_train, reid_user_valid = pad_reviewid(reid_user_train, reid_user_valid, u_len, item_num + 1)

    print("pad user done")
    i_text, item_aspects, item_polarity = pad_sentences(i_text, i_len, i2_len, item_aspects, item_polarity)
    reid_item_train, reid_item_valid = pad_reviewid(reid_item_train, reid_item_valid, i_len, user_num + 1)

    print("pad item done")

    user_voc = [xx for x in u_text.values() for xx in x]
    item_voc = [xx for x in i_text.values() for xx in x]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
    print("Vocabulary_user:", len(vocabulary_user))
    print("vocabulary_item:", len(vocabulary_item))
    u_text, i_text = build_input_data(u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)

    return [user_aspects, item_aspects, user_polarity, item_polarity, u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item,
            vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train,
            reid_item_train, reid_user_valid, reid_item_valid]


def clean(user_reviews):
    for (key, value) in user_reviews.items():
        new_value = []
        for s in value:
            s1 = clean_str(s)
            s1 = s1.split(" ")
            new_value.append(s1)
        user_reviews[key] = new_value
    return user_reviews


def load_data_and_labels(train_data, valid_data, user_review, item_review, user_rid, item_rid, user_aspects,
                        item_aspects, user_polarity, item_polarity, stopwords):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    f_train = open(train_data, "r")
    # f1 = open(user_review, 'rb')
    # f2 = open(item_review, 'rb')
    # f3 = open(user_rid, 'rb')
    # f4 = open(item_rid, 'rb')
    print("open train_data")
    with open(user_review, 'rb') as f1:
        # user_reviews = pickle.load(f1, encoding='utf-8')
        user_reviews = pickle.load(f1)
        print("load user review")
    with open(item_review, 'rb') as f2:
        # item_reviews = pickle.load(f2, encoding='utf-8')
        item_reviews = pickle.load(f2)
        print("load item review")
    with open(user_rid, 'rb') as f3:
        # user_rids = pickle.load(f3, encoding='utf-8')
        user_rids = pickle.load(f3)
        print("load user_rid")
    with open(item_rid, 'rb') as f4:
        # item_rids = pickle.load(f4, encoding='utf-8')
        item_rids = pickle.load(f4)
        print("load item_rid")
    with open(user_aspects, 'rb') as f5:
        # user_reviews = pickle.load(f1, encoding='utf-8')
        user_aspects = pickle.load(f5)
        print("load user user_aspects")
    with open(item_aspects, 'rb') as f6:
        # item_reviews = pickle.load(f2, encoding='utf-8')
        item_aspects = pickle.load(f6)
        print("load item item_aspects")
    with open(user_polarity, 'rb') as f7:
        # user_rids = pickle.load(f3, encoding='utf-8')
        user_polarity = pickle.load(f7)
        print("load user_polarity")
    with open(item_polarity, 'rb') as f8:
        # item_rids = pickle.load(f4, encoding='utf-8')
        item_polarity = pickle.load(f8)
        print("load item_polarity")
    # with open(user_review, 'rb') as f1:
    #     user_reviews = pickle.load(f1, encoding='utf-8')
    #     print("load user review")
    # with open(item_review, 'rb') as f2:
    #     item_reviews = pickle.load(f2, encoding='utf-8')
    #     print("load item review")
    # with open(user_rid, 'rb') as f3:
    #     user_rids = pickle.load(f3, encoding='utf-8')
    #     print("load user_rid")
    # with open(item_rid, 'rb') as f4:
    #     item_rids = pickle.load(f4, encoding='utf-8')
    #     print("load item_rid")

    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)

    reid_user_train = [] # id->all_reviewed_item_id
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    u_text = clean(user_reviews)  # u->text
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)
    u_rid = user_rids
    i_text = clean(item_reviews)
    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)
    i_rid = item_rids
    i = 0
    lines_length = 3099639

    print(len(u_text))
    print(u_text.keys())
    # print(u_text[100])
    aspect_train = []
    polarity_train = []

    for line in f_train:
        i = i + 1
        if i % 10000 == 0:
            print("{:.2f}%".format(i/lines_length * 100))
        line = line.replace("\"", "").replace("[", "").replace("]", "")
        # print line
        line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        # if int(line[0]) not in u_text:
        #     u_text[int(line[0])] = []
        #     for s in user_reviews[int(line[0])]:
        #         s1 = clean_str(s)
        #         s1 = s1.split(" ")
        #         u_text[int(line[0])].append(s1)
        #     u_rid[int(line[0])] = []
        #     for s in user_rids[int(line[0])]:
        #         u_rid[int(line[0])].append(int(s))

        # print(int(line[0]))
        # print(user_review[int(line[0])])
        # print line
        reid_user_train.append(u_rid[int(line[0])])

        # if int(line[1]) in i_text:
        #     reid_item_train.append(i_rid[int(line[1])])  #####write here
        # else:
        #     i_text[int(line[1])] = []
        #     for s in item_reviews[int(line[1])]:
        #         s1 = clean_str(s)
        #         s1 = s1.split(" ")
        #
        #         i_text[int(line[1])].append(s1)
        #     i_rid[int(line[1])] = []
        #     for s in item_rids[int(line[1])]:
        #         i_rid[int(line[1])].append(int(s))

        reid_item_train.append(i_rid[int(line[1])])
        y_train.append(float(line[2]))
        aspect = [int(j) for j in line[3:8]]
        polarity = [int(j) for j in line[8:13]]
        # print aspect, polarity
        aspect_train.append(aspect)
        polarity_train.append(polarity)

    print("valid")
    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    f_valid = open(valid_data)
    aspect_valid = []
    polarity_valid = []
    for line in f_valid:
        line = line.replace( "\"", "" ).replace( "[", "" ).replace( "]", "" )
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        # if int(line[0]) in u_text:
        #     reid_user_valid.append(u_rid[int(line[0])])
        # else:
        #     u_text[int(line[0])] = [['<PAD/>']]
        #     u_rid[int(line[0])] = [int(0)]
        reid_user_valid.append(u_rid[int(line[0])])

        # if int(line[1]) in i_text:
        #     reid_item_valid.append(i_rid[int(line[1])])
        # else:
        #     i_text[int(line[1])] = [['<PAD/>']]
        #     i_rid[int(line[1])] = [int(0)]
        reid_item_valid.append(i_rid[int(line[1])])

        y_valid.append(float(line[2]))
        aspect = [int(j) for j in line[3:8]]
        polarity = [int(j) for j in line[8:13]]
        # print aspect, polarity
        aspect_valid.append(aspect)
        polarity_valid.append(polarity)
    print("len")

    time_stamp = time.asctime().replace(':', '_').split()
    print(time_stamp)

    review_num_u = np.array([len(x) for x in u_text.values()])
    x = np.sort(review_num_u)
    u_len = x[int(0.9 * len(review_num_u)) - 1]
    review_len_u = np.array([len(j) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text.values()])
    y = np.sort(review_num_i)
    i_len = y[int(0.9 * len(review_num_i)) - 1]
    review_len_i = np.array([len(j) for i in i_text.values() for j in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print("u_len (user_review max num):", u_len, x[-1])
    print("i_len (item_review max num):", i_len, y[-1])
    print("u2_len (user_review max length):", u2_len, x2[-1])
    print("i2_len (item_review max length):", i2_len, y2[-1])
    user_num = len(u_text)
    item_num = len(i_text)
    print("user_num:", user_num)
    print("item_num:", item_num)
    return [user_aspects, item_aspects, user_polarity, item_polarity, u_text, i_text, y_train, y_valid, u_len, i_len, u2_len, i2_len, uid_train,
            iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid]


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    # FLAGS(sys.argv)
    FLAGS._parse_flags()
    user_aspects, item_aspects, user_polarity, item_polarity, u_text, i_text, y_train, y_valid, vocabulary_user, vocabulary_inv_user, vocabulary_item, \
     vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, user_num, item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.user_review, FLAGS.item_review, FLAGS.user_review_id,
                  FLAGS.item_review_id, FLAGS.user_aspects, FLAGS.item_aspects, FLAGS.user_polarity, FLAGS.item_polarity, FLAGS.stopwords)

    # np.random.seed(2017)
    #
    # shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train
    itemid_train = iid_train
    # userid_train = uid_train[shuffle_indices]
    # itemid_train = iid_train[shuffle_indices]
    # y_train = y_train[shuffle_indices]
    # reid_user_train = reid_user_train[shuffle_indices]
    # reid_item_train = reid_item_train[shuffle_indices]
    # aspect_train = aspect_train[shuffle_indices]
    # polarity_train = polarity_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]
    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    batches_train = list(
        zip(userid_train, itemid_train, reid_user_train, reid_item_train, y_train))
    batches_test = list(zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid, y_valid))
    print('write begin')
    print("data.train"+str(batches_train[0]))
    output = open(os.path.join(TPS_DIR, 'data.train'), 'wb')
    pickle.dump(batches_train, output)
    output = open(os.path.join(TPS_DIR, 'data.test'), 'wb')
    pickle.dump(batches_test, output)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    para['review_num_u'] = u_text[0].shape[0]
    # print(u_text[0])
    # print(u_text[1])
    # print(u_text[0].shape[0])
    # print(u_text[0].shape[1])
    # print(u_text[1].shape[0])
    # print(u_text[1].shape[1])
    # print(user_aspects[0])
    # print(item_aspects[0])
    # print(user_polarity[0])
    # print(item_polarity[0])
    # print len(user_aspects[0])
    # print len(user_aspects[1])
    # print len(user_aspects[2])
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[0].shape[1]
    para['review_len_i'] = i_text[0].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['test_length'] = len(y_valid)
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['user_aspects'] = user_aspects
    para['item_aspects'] = item_aspects
    para['user_polarity'] = user_polarity
    para['item_polarity'] = item_polarity
    output = open(os.path.join(TPS_DIR, 'data.para'), 'wb')
    # print("data.para:"+str(para))
    # Pickle dictionary using protocol 0.
    pickle.dump(para, output)










