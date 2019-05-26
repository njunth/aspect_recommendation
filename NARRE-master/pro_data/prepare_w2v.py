import pickle
from gensim.models import KeyedVectors
import numpy as np

import struct

TPS = "../data/yelp13res"

def load_vocabulary(pkl_file):
    para = pickle.load(pkl_file)
    # user_num = para['user_num']
    # item_num = para['item_num']
    # review_num_u = para['review_num_u']
    # review_num_i = para['review_num_i']
    # review_len_u = para['review_len_u']
    # review_len_i = para['review_len_i']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    return vocabulary_user, vocabulary_item


def build_table(w2v_model, vocab):
    initW = np.random.uniform(-1.0, 1.0, (len(vocab), 300))
    u = 0
    for word in vocab:
        u = u + 1
        idx = vocab[word]
        if word in w2v_model:
            initW[idx] = w2v_model[word]
        else:
            print(word)
    print(u)
    return initW


def new_w2v():
    pkl_file = open("%s/data.para" % TPS, 'rb')
    vocab_u, vocab_i = load_vocabulary(pkl_file)
    # print(vocab_u)
    print(len(vocab_u))
    print(len(vocab_i))
    print(vocab_u['love'])
    all_words = set()
    all_words = all_words.union(set(vocab_u.keys()))
    print(len(all_words))
    all_words = all_words.union(set(vocab_i.keys()))
    print(len(all_words))
    length = len(all_words)
    w2v_model = KeyedVectors.load_word2vec_format('E:/embedding/GoogleNews-vectors-negative300.bin', binary=True)
    word_list = list(all_words)
    embeds_list = []
    miss = set()
    for w in word_list:
        if w in w2v_model:
            # in_set.add(w)
            embeds = w2v_model[w]
        else:
            miss.add(w)
            embeds = np.random.uniform(-0.25, 0.25, 300)
        embeds_list.append(embeds)
    print("miss:", len(miss)/len(all_words))
    new_w2v = KeyedVectors(300)
    new_w2v.add(word_list, embeds_list)
    new_w2v.save_word2vec_format("%s/google.w2v.bin" % TPS, binary=True)


if __name__ == '__main__':
    # new_w2v()

    w2v_model = KeyedVectors.load_word2vec_format("%s/google.w2v.bin" % TPS, binary=True)
    print(w2v_model['love'])

    pkl_file = open("%s/data.para" % TPS, 'rb')
    para = pickle.load(pkl_file)
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    print(len(vocabulary_user))
    print(len(vocabulary_item))
    # print(vocabulary_user)
    # initW1 = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
    W1 = build_table(w2v_model, vocabulary_user)
    W2 = build_table(w2v_model, vocabulary_item)
    np.save("%s/user_embedding_table" % TPS, W1)
    np.save("%s/item_embedding_table" % TPS, W2)



