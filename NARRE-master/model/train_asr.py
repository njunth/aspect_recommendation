'''
NARRE
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:

'''

import numpy as np
import tensorflow as tf
import pickle
import datetime
import NARRE as NARRE
# import DeepCoNN as DeepCoNN
import sys
import time
import os

# tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")

tf.flags.DEFINE_string("dir", "../data/yelp13res_filtered", "The directory of the dataset")
tf.flags.DEFINE_string("word2vec", "google.w2v.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("W1", "user_embedding_table.npy", "user embedding look up table")
tf.flags.DEFINE_string("W2", "item_embedding_table.npy", "item embedding look up table")

tf.flags.DEFINE_string("valid_data","data.test", " Data for validation")
tf.flags.DEFINE_string("para_data", "data.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "data.train", "Data for training")

tf.flags.DEFINE_string("model", "ASR", "model to train")
# ==================================================

# Model Hyperparameters
# tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs ")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.drop0: 0.8,
        deep.input_u_feature: u_batch,
        deep.input_i_feature: i_batch,
        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    u_a, i_a = 0, 0
    # result = sess.run([deep.batch_size, deep.ufmf], feed_dict)
    # print(result)
    # print(result[-1])
    # print(result[-1].shape)

    # exit(-1)
    _, step, loss, accuracy, mae, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.score],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae, u_a, i_a, fm


def dev_step(u_batch, i_batch, uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_u_feature: u_batch,
        deep.input_i_feature: i_batch,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    # print(step, loss, accuracy, mae)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

def normalize(x):
    sum = x.sum(axis=0)
    if sum != 0:
        x = x/sum
    else:
        x = np.ones(x.size)/x.size
    return x


def normalize2(x):
    score_tensor = np.zeros(x.shape[0])
    for i, count in enumerate(x):
        probas = softmax(count)
        score_tensor[i] = np.sum(probas*np.asarray([1, 3, 5]))
    return score_tensor

def normalize3(x):
    score_tensor = np.zeros(x.shape[0])
    for i, count in enumerate(x):
        probas = normalize(count)
        score_tensor[i] = np.sum(probas*np.asarray([1, 3, 5]))
    return score_tensor


def build_user_preference(reviews_aspect_feature, aspect_size=5):
    count_tensor = np.zeros(aspect_size)
    if reviews_aspect_feature is None:
        return np.ones(aspect_size)/aspect_size
    else:
        for review in reviews_aspect_feature:
            iid = review['iid']
            rid = review['rid']
            sentence_features = review['features']
            if sentence_features is None:
                continue
            for s_feature in sentence_features:
                text_length = s_feature['text_length']
                predicted_a = s_feature['predicted_a']
                predicted_p = s_feature['predicted_p']
                for aspect_i in predicted_a['aspects']:
                    count_tensor[aspect_i] += 1
    # print(count_tensor)
    aspect_importance = softmax(count_tensor)
    # print(aspect_importance)
    return aspect_importance


def build_user_preference2(reviews_aspect_feature, aspect_size=5):
    count_tensor = np.zeros(aspect_size)
    count_proba_tensor = np.zeros(aspect_size)
    count = 0
    if reviews_aspect_feature is None:
        return np.ones(aspect_size)/aspect_size
    else:
        for review in reviews_aspect_feature:
            iid = review['iid']
            rid = review['rid']
            sentence_features = review['features']
            if sentence_features is None:
                continue
            for s_feature in sentence_features:
                text_length = s_feature['text_length']
                if text_length < 2:
                    continue
                predicted_a = s_feature['predicted_a']
                predicted_p = s_feature['predicted_p']
                # print(predicted_p)
                count_proba_tensor += np.asarray(predicted_a['vector_a'])
                for aspect_i in predicted_a['aspects']:
                    count_tensor[aspect_i] += 1
                count += 1
    # print(count_tensor)
    # aspect_importance = softmax(count_tensor)
    # print(aspect_importance)
    # count_proba_tensor = count_proba_tensor/count
    aspect_importance = normalize(count_proba_tensor)
    # aspect_importance = count_proba_tensor/count
    # print(count_proba_tensor)
    # print(aspect_importance)
    return aspect_importance


# def build_item_characteristics(reviews_polarity_feature, aspect_size=5):
#     count_tensor = np.zeros(aspect_size*3)
#     if reviews_polarity_feature is None:
#         return np.ones(aspect_size*3)/(aspect_size*3)
#     else:
#         for review in reviews_polarity_feature:
#             uid = review['uid']
#             rid = review['rid']
#             sentence_features = review['features']
#             if sentence_features is None:
#                 continue
#             for s_feature in sentence_features:
#                 text_length = s_feature['text_length']
#                 predicted_a = s_feature['predicted_a']
#                 predicted_p = s_feature['predicted_p']
#                 for apv in predicted_p:
#                     aspect = apv['aspect']
#                     polarity = apv['polarity']
#                     polarity_vector = apv['vector_p']
#                     count_tensor[aspect*3 + polarity] += 1
#     # print(count_tensor)
#     polarity_importance = normalize(count_tensor)
#     # print(polarity_importance)
#     return polarity_importance

def build_item_characteristics(reviews_polarity_feature, aspect_size=5):
    count_tensor = np.zeros((aspect_size, 3))
    if reviews_polarity_feature is None:
        return np.zeros(aspect_size)
    else:
        for review in reviews_polarity_feature:
            uid = review['uid']
            rid = review['rid']
            sentence_features = review['features']
            if sentence_features is None:
                continue
            for s_feature in sentence_features:
                text_length = s_feature['text_length']
                predicted_a = s_feature['predicted_a']
                predicted_p = s_feature['predicted_p']
                for apv in predicted_p:
                    aspect = apv['aspect']
                    polarity = apv['polarity']
                    polarity_vector = apv['vector_p']
                    count_tensor[aspect][polarity] += 1
    # print(count_tensor)
    polarity_importance = normalize3(count_tensor)
    # print(polarity_importance)
    return polarity_importance


def build_item_characteristics2(reviews_polarity_feature, aspect_size=5):
    count_tensor = np.zeros((aspect_size, 3))
    if reviews_polarity_feature is None:
        return np.zeros(aspect_size)
    else:
        for review in reviews_polarity_feature:
            uid = review['uid']
            rid = review['rid']
            sentence_features = review['features']
            if sentence_features is None:
                continue
            for s_feature in sentence_features:
                text_length = s_feature['text_length']
                if text_length < 2:
                    continue
                predicted_a = s_feature['predicted_a']
                predicted_p = s_feature['predicted_p']
                for apv in predicted_p:
                    aspect = apv['aspect']
                    polarity = apv['polarity']
                    polarity_vector = apv['vector_p']
                    count_tensor[aspect] += polarity_vector
    # print(count_tensor)
    polarity_importance = normalize3(count_tensor)
    # print(polarity_importance)
    return polarity_importance

# def build_item_characteristics(reviews_polarity_feature, aspect_size=5):
#     count_tensor = np.zeros(aspect_size)
#     if reviews_polarity_feature is None:
#         return np.zeros(aspect_size)
#     else:
#         count_sentence = 0
#         for review in reviews_polarity_feature:
#             uid = review['uid']
#             rid = review['rid']
#             sentence_features = review['features']
#             if sentence_features is None:
#                 continue
#             for s_feature in sentence_features:
#                 count_sentence += 1
#                 text_length = s_feature['text_length']
#                 predicted_a = s_feature['predicted_a']
#                 predicted_p = s_feature['predicted_p']
#                 for apv in predicted_p:
#                     aspect = apv['aspect']
#                     polarity = apv['polarity']
#                     polarity_vector = apv['vector_p']
#                     count_tensor[aspect] += polarity*2+1
#     # print(count_tensor)
#     polarity_importance = count_tensor/count_sentence
#     # print(polarity_importance)
#     return polarity_importance


def build_user_feature(reviews_polarity_feature, skip_iid=None, aspect_size=5):
    pos_fields = {'0': [], '1': [], '2': [], '3': [], '4': []}
    neg_fields = {'0': [], '1': [], '2': [], '3': [], '4': []}
    count_tensor = np.zeros((aspect_size, 3))
    if reviews_polarity_feature is None:
        return np.zeros(aspect_size)
    else:
        for review in reviews_polarity_feature:
            iid = review['iid']
            rid = review['rid']
            rating = int(review['rating'])
            sentence_features = review['features']
            if sentence_features is None:
                continue
            if iid == skip_iid:
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
        # print(key + ':' + str(np.mean(pos_fields[key])) + ':' + str(pos_fields[key]))
        mean_rating = np.mean(pos_fields[key]) if pos_fields[key] != [] else 3
        pos_mean[int(key)] = mean_rating
    for key in neg_fields:
        # print(key + ':' + str(np.mean(neg_fields[key])) + ':' + str(neg_fields[key]))
        mean_rating = np.mean(neg_fields[key]) if neg_fields[key] != [] else 3
        neg_mean[int(key)] = mean_rating
    aspect_sensitive = np.exp(pos_mean - neg_mean)
    # print(aspect_sensitive)
    # return np.zeros(aspect_size)
    return aspect_sensitive


def build_item_feature(reviews_polarity_feature, skip_uid=None, aspect_size=5):
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
            if uid == skip_uid:
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
                        score = np.exp(rating - 3)*5
                        neg_fields[aspect].append(score)
                    elif polarity == 0:
                        score = np.exp(3 - rating)*1
                        neg_fields[aspect].append(score)
    # for key in pos_fields:
    #     print(key + ':' + str(sum(pos_fields[key])/len(pos_fields[key])) + ':' + str(pos_fields[key]))
    aspect_characteristics = np.zeros(aspect_size)
    for key in neg_fields:
        # print(key + ':' + str(sum(neg_fields[key])/len(neg_fields[key])) + ':' + str(neg_fields[key]))
        mean_rating = np.mean(neg_fields[key]) if neg_fields[key] != [] else 0
        aspect_characteristics[int(key)] = mean_rating
    # print(aspect_characteristics)
    return aspect_characteristics
    # return np.zeros(aspect_size)


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    # FLAGS._parse_flags()
    FLAGS(sys.argv)
    print(FLAGS.dir)
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    print("Loading data...")
    pkl_file = open(os.path.join(FLAGS.dir, FLAGS.para_data), 'rb')

    para = pickle.load(pkl_file)
    user_num = para['user_num']
    item_num = para['item_num']
    review_num_u = para['review_num_u']
    review_num_i = para['review_num_i']
    review_len_u = para['review_len_u']
    review_len_i = para['review_len_i']
    vocabulary_user = para['user_vocab']
    vocabulary_item = para['item_vocab']
    train_length = para['train_length']
    test_length = para['test_length']
    u_text = para['u_text']
    i_text = para['i_text']

    user_features = pickle.load(open(os.path.join(FLAGS.dir, "user_features"), 'rb'))
    item_features = pickle.load(open(os.path.join(FLAGS.dir, "item_features"), 'rb'))

    np.random.seed(2017)
    random_seed = 2017
    print(user_num)
    print(item_num)
    print(review_num_u)
    print(review_len_u)
    print(review_num_i)
    print(review_len_i)
    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = False
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            if FLAGS.model == 'ASR':
                deep = NARRE.APNCF(
                    review_num_u=review_num_u,
                    review_num_i=review_num_i,
                    review_len_u=review_len_u,
                    review_len_i=review_len_i,
                    user_num=user_num,
                    item_num=item_num,
                    num_classes=1,
                    user_vocab_size=len(vocabulary_user),
                    item_vocab_size=len(vocabulary_item),
                    embedding_size=FLAGS.embedding_dim,
                    embedding_id=32,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    attention_size=32,
                    n_latent=32)
            elif FLAGS.model == 'DeepCoNN':
                deep = NARRE.DeepCoNN(
                    user_num=user_num,
                    item_num=item_num,
                    user_length=review_num_u,
                    item_length=review_num_i,
                    num_classes=1,
                    user_vocab_size=len(vocabulary_user),
                    item_vocab_size=len(vocabulary_item),
                    embedding_size=FLAGS.embedding_dim,
                    fm_k=8,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    l2_reg_V=FLAGS.l2_reg_V,
                    n_latent=32)
            else:
                print("No model")
                exit(-1)
            tf.set_random_seed(random_seed)
            print(user_num)
            print(item_num)
            global_step = tf.Variable(0, name="global_step", trainable=False)

            # optimizer = tf.train.AdagradOptimizer(learning_rate=0.01, initial_accumulator_value=1e-8).minimize(deep.loss)
            optimizer = tf.train.AdamOptimizer(0.002, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(deep.loss)

            train_op = optimizer  # .apply_gradients(grads_and_vars, global_step=global_step)

            sess.run(tf.initialize_all_variables())

            saver = tf.train.Saver()

            epoch = 1
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0

            pkl_file = open(os.path.join(FLAGS.dir, FLAGS.train_data), 'rb')

            train_data = pickle.load(pkl_file)

            train_data = np.array(train_data)
            pkl_file.close()

            pkl_file = open(os.path.join(FLAGS.dir, FLAGS.valid_data), 'rb')

            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            pkl_file.close()

            user_features_dict = {}
            item_features_dict = {}
            train_uid, train_iid, _, _, _ = zip(*train_data)
            test_uid, test_iid, _, _, _ = zip(*test_data)
            train_uid = [id[0] for id in train_uid]
            train_iid = [id[0] for id in train_iid]
            test_uid = [id[0] for id in test_uid]
            test_iid = [id[0] for id in test_iid]
            all_uid = set(train_uid + test_uid)
            all_iid = set(train_iid + test_iid)
            for uid in all_uid:
                # print(uid)
                # uid = uid[0]
                user_features_dict[uid] = build_user_feature(user_features[uid])
            for iid in all_iid:
                # print(iid)
                # iid = iid[0]
                item_features_dict[iid] = build_item_feature(item_features[iid])

            # feature_dict = {}
            # for uid, iid in zip(train_uid, train_iid):
            #     feature_dict[(uid, iid)] = (build_user_feature(user_features[uid], iid), build_item_feature(item_features[iid], uid))
            # for uid, iid in zip(test_uid, test_iid):
            #     if (uid, iid) in feature_dict:
            #         print("error")
            #         exit(-1)
            #     feature_dict[(uid, iid)] = (build_user_feature(user_features[uid], iid), build_item_feature(item_features[iid], uid))

            print("finish computing")
            data_size_train = len(train_data)
            data_size_test = len(test_data)
            batch_size = FLAGS.batch_size
            ll = int(len(train_data) / batch_size)
            for epoch in range(FLAGS.num_epochs):
                # Shuffle the data at each epoch
                shuffle_indices = np.random.permutation(np.arange(data_size_train))
                shuffled_data = train_data[shuffle_indices]
                for batch_num in range(ll):

                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]
                    print(data_train)
                    uid, iid, reuid, reiid, y_batch = zip(*data_train)
                    print(uid)
                    print(iid)
                    exit(-1)
                    u_f_batch = []
                    i_f_batch = []
                    for i in range(len(uid)):
                        # u_f, i_f = feature_dict[(uid[i][0], iid[i][0])]
                        # u_f_batch.append(u_f)
                        # i_f_batch.append(i_f)
                        u_f_batch.append(user_features_dict[uid[i][0]])
                        i_f_batch.append(item_features_dict[iid[i][0]])
                    u_f_batch = np.array(u_f_batch)
                    i_f_batch = np.array(i_f_batch)

                    # print(u_f_batch, i_f_batch, y_batch)

                    t_rmse, t_mae, u_a, i_a, fm = train_step(u_f_batch, i_f_batch, uid, iid, reuid, reiid, y_batch,
                                                             batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae
                    if batch_num % 500 == 0 and batch_num > 1:
                        print("\nEvaluation:")
                        print(batch_num)

                        loss_s = 0
                        accuracy_s = 0
                        mae_s = 0

                        ll_test = int(len(test_data) / batch_size) + 1
                        for batch_num in range(ll_test):
                            start_index = batch_num * batch_size
                            end_index = min((batch_num + 1) * batch_size, data_size_test)
                            data_test = test_data[start_index:end_index]
                            # print(data_test[0])
                            # exit(-1)

                            userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                            uf_valid = []
                            if_valid = []
                            # for i in range(len(userid_valid)):
                            #     u_valid.append(u_text[userid_valid[i][0]])
                            #     i_valid.append(i_text[itemid_valid[i][0]])
                            # u_valid = np.array(u_valid)
                            # i_valid = np.array(i_valid)
                            for i in range(len(userid_valid)):
                                # u_f, i_f = feature_dict[(userid_valid[i][0], itemid_valid[i][0])]
                                # uf_valid.append(u_f)
                                # if_valid.append(i_f)
                                uf_valid.append(user_features_dict[userid_valid[i][0]])
                                if_valid.append(item_features_dict[itemid_valid[i][0]])
                            uf_valid = np.array(uf_valid)
                            if_valid = np.array(if_valid)

                            loss, accuracy, mae = dev_step(uf_valid, if_valid, userid_valid, itemid_valid, reuid, reiid,
                                                           y_valid)
                            loss_s = loss_s + len(y_valid) * loss
                            accuracy_s = accuracy_s + len(y_valid) * np.square(accuracy)
                            mae_s = mae_s + len(y_valid) * mae
                        print("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                        np.sqrt(
                                                                                            accuracy_s / test_length),
                                                                                        mae_s / test_length))
                        rmse = np.sqrt(accuracy_s / test_length)
                        mae = mae_s / test_length
                        if best_rmse > rmse:
                            best_rmse = rmse
                        if best_mae > mae:
                            best_mae = mae
                        print("")

                print(str(epoch) + ':\n')
                print("\nEvaluation:")
                print("train:rmse,mae:", train_rmse / ll, train_mae / ll)
                # u_a = np.reshape(u_a[0], (1, -1))
                # i_a = np.reshape(i_a[0], (1, -1))
                #
                # print(u_a)
                # print(i_a)
                train_rmse = 0
                train_mae = 0

                loss_s = 0
                accuracy_s = 0
                mae_s = 0

                ll_test = int(len(test_data) / batch_size) + 1
                for batch_num in range(ll_test):
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_test)
                    data_test = test_data[start_index:end_index]
                    # print(data_test[0])
                    # exit(-1)

                    userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                    uf_valid = []
                    if_valid = []
                    # for i in range(len(userid_valid)):
                    #     u_valid.append(u_text[userid_valid[i][0]])
                    #     i_valid.append(i_text[itemid_valid[i][0]])
                    # u_valid = np.array(u_valid)
                    # i_valid = np.array(i_valid)
                    for i in range(len(userid_valid)):
                        # u_f, i_f = feature_dict[(userid_valid[i][0], itemid_valid[i][0])]
                        # uf_valid.append(u_f)
                        # if_valid.append(i_f)
                        uf_valid.append(user_features_dict[userid_valid[i][0]])
                        if_valid.append(item_features_dict[itemid_valid[i][0]])
                    uf_valid = np.array(uf_valid)
                    if_valid = np.array(if_valid)

                    loss, accuracy, mae = dev_step(uf_valid, if_valid, userid_valid, itemid_valid, reuid, reiid,
                                                   y_valid)
                    loss_s = loss_s + len(y_valid) * loss
                    accuracy_s = accuracy_s + len(y_valid) * np.square(accuracy)
                    mae_s = mae_s + len(y_valid) * mae
                print("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                np.sqrt(
                                                                                    accuracy_s / test_length),
                                                                                mae_s / test_length))
                rmse = np.sqrt(accuracy_s / test_length)
                mae = mae_s / test_length
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                print("")
            print('best rmse:', best_rmse, ' best mae:', best_mae)
