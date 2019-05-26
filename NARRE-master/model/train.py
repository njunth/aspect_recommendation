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
import math
import NARRE as NARRE
import NARRE_aspect as NA_asp
# import DeepCoNN as DeepCoNN
import sys
import time
import os

# tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("dir", "../data/yelp13res_split/origin", "The directory of the dataset")
tf.flags.DEFINE_string("word2vec", "google.w2v.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_string("W1", "user_embedding_table.npy", "user embedding look up table")
tf.flags.DEFINE_string("W2", "item_embedding_table.npy", "item embedding look up table")

tf.flags.DEFINE_string("valid_data","data.test", " Data for validation")
tf.flags.DEFINE_string("para_data", "data.para", "Data parameters")
tf.flags.DEFINE_string("train_data", "data.train", "Data for training")

tf.flags.DEFINE_string("model", "NARRE", "model to train")
# ==================================================

# Model Hyperparameters
# tf.flags.DEFINE_string("word2vec", "./data/rt-polaritydata/google.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding ")
tf.flags.DEFINE_string("filter_sizes", "3", "Comma-separated filter sizes ")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability ")
tf.flags.DEFINE_float("l2_reg_lambda", 0.001, "L2 regularizaion lambda")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size ")
tf.flags.DEFINE_integer("num_epochs", 30, "Number of training epochs ")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


def train_step(u_batch, i_batch, user_aspects_batch, item_aspects_batch, user_polarity_batch,
               item_polarity_batch, uid, iid, reuid, reiid, y_batch, batch_num):
    """
    A single training step
    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_u_aspects: user_aspects_batch,
        deep.input_i_aspects: item_aspects_batch,
        deep.input_u_polarity: user_polarity_batch,
        deep.input_i_polarity: item_polarity_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_y: y_batch,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 0.8,

        deep.dropout_keep_prob: FLAGS.dropout_keep_prob
    }

    _, step, loss, accuracy, mae, u_a, i_a, fm = sess.run(
        [train_op, global_step, deep.loss, deep.accuracy, deep.mae, deep.u_a, deep.i_a, deep.score],
        feed_dict)
    # print(h)
    # embedded_user, embedded_users, conv, h, pooled, h_pool_u, h_pool_flat_u, u_j, u_a, u_f, u_f2 \
    #     = sess.run([deep.embedded_user, deep.embedded_users, deep.conv, deep.h, deep.pooled, deep.h_pool_u, deep.h_pool_flat_u, deep.u_j, deep.u_a, deep.u_f, deep.u_f2], feed_dict)
    # # print(embedded_user)
    # print(embedded_user.shape)
    # print(embedded_users.shape)
    # print("conv:", conv.shape)
    # print(h.shape)
    # print(pooled.shape)
    # print(h_pool_flat_u.shape)
    # print(h_pool_flat_u.shape)
    # print(u_j.shape)
    # # print(u_a)
    # print(u_f.shape)
    # print(u_f2.shape)
    # exit(-1)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step {}, loss {:g}, rmse {:g},mae {:g}".format(time_str, batch_num, loss, accuracy, mae))
    return accuracy, mae, u_a, i_a, fm


def dev_step(u_batch, i_batch, user_aspects_valid, item_aspects_valid, user_polarity_valid, item_polarity_valid,
             uid, iid, reuid, reiid, y_batch, writer=None):
    """
    Evaluates model on a dev set

    """
    feed_dict = {
        deep.input_u: u_batch,
        deep.input_i: i_batch,
        deep.input_u_aspects: user_aspects_valid,
        deep.input_i_aspects: item_aspects_valid,
        deep.input_u_polarity: user_polarity_valid,
        deep.input_i_polarity: item_polarity_valid,
        deep.input_y: y_batch,
        deep.input_uid: uid,
        deep.input_iid: iid,
        deep.input_reuid: reuid,
        deep.input_reiid: reiid,
        deep.drop0: 1.0,
        deep.dropout_keep_prob: 1.0
    }
    step, loss, accuracy, mae = sess.run(
        [global_step, deep.loss, deep.accuracy, deep.mae],
        feed_dict)
    time_str = datetime.datetime.now().isoformat()
    # print("{}: step{}, loss {:g}, rmse {:g},mae {:g}".format(time_str, step, loss, accuracy, mae))

    return [loss, accuracy, mae]


if __name__ == '__main__':
    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    # FLAGS(sys.argv)
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
    user_aspects = para['user_aspects']
    item_aspects = para['item_aspects']
    user_polarity = para['user_polarity']
    item_polarity = para['item_polarity']

    np.random.seed(2017)
    random_seed = 2017
    print(user_num)
    print(item_num)
    print(review_num_u)
    print(review_len_u)
    print(review_num_i)
    print(review_len_i)
    # print(user_aspects)

    # print(type(user_aspects[16237]))
    # print(user_aspects[16237])
    # print(user_polarity[16237])
    # user_asp = [0 for i in range(5)]
    # user_pol = [0 for i in range(5)]
    # for i, ua in enumerate(user_aspects[16237]):
    #     for j in range(5):
    #         user_asp[j] += ua[j]
    #         user_pol[j] += user_polarity[16237][i][j]
    # print user_asp, user_pol
    # print(item_aspects[4274])
    # print(item_polarity[4274])
    # item_asp = [0 for i in range( 5 )]
    # item_pol = [0 for i in range( 5 )]
    # for i, ia in enumerate( item_aspects[4274] ):
    #     for j in range( 5 ):
    #         item_asp[j] += ia[j]
    #         item_pol[j] += item_polarity[4274][i][j]
    # print item_asp, item_pol
    # user_asp_all = 0
    # item_pol_all = 0
    # for j in range(5):
    #     user_asp_all += math.exp(user_asp[j])
    #     item_pol_all += math.exp(item_pol[j])
    # for j in range(5):
    #     user_asp[j] = math.exp(user_asp[j]) / user_asp_all
    #     item_pol[j] = math.exp(item_pol[j]) / item_pol_all
    # print user_asp, item_pol
    # score = [0 for i in range( 5 )]
    # for j in range( 5 ):
    #     score[j] = (user_asp[j]*item_pol[j])
    # print score
    # score_all = 0.
    # for j in range(5):
    #     score_all += math.exp(score[j])
    # for j in range(5):
    #     score[j] = math.exp(score[j]) / score_all
    # print score

    with tf.Graph().as_default():

        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            print(FLAGS.model)
            if FLAGS.model == 'NARRE':
                deep = NARRE.NARRE(
                    aspect_num=5,
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
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    n_latent=32)
            elif FLAGS.model == 'NARRE_aspect':
                deep = NA_asp.NARRE_asp(
                    aspect_num=5,
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

            if FLAGS.word2vec:
                # # initial matrix with random uniform
                # u = 0
                # initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_user), FLAGS.embedding_dim))
                # # load any vectors from the word2vec
                print("Load word2vec u file {}\n".format(FLAGS.word2vec))
                # with open(FLAGS.word2vec, "rb") as f:
                #     header = f.readline()
                #     vocab_size, layer1_size = map(int, header.opsplit())
                #     binary_len = np.dtype('float32').itemsize * layer1_size
                #     for line in xrange(vocab_size):
                #         word = []
                #         while True:
                #             ch = f.read(1)
                #             if ch == ' ':
                #                 word = ''.join(word)
                #                 break
                #             if ch != '\n':
                #                 word.append(ch)
                #         idx = 0
                #
                #         if word in vocabulary_user:
                #             u = u + 1
                #             idx = vocabulary_user[word]
                #             initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                #         else:
                #             f.read(binary_len)
                initW = np.load(os.path.join(FLAGS.dir, FLAGS.W1))
                sess.run(deep.W1.assign(initW))
                del initW

                # initW = np.random.uniform(-1.0, 1.0, (len(vocabulary_item), FLAGS.embedding_dim))
                # load any vectors from the word2vec
                print("Load word2vec i file {}\n".format(FLAGS.word2vec))

                # item = 0
                # with open(FLAGS.word2vec, "rb") as f:
                #     header = f.readline()
                #     vocab_size, layer1_size = map(int, header.split())
                #     binary_len = np.dtype('float32').itemsize * layer1_size
                #     for line in xrange(vocab_size):
                #         word = []
                #         while True:
                #             ch = f.read(1)
                #             if ch == ' ':
                #                 word = ''.join(word)
                #                 break
                #             if ch != '\n':
                #                 word.append(ch)
                #         idx = 0
                #         if word in vocabulary_item:
                #             item = item + 1
                #             idx = vocabulary_item[word]
                #             initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                #         else:
                #             f.read(binary_len)
                initW = np.load(os.path.join(FLAGS.dir, FLAGS.W2))
                sess.run(deep.W2.assign(initW))
                del initW

            epoch = 1
            best_mae = 5
            best_rmse = 5
            train_mae = 0
            train_rmse = 0

            pkl_file = open(os.path.join(FLAGS.dir, FLAGS.train_data), 'rb')
            train_data = pickle.load(pkl_file)
            train_data = np.array(train_data)
            # print len(train_data)
            # print train_data[0]
            pkl_file.close()

            pkl_file = open(os.path.join(FLAGS.dir, FLAGS.valid_data), 'rb')
            test_data = pickle.load(pkl_file)
            test_data = np.array(test_data)
            # print len(test_data)
            # print test_data[0]
            pkl_file.close()

            data_size_train = len(train_data)
            data_size_test = len(test_data)
            batch_size = FLAGS.batch_size
            ll = int(len(train_data) / batch_size)
            for epoch in range(FLAGS.num_epochs):
                # Shuffle the data at each epoch
                # shuffle_indices = np.random.permutation(np.arange(data_size_train))
                # shuffled_data = train_data[shuffle_indices]
                shuffled_data = train_data
                for batch_num in range(ll):
                    print ("epoch: %s, batch_num: %s" % (str(epoch), str(batch_num)))
                    start_index = batch_num * batch_size
                    end_index = min((batch_num + 1) * batch_size, data_size_train)
                    data_train = shuffled_data[start_index:end_index]

                    uid, iid, reuid, reiid, y_batch = zip(*data_train)
                    u_batch = []
                    i_batch = []
                    user_aspects_batch = []
                    item_aspects_batch = []
                    user_polarity_batch = []
                    item_polarity_batch = []
                    for i in range(len(uid)):
                        u_batch.append(u_text[uid[i][0]])
                        i_batch.append(i_text[iid[i][0]])
                        user_aspects_batch.append(user_aspects[uid[i][0]])
                        item_aspects_batch.append(item_aspects[iid[i][0]])
                        user_polarity_batch.append(user_polarity[uid[i][0]])
                        item_polarity_batch.append(item_polarity[iid[i][0]])
                    u_batch = np.array(u_batch)
                    i_batch = np.array(i_batch)
                    user_aspects_batch = np.array(user_aspects_batch)
                    item_aspects_batch = np.array(item_aspects_batch)
                    user_polarity_batch = np.array(user_polarity_batch)
                    item_polarity_batch = np.array(item_polarity_batch)
                    print u_batch.shape
                    print user_aspects_batch.shape
                    print i_batch.shape
                    print item_aspects_batch.shape

                    t_rmse, t_mae, u_a, i_a, fm = train_step(u_batch, i_batch, user_aspects_batch, item_aspects_batch,
                                    user_polarity_batch, item_polarity_batch, uid, iid, reuid, reiid, y_batch,batch_num)
                    current_step = tf.train.global_step(sess, global_step)
                    train_rmse += t_rmse
                    train_mae += t_mae
                    if batch_num % 5 == 0 and batch_num > 1:
                        print("Evaluation:"+str(batch_num))
                        # print(batch_num)

                        loss_s = 0
                        accuracy_s = 0
                        mae_s = 0

                        ll_test = int(len(test_data) / batch_size) + 1
                        for batch_num in range(ll_test):
                            start_index = batch_num * batch_size
                            end_index = min((batch_num + 1) * batch_size, data_size_test)
                            data_test = test_data[start_index:end_index]

                            userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                            u_valid = []
                            i_valid = []
                            user_aspects_valid = []
                            item_aspects_valid = []
                            user_polarity_valid = []
                            item_polarity_valid = []
                            for i in range(len(userid_valid)):
                                u_valid.append(u_text[userid_valid[i][0]])
                                i_valid.append(i_text[itemid_valid[i][0]])
                                user_aspects_valid.append( user_aspects[userid_valid[i][0]] )
                                item_aspects_valid.append( item_aspects[itemid_valid[i][0]] )
                                user_polarity_valid.append( user_polarity[userid_valid[i][0]] )
                                item_polarity_valid.append( item_polarity[itemid_valid[i][0]] )
                            u_valid = np.array(u_valid)
                            i_valid = np.array(i_valid)
                            user_aspects_valid = np.array( user_aspects_valid )
                            item_aspects_valid = np.array( item_aspects_valid )
                            user_polarity_valid = np.array( user_polarity_valid )
                            item_polarity_valid = np.array( item_polarity_valid )

                            loss, accuracy, mae = dev_step(u_valid, i_valid, user_aspects_valid, item_aspects_valid,
                                                           user_polarity_valid, item_polarity_valid,
                                                           userid_valid, itemid_valid, reuid, reiid, y_valid)
                            print loss.shape
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

                print(str(epoch) + ':')
                print("Evaluation:")
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

                    userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_test)
                    u_valid = []
                    i_valid = []
                    user_aspects_valid = []
                    item_aspects_valid = []
                    user_polarity_valid = []
                    item_polarity_valid = []
                    for i in range( len( userid_valid ) ):
                        u_valid.append( u_text[userid_valid[i][0]] )
                        i_valid.append( i_text[itemid_valid[i][0]] )
                        user_aspects_valid.append( user_aspects[userid_valid[i][0]] )
                        item_aspects_valid.append( item_aspects[itemid_valid[i][0]] )
                        user_polarity_valid.append( user_polarity[userid_valid[i][0]] )
                        item_polarity_valid.append( item_polarity[itemid_valid[i][0]] )
                    u_valid = np.array( u_valid )
                    i_valid = np.array( i_valid )
                    user_aspects_valid = np.array( user_aspects_valid )
                    item_aspects_valid = np.array( item_aspects_valid )
                    user_polarity_valid = np.array( user_polarity_valid )
                    item_polarity_valid = np.array( item_polarity_valid )

                    loss, accuracy, mae = dev_step(u_valid, i_valid, user_aspects_valid, item_aspects_valid,
                                                           user_polarity_valid, item_polarity_valid,
                                                   userid_valid, itemid_valid, reuid, reiid, y_valid)
                    loss_s = loss_s + len(y_valid) * loss
                    accuracy_s = accuracy_s + len(y_valid) * np.square(accuracy)
                    mae_s = mae_s + len(y_valid) * mae
                print("loss_valid {:g}, rmse_valid {:g}, mae_valid {:g}".format(loss_s / test_length,
                                                                                 np.sqrt(accuracy_s / test_length),
                                                                                 mae_s / test_length))
                time_stamp = time.asctime().replace(':', '_').split()
                print(time_stamp)
                rmse = np.sqrt(accuracy_s / test_length)
                mae = mae_s / test_length
                if best_rmse > rmse:
                    best_rmse = rmse
                if best_mae > mae:
                    best_mae = mae
                # print("")
            print('best rmse:', best_rmse)
            print('best mae:', best_mae)
