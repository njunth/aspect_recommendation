'''
NARRE
@author:
Chong Chen (cstchenc@163.com)

@ created:
27/8/2017
@references:
Chong Chen, Min Zhang, Yiqun Liu, and Shaoping Ma. 2018. Neural Attentional Rating Regression with Review-level Explanations. In WWW'18.
'''


import tensorflow as tf
import numpy as np


class NARRE(object):
    def __init__(
            self, aspect_num, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        self.input_u_aspects = tf.placeholder(tf.float32, [None, review_num_u, aspect_num], name="input_u_aspects")
        self.input_i_aspects = tf.placeholder(tf.float32, [None, review_num_i, aspect_num], name="input_i_aspects")
        self.input_u_polarity = tf.placeholder(tf.float32, [None, review_num_u, aspect_num], name="input_u_polarity")
        self.input_i_polarity = tf.placeholder(tf.float32, [None, review_num_i, aspect_num], name="input_i_polarity")
        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)
        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W1")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W2")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        pooled_outputs_u = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_users = tf.reshape(self.embedded_users, [-1, review_len_u, embedding_size, 1])

                self.conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                self.h = tf.nn.relu(tf.nn.bias_add(self.conv, b), name="relu")

                # Maxpooling over the outputs
                self.pooled = tf.nn.max_pool(
                    self.h,
                    ksize=[1, review_len_u - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(self.pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, review_num_u, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_items = tf.reshape(self.embedded_items, [-1, review_len_i, embedding_size, 1])
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, review_len_i - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, review_num_i, num_filters_total])
        
        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("attention"):
            Wau = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wau')
            Wru = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wru')
            Wpu = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpu')
            bau = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bau")
            bbu = tf.Variable(tf.constant(0.1, shape=[1]), name="bbu")
            self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(iidW, self.input_reuid))
            self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_u, Wau) + tf.einsum('ajk,kl->ajl', self.iid_a, Wru) + bau),
                                             Wpu)+bbu  # None*u_len*1

            self.u_a = tf.nn.softmax(self.u_j, 1)  # none*u_len*1

            print(self.u_a)

            Wai = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.1, 0.1), name='Wai')
            Wri = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.1, 0.1), name='Wri')
            Wpi = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.1, 0.1), name='Wpi')
            bai = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
            self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(uidW, self.input_reiid))
            self.i_j =tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_i, Wai) + tf.einsum('ajk,kl->ajl', self.uid_a, Wri) + bai),
                                             Wpi)+bbi

            self.i_a = tf.nn.softmax(self.i_j,1)  # none*len*1

            l2_loss += tf.nn.l2_loss(Wau)
            l2_loss += tf.nn.l2_loss(Wru)
            l2_loss += tf.nn.l2_loss(Wri)
            l2_loss += tf.nn.l2_loss(Wai)

        with tf.name_scope("add_reviews"):
            self.u_f = tf.multiply(self.u_a, self.h_drop_u)
            self.u_f2 = tf.reduce_sum(self.u_f, 1)
            self.u_feas = tf.reduce_sum(tf.multiply(self.u_a, self.h_drop_u), 1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_sum(tf.multiply(self.i_a, self.h_drop_i), 1)
            self.i_feas = tf.nn.dropout(self.i_feas, self.dropout_keep_prob)

        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf,self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf,self.input_iid)
            self.uid = tf.reshape(self.uid,[-1,embedding_id])
            self.iid = tf.reshape(self.iid,[-1,embedding_id])
            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_feas = tf.matmul(self.u_feas, Wu)+self.uid + bu

            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi

        with tf.name_scope("ui_aspect_polarity"):
            user_asp_sum = tf.reduce_sum(self.input_u_aspects, axis=1)
            user_pol_sum = tf.reduce_sum(self.input_u_polarity, axis=1)
            user_preference = tf.concat([user_asp_sum, user_pol_sum], axis=1)

            item_asp_sum = tf.reduce_sum( self.input_i_aspects, axis=1)
            item_pol_sum = tf.reduce_sum( self.input_i_polarity, axis=1)
            item_preference = tf.concat([item_asp_sum, item_pol_sum], axis=1)

            # self.u_i_aspect_polarity = tf.multiply(user_preference, item_preference)
            # self.u_i_aspect_polarity = tf.nn.relu(self.u_i_aspect_polarity)
            # self.u_i_aspect_polarity = tf.nn.dropout(self.u_i_aspect_polarity, self.dropout_keep_prob)
            # self.u_i_aspect_polarity = tf.layers.dropout(self.u_i_aspect_polarity, rate=0.5)
            # ui_w = tf.Variable(
            #     tf.random_uniform( [2*aspect_num, 1], -0.1, 0.1 ), name='ui_w' )

            # self.u_i_aspect_polarity = tf.multiply( user_asp_sum, item_pol_sum )
            # self.u_i_aspect_polarity = tf.nn.relu( self.u_i_aspect_polarity )
            # ui_w = tf.Variable(
            #     tf.random_uniform( [aspect_num, 1], -0.1, 0.1 ), name='ui_w' )
            #
            #
            # self.u_i_mul = tf.matmul( self.u_i_aspect_polarity, ui_w )
            # self.ui_asp_score = tf.reduce_sum( self.u_i_mul, 1, keep_dims=True )

            # self.u_i_aspect_polarity = tf.concat([user_preference, item_preference], axis=1)
            # self.ui_asp_score = tf.contrib.layers.fullyconnected(inputs=self.u_i_aspect_polarity, num_outputs=5)
            # self.ui_asp_score = tf.contrib.layers.fullyconnected(inputs=self.ui_asp_score, num_outputs=1)

        with tf.name_scope('ncf'):
            # self.u_feas = tf.concat([self.u_feas, user_preference], axis=1)
            # self.i_feas = tf.concat([self.i_feas, item_preference], axis=1)
            # self.u_feas = tf.concat( [self.u_feas, user_asp_sum], axis=1 )
            # self.i_feas = tf.concat( [self.i_feas, item_pol_sum], axis=1 )
            #
            # self.FM = tf.multiply(self.u_feas, self.i_feas)
            # self.FM = tf.nn.relu(self.FM)
            #
            # self.FM=tf.nn.dropout(self.FM,self.dropout_keep_prob)
            #
            # Wmul=tf.Variable(
            #     tf.random_uniform([5+n_latent, 1], -0.1, 0.1), name='wmul')
            #
            # self.mul=tf.matmul(self.FM, Wmul)

            self.u_feas = tf.concat( [self.u_feas, user_asp_sum], axis=1 )
            self.i_feas = tf.concat( [self.i_feas, item_pol_sum], axis=1 )
            self.u_i_feas = tf.concat( [self.i_feas, self.u_feas], axis=1 )

            self.ui_score = tf.contrib.layers.fully_connected( inputs=self.u_i_feas, num_outputs=32 )
            self.mul = tf.contrib.layers.fully_connected( inputs=self.ui_score, num_outputs=1 )

            self.score=tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            # self.all_feat = tf.concat( [self.score, self.ui_asp_score], axis=1 )
            # self.predictions = tf.contrib.layers.fully_connected( inputs=self.all_feat, num_outputs=1,
            #                                                       activation_fn=tf.nn.relu )

            self.predictions = self.score + self.Feature_bias + self.bised
            # self.predictions = self.score
            # self.predictions = self.ui_asp_score


        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy =tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


class NCF(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, aspect_size=5):
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        # self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        # self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        # iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        # uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            # Wu = tf.Variable(
            #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu
            self.u_feas = self.uid

            # Wi = tf.Variable(
            #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi
            self.i_feas = self.iid

        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.concat([self.u_feas, self.i_feas, self.FM], axis=-1)
            self.FM = tf.keras.layers.Dense(n_latent, activation='relu')(self.FM)
            # self.FM = tf.nn.relu(self.FM)

            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            Wmul = tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

            self.mul = tf.matmul(self.FM, Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


class APNCF(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, aspect_size=5):
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        # self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        # self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        # self.input_u_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_u_feature')
        # self.input_i_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_i_feature')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        # iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        # uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            # # Wu = tf.Variable(
            # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            # bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # # self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu
            # self.u_feas = self.uid + bu
            #
            # # Wi = tf.Variable(
            # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            # bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # # self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi
            # self.i_feas = self.iid + bi

            amf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="amf")
            pmf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="pmf")

            # self.u_feas1 = tf.matmul(self.input_u_feature, amf)
            # self.i_feas1 = tf.matmul(self.input_i_feature, pmf)
            # self.u_feas1 = self.input_u_feature
            # self.i_feas1 = self.input_i_feature

            # self.u_feas = tf.reshape(self.u_feas, [-1, embedding_id])
            # self.i_feas = tf.reshape(self.i_feas, [-1, embedding_id])

            Wu = tf.Variable(
                tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # self.u_feas = tf.matmul(self.u_feas1, Wu) + self.uid + bu
            self.u_feas = tf.concat([self.uid], axis=-1)
            self.u_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.u_feas)
            # self.u_feas = self.u_feas1 + self.uid + bu

            Wi = tf.Variable(
                tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # self.i_feas = tf.matmul(self.i_feas1, Wi) + self.iid + bi
            # self.i_feas = self.i_feas1 + self.iid + bi
            self.i_feas = tf.concat([self.iid], axis=-1)
            self.i_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.i_feas)

        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.keras.layers.Dense(n_latent, activation='relu')(self.FM)
            # self.FM = tf.nn.relu(self.FM)

            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            # Wmul = tf.Variable(
            #     tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')
            #
            # self.mul = tf.matmul(self.FM, Wmul)
            # self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.score = tf.keras.layers.Dense(1, activation='linear')(self.FM)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

class BIAS(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, aspect_size=5):
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        # self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        # self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        # self.input_u_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_u_feature')
        # self.input_i_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_i_feature')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        # iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        # uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)

        # with tf.name_scope("get_fea"):

        #     iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
        #     uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

        #     self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
        #     self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
        #     self.uid = tf.reshape(self.uid, [-1, embedding_id])
        #     self.iid = tf.reshape(self.iid, [-1, embedding_id])
        #     # # Wu = tf.Variable(
        #     # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
        #     # bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
        #     # # self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu
        #     # self.u_feas = self.uid + bu
        #     #
        #     # # Wi = tf.Variable(
        #     # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
        #     # bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
        #     # # self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi
        #     # self.i_feas = self.iid + bi

        #     amf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="amf")
        #     pmf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="pmf")

        #     # self.u_feas1 = tf.matmul(self.input_u_feature, amf)
        #     # self.i_feas1 = tf.matmul(self.input_i_feature, pmf)
        #     # self.u_feas1 = self.input_u_feature
        #     # self.i_feas1 = self.input_i_feature

        #     # self.u_feas = tf.reshape(self.u_feas, [-1, embedding_id])
        #     # self.i_feas = tf.reshape(self.i_feas, [-1, embedding_id])

        #     Wu = tf.Variable(
        #         tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wu')
        #     bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
        #     # self.u_feas = tf.matmul(self.u_feas1, Wu) + self.uid + bu
        #     self.u_feas = tf.concat([self.uid], axis=-1)
        #     self.u_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.u_feas)
        #     # self.u_feas = self.u_feas1 + self.uid + bu

        #     Wi = tf.Variable(
        #         tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wi')
        #     bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
        #     # self.i_feas = tf.matmul(self.i_feas1, Wi) + self.iid + bi
        #     # self.i_feas = self.i_feas1 + self.iid + bi
        #     self.i_feas = tf.concat([self.iid], axis=-1)
        #     self.i_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.i_feas)

        with tf.name_scope('ncf'):

            # self.FM = tf.multiply(self.u_feas, self.i_feas)
            # self.FM = tf.keras.layers.Dense(n_latent, activation='relu')(self.FM)
            # # self.FM = tf.nn.relu(self.FM)

            # self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            # # Wmul = tf.Variable(
            # #     tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')
            # #
            # # self.mul = tf.matmul(self.FM, Wmul)
            # # self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            # self.score = tf.keras.layers.Dense(1, activation='linear')(self.FM)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))

class ASR(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, aspect_size=5):
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        # self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        # self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_u_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_u_feature')
        self.input_i_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_i_feature')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        # iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        # uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            # # Wu = tf.Variable(
            # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            # bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # # self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu
            # self.u_feas = self.uid + bu
            #
            # # Wi = tf.Variable(
            # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            # bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # # self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi
            # self.i_feas = self.iid + bi

            amf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="amf")
            pmf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="pmf")

            self.u_a = tf.multiply(tf.expand_dims(amf, 0), tf.expand_dims(self.uid, 1))
            self.i_a = tf.multiply(tf.expand_dims(pmf, 0), tf.expand_dims(self.iid, 1))

            # self.u_feas1 = tf.matmul(self.input_u_feature, amf)
            # self.i_feas1 = tf.matmul(self.input_i_feature, pmf)
            self.u_feas1 = self.input_u_feature
            self.i_feas1 = self.input_i_feature

            # self.u_feas = tf.reshape(self.u_feas, [-1, embedding_id])
            # self.i_feas = tf.reshape(self.i_feas, [-1, embedding_id])

            # Wu = tf.Variable(
            #     tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wu')
            # bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # # self.u_feas = tf.matmul(self.u_feas1, Wu) + self.uid + bu
            # self.u_feas = tf.concat([self.u_feas1], axis=-1)

            self.u_feas = tf.multiply(tf.expand_dims(self.u_feas1, -1), self.u_a)
            # self.u_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.u_feas)
            # self.u_feas = self.u_feas1 + self.uid + bu

            # Wi = tf.Variable(
            #     tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wi')
            # bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # # self.i_feas = tf.matmul(self.i_feas1, Wi) + self.iid + bi
            # # self.i_feas = self.i_feas1 + self.iid + bi
            # self.i_feas = tf.concat([self.i_feas1], axis=-1)

            self.i_feas = tf.multiply(tf.expand_dims(self.i_feas1, -1), self.i_a)
            # self.i_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.i_feas)

        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            # self.FM = tf.nn.relu(self.FM)

            # self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            # Wmul = tf.Variable(
            #     tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')
            #
            # self.mul = tf.matmul(self.FM, Wmul)
            # self.score = tf.reduce_sum(self.FM, 1, keep_dims=True)

            self.score = tf.keras.layers.Dense(1, activation='linear')(self.FM)
            self.score = tf.reduce_sum(self.score, 1)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


class DeepFM(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, aspect_size=5):
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        # self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        # self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_u_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_u_feature')
        self.input_i_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_i_feature')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        # iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        # uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)
        self.embedding_size = embedding_id
        self.batch_size = tf.shape(self.input_uid)[0]
        self.field_size = 2*aspect_size + 2
        with tf.name_scope("get_embeddings"):
            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")  # 0-0.1???
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])

            ufmf = tf.Variable(tf.random_uniform([1, aspect_size, embedding_id], -0.1, 0.1), name="ufmf")
            ifmf = tf.Variable(tf.random_uniform([1, aspect_size, embedding_id], -0.1, 0.1), name="ifmf")

            self.ufmf = tf.tile(ufmf, [tf.shape(self.input_uid)[0], 1, 1])
            self.embeddings = tf.concat([tf.expand_dims(self.uid, 1), tf.expand_dims(self.iid, 1), tf.tile(ufmf, [tf.shape(self.input_uid)[0], 1, 1]),
                                         tf.tile(ifmf, [tf.shape(self.input_uid)[0], 1, 1])], 1)

            self.feature_value = tf.concat([tf.ones([self.batch_size, 1], dtype=tf.float32), tf.ones([self.batch_size, 1], dtype=tf.float32), self.input_u_feature, self.input_i_feature], -1)
            self.feature_value = tf.expand_dims(self.feature_value, -1)
            self.embeddings = tf.multiply(self.embeddings, self.feature_value)

        with tf.name_scope("first_order"):
            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.uf_bias = tf.Variable(tf.constant(0.1, shape=[1, aspect_size]), name="uf_bias")
            self.if_bias = tf.Variable(tf.constant(0.1, shape=[1, aspect_size]), name="if_bias")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)

            self.y_first_order = tf.concat([self.u_bias, self.i_bias, tf.tile(self.uf_bias, [tf.shape(self.input_uid)[0], 1]),
                                            tf.tile(self.uf_bias, [tf.shape(self.input_uid)[0], 1])], -1)
            self.y_first_order = tf.reduce_sum(tf.multiply(tf.expand_dims(self.y_first_order, -1), self.feature_value), 2)  # ??? 1 or 2

        with tf.name_scope("second_order"):

            self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # square_sum part
            self.squared_features_emb = tf.square(self.embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # second order
            self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                    self.squared_sum_features_emb)  # None * K
            self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_prob)  # None * K

        with tf.name_scope('deep'):
            self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size])  # None * (F*K)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_prob)

            num_layer = 3
            self.weights = dict()
            input_size = self.field_size * self.embedding_size
            glorot = np.sqrt(2.0 / (input_size + n_latent))
            self.weights["layer_0"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, n_latent)), dtype=np.float32)
            self.weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, n_latent)),
                                            dtype=np.float32)  # 1 * layers[0]

            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (n_latent + n_latent))
                self.weights["layer_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(n_latent, n_latent)),
                    dtype=np.float32)  # layers[i-1] * layers[i]
                self.weights["bias_%d" % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, n_latent)), dtype=np.float32)  # 1 * layer[i]

            for i in range(0, num_layer):
                self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
                                     self.weights["bias_%d" % i])  # None * layer[i] * 1
                # if self.batch_norm:
                #     self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase,
                #                                         scope_bn="bn_%d" % i)  # None * layer[i] * 1
                self.y_deep = tf.nn.relu(self.y_deep)
                self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_prob)  # dropout at each Deep layer

        with tf.name_scope('ncf'):
            concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)

            input_size = self.field_size + self.embedding_size + n_latent
            glorot = np.sqrt(2.0 / (input_size + 1))
            self.weights["concat_projection"] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                dtype=np.float32)  # layers[i-1]*layers[i]
            self.weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

            self.score = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

            # #
            # self.score = tf.keras.layers.Dense(1, activation='linear')(self.FM)
            # # self.score = tf.reduce_sum(self.score, 1)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')
            #
            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


class A3NCF(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, aspect_size=5):
        # embedding_id = aspect_size
        # self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        # self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name='input_reuid')
        # self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name='input_reuid')
        self.input_u_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_u_feature')
        self.input_i_feature = tf.placeholder(tf.float32, [None, aspect_size], name='input_i_feature')
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")
        # iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidW")
        # uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidW")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("get_fea"):

            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.1, 0.1), name="iidmf")
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.1, 0.1), name="uidmf")

            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            # # Wu = tf.Variable(
            # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            # bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # # self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu
            # self.u_feas = self.uid + bu
            #
            # # Wi = tf.Variable(
            # #     tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            # bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # # self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi
            # self.i_feas = self.iid + bi

            amf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="amf")
            pmf = tf.Variable(tf.random_uniform([aspect_size, embedding_id], -0.1, 0.1), name="pmf")

            # self.u_feas1 = tf.matmul(self.input_u_feature, amf)
            # self.i_feas1 = tf.matmul(self.input_i_feature, pmf)
            self.u_feas1 = self.input_u_feature
            self.i_feas1 = self.input_i_feature

            # self.u_feas = tf.reshape(self.u_feas, [-1, embedding_id])
            # self.i_feas = tf.reshape(self.i_feas, [-1, embedding_id])

            Wu = tf.Variable(
                tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            # self.u_feas = tf.matmul(self.u_feas1, Wu) + self.uid + bu
            self.u_feas = self.uid + bu
            # self.u_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.u_feas)
            # self.u_feas = self.u_feas1 + self.uid + bu

            Wi = tf.Variable(
                tf.random_uniform([n_latent, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            # self.i_feas = tf.matmul(self.i_feas1, Wi) + self.iid + bi
            # self.i_feas = self.i_feas1 + self.iid + bi
            self.i_feas = self.iid + bi
            # self.i_feas = tf.keras.layers.Dense(n_latent, activation='relu')(self.i_feas)

        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.nn.relu(self.FM)

            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            Wmul = tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

            self.mul = tf.matmul(self.FM, Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


class DeepCoNN(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")

        # self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # self.input_reuid = None
        # self.input_reiid = None
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        pooled_outputs_u = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_users = tf.reshape(self.embedded_users, [-1, review_len_u, embedding_size, 1])  #

                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # [B*rn, len2, 1, 100)
                self.h_u = tf.reshape(h, [-1, review_num_u, review_len_u - filter_size + 1, num_filters])  #
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, review_len_u - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.nn.max_pool(
                    self.h_u,
                    ksize=[1, review_num_u, review_len_u - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_items = tf.reshape(self.embedded_items, [-1, review_len_i, embedding_size, 1])
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.h_i = tf.reshape(h, [-1, review_num_i, review_len_i - filter_size + 1, num_filters])  #
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, item_length - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.nn.max_pool(
                    self.h_i,
                    ksize=[1, review_num_i, review_len_i - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("get_fea"):
            # Wu = tf.get_variable(
            #     "Wu",
            #     shape=[num_filters_total, n_latent],
            #     initializer=tf.contrib.layers.xavier_initializer())
            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_fea = tf.matmul(self.h_drop_u, Wu) + bu
            # self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
            # Wi = tf.get_variable(
            #     "Wi",
            #     shape=[num_filters_total, n_latent],
            #     initializer=tf.contrib.layers.xavier_initializer())
            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_fea = tf.matmul(self.h_drop_i, Wi) + bi
            # self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

        with tf.name_scope('fm'):
            self.z = tf.nn.relu(tf.concat([self.u_fea, self.i_fea], 1))

            # self.z=tf.nn.dropout(self.z,self.dropout_keep_prob)

            WF1 = tf.Variable(
                tf.random_uniform([n_latent * 2, 1], -0.1, 0.1), name='fm1')
            Wf2 = tf.Variable(
                tf.random_uniform([n_latent * 2, 8], -0.1, 0.1), name='fm2')  # fmk
            one = tf.matmul(self.z, WF1)

            inte1 = tf.matmul(self.z, Wf2)
            inte2 = tf.matmul(tf.square(self.z), tf.square(Wf2))

            inter = (tf.square(inte1) - inte2) * 0.5

            inter = tf.nn.dropout(inter, self.dropout_keep_prob)

            inter = tf.reduce_sum(inter, 1, keep_dims=True)
            print(inter)
            b = tf.Variable(tf.constant(0.1), name='bias')

            self.predictions = one + inter + b

            print(self.predictions)

        with tf.name_scope("loss"):
            # losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))


class DeepCoNNpp(object):
    def __init__(
            self, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num, num_classes,
            user_vocab_size, item_vocab_size, n_latent, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")

        # self.input_u = tf.placeholder(tf.int32, [None, user_length], name="input_u")
        # self.input_i = tf.placeholder(tf.int32, [None, item_length], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # self.input_reuid = None
        # self.input_reiid = None
        self.drop0 = tf.placeholder(tf.float32, name="dropout0")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("user_embedding"):
            self.W1 = tf.Variable(
                tf.random_uniform([user_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_user = tf.nn.embedding_lookup(self.W1, self.input_u)
            self.embedded_users = tf.expand_dims(self.embedded_user, -1)

        with tf.name_scope("item_embedding"):
            self.W2 = tf.Variable(
                tf.random_uniform([item_vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_item = tf.nn.embedding_lookup(self.W2, self.input_i)
            self.embedded_items = tf.expand_dims(self.embedded_item, -1)

        pooled_outputs_u = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_users = tf.reshape(self.embedded_users, [-1, review_len_u, embedding_size, 1])  #

                conv = tf.nn.conv2d(
                    self.embedded_users,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # [B*rn, len2, 1, 100)
                self.h_u = tf.reshape(h, [-1, review_num_u, review_len_u - filter_size + 1, num_filters])  #
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, review_len_u - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.nn.max_pool(
                    self.h_u,
                    ksize=[1, review_num_u, review_len_u - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, num_filters_total])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.embedded_items = tf.reshape(self.embedded_items, [-1, review_len_i, embedding_size, 1])
                conv = tf.nn.conv2d(
                    self.embedded_items,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.h_i = tf.reshape(h, [-1, review_num_i, review_len_i - filter_size + 1, num_filters])  #
                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, item_length - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.nn.max_pool(
                    self.h_i,
                    ksize=[1, review_num_i, review_len_i - filter_size + 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)
        self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
        with tf.name_scope("get_fea"):
            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wu')
            bu = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bu")
            self.u_feas = tf.matmul(self.h_drop_u, Wu) + bu
            # self.u_fea = tf.nn.dropout(self.u_fea,self.dropout_keep_prob)
            # Wi = tf.get_variable(
            #     "Wi",
            #     shape=[num_filters_total, n_latent],
            #     initializer=tf.contrib.layers.xavier_initializer())
            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.1, 0.1), name='Wi')
            bi = tf.Variable(tf.constant(0.1, shape=[n_latent]), name="bi")
            self.i_feas = tf.matmul(self.h_drop_i, Wi) + bi
            # self.i_fea=tf.nn.dropout(self.i_fea,self.dropout_keep_prob)

        with tf.name_scope('ncf'):

            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.nn.relu(self.FM)

            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            Wmul = tf.Variable(
                tf.random_uniform([n_latent, 1], -0.1, 0.1), name='wmul')

            self.mul = tf.matmul(self.FM, Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.01, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.01, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')
            l2_loss2 = 0
            l2_loss2 += tf.nn.l2_loss(self.uidW2)
            l2_loss2 += tf.nn.l2_loss(self.iidW2)
            self.predictions = self.score + self.Feature_bias + self.bised
        with tf.name_scope("loss"):
            # losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))

            self.loss = losses + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))
