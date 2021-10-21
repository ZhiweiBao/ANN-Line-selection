from datetime import datetime
import os, time, sys

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from test_process import *
import matplotlib.pyplot as plt

np.set_printoptions(threshold=5)


class mi_net(object):
    def __init__(self, dataset, model):
        self.start_lr = 1e-3
        self.train_iters = 100000
        self.datasets = dataset

        self.dimension = 315
        self.display_step = 50
        self.snapshot = 1000
        self.model_save_dir = model
        self.batch_size = 8
        self.weight_decay = 1e-3
        self.stddev = 0.1
        self.classes = ['1', '2', '3', '4']

        self.results_dir = './results/result_data190724_10'

    def model1(self, inputs, is_training):

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            ):
            def rc_block(net, output_nodes, is_training, scope):
                fc = slim.fully_connected(net, output_nodes, scope=scope + '1/fc1')
                net = slim.dropout(fc, keep_prob=0.75, is_training=is_training, scope=scope + '1/dp')
                return net
                # raw data

            fc1 = rc_block(inputs, 64, is_training=is_training, scope='rc1')
            fc2 = rc_block(fc1, 64, is_training=is_training, scope='rc2')
            fc3 = rc_block(fc2, 64, is_training=is_training, scope='rc3')
            fc4 = rc_block(fc3, 64, is_training=is_training, scope='rc4')
            fc5 = rc_block(fc4, 64, is_training=is_training, scope='rc5')
            fc6 = rc_block(fc5, 64, is_training=is_training, scope='rc6')

            fc9 = tf.add_n([fc1, fc2, fc3, fc4, fc5, fc6])
            # net = slim.fully_connected(fc9, 6, activation_fn=None, normalizer_fn=None, scope='out1')

        return fc9

    def model2(self, inputs, is_training):

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            ):
            def rc_block(net, output_nodes, is_training, scope):
                fc = slim.fully_connected(net, output_nodes, scope=scope + '2/fc1')
                net = slim.dropout(fc, keep_prob=0.75, is_training=is_training, scope=scope + '2/dp')
                return net
                # raw data

            fc1 = rc_block(inputs, 64, is_training=is_training, scope='rc1')
            fc2 = rc_block(fc1, 64, is_training=is_training, scope='rc2')
            fc3 = rc_block(fc2, 64, is_training=is_training, scope='rc3')
            fc4 = rc_block(fc3, 64, is_training=is_training, scope='rc4')
            fc5 = rc_block(fc4, 64, is_training=is_training, scope='rc5')
            fc6 = rc_block(fc5, 64, is_training=is_training, scope='rc6')

            fc9 = tf.add_n([fc1, fc2, fc3, fc4, fc5, fc6])
            # net = slim.fully_connected(fc9, 6, activation_fn=None, normalizer_fn=None, scope='out2')

        return fc9

    def model3(self, inputs, is_training):

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            ):
            def rc_block(net, output_nodes, is_training, scope):
                fc = slim.fully_connected(net, output_nodes, scope=scope + '3/fc1')
                net = slim.dropout(fc, keep_prob=0.75, is_training=is_training, scope=scope + '3/dp')
                return net
                # raw data

            fc1 = rc_block(inputs, 64, is_training=is_training, scope='rc1')
            fc2 = rc_block(fc1, 64, is_training=is_training, scope='rc2')
            fc3 = rc_block(fc2, 64, is_training=is_training, scope='rc3')
            fc4 = rc_block(fc3, 64, is_training=is_training, scope='rc4')
            fc5 = rc_block(fc4, 64, is_training=is_training, scope='rc5')
            fc6 = rc_block(fc5, 64, is_training=is_training, scope='rc6')

            fc9 = tf.add_n([fc1, fc2, fc3, fc4, fc5, fc6])
            # net = slim.fully_connected(fc9, 6, activation_fn=None, normalizer_fn=None, scope='out3')

        return fc9

    def model4(self, inputs, is_training):

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            ):
            def rc_block(net, output_nodes, is_training, scope):
                fc = slim.fully_connected(net, output_nodes, scope=scope + '4/fc1')
                net = slim.dropout(fc, keep_prob=0.75, is_training=is_training, scope=scope + '4/dp')
                return net
                # raw data

            fc1 = rc_block(inputs, 64, is_training=is_training, scope='rc1')
            fc2 = rc_block(fc1, 64, is_training=is_training, scope='rc2')
            fc3 = rc_block(fc2, 64, is_training=is_training, scope='rc3')
            fc4 = rc_block(fc3, 64, is_training=is_training, scope='rc4')
            fc5 = rc_block(fc4, 64, is_training=is_training, scope='rc5')
            fc6 = rc_block(fc5, 64, is_training=is_training, scope='rc6')

            fc9 = tf.add_n([fc1, fc2, fc3, fc4, fc5, fc6])
            # net = slim.fully_connected(fc9, 6, activation_fn=None, normalizer_fn=None, scope='out4')

        return fc9

    def model5(self, inputs, is_training):

        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay),
                            ):
            def rc_block(net, output_nodes, is_training, scope):
                fc = slim.fully_connected(net, output_nodes, scope=scope + '5/fc1')
                net = slim.dropout(fc, keep_prob=0.75, is_training=is_training, scope=scope + '5/dp')
                return net
                # raw data

            fc1 = rc_block(inputs, 64, is_training=is_training, scope='rc1')
            fc2 = rc_block(fc1, 64, is_training=is_training, scope='rc2')
            fc3 = rc_block(fc2, 64, is_training=is_training, scope='rc3')
            fc4 = rc_block(fc3, 64, is_training=is_training, scope='rc4')
            fc5 = rc_block(fc4, 64, is_training=is_training, scope='rc5')
            fc6 = rc_block(fc5, 64, is_training=is_training, scope='rc6')

            fc9 = tf.add_n([fc1, fc2, fc3, fc4, fc5, fc6])
            # net = slim.fully_connected(fc9, 6, activation_fn=None, normalizer_fn=None, scope='out5')

        return fc9

    def model(self, fea1, fea2, fea3, fea4, fea, is_training):
        with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(stddev=self.stddev),
                            weights_regularizer=slim.l2_regularizer(self.weight_decay)):
            inputs = tf.concat([fea1, fea2, fea3, fea4], 1)

            def rc_block(net, output_nodes, is_training, scope):
                fc = slim.fully_connected(net, output_nodes, scope=scope + '6/fc1')
                net = slim.dropout(fc, keep_prob=0.75, is_training=is_training, scope=scope + '6/dp')
                return net
                # raw data

            fc1 = rc_block(fea, 64, is_training=is_training, scope='rc1')
            fc2 = rc_block(fc1, 64, is_training=is_training, scope='rc2')
            fc3 = rc_block(fc2, 64, is_training=is_training, scope='rc3')
            fc4 = rc_block(fc3, 64, is_training=is_training, scope='rc4')
            fc5 = rc_block(fc4, 64, is_training=is_training, scope='rc5')
            fc6 = rc_block(fc5, 64, is_training=is_training, scope='rc6')

            fc7 = tf.add_n([fc1, fc2, fc3, fc4, fc5, fc6])
            inputs = tf.concat([fc7, inputs], 1)

            net = slim.fully_connected(inputs, 4, activation_fn=None, normalizer_fn=None, scope='out')
        return net

    def compute_loss(self, y, logits, is_training):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        tf.add_to_collection('losses', regularization_losses)
        return tf.cond(is_training,
                       lambda: tf.add_n(tf.get_collection('losses'), name='total_loss'),
                       lambda: cross_entropy_mean)

    def compute_accuracy(self, y, logits):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return acc

    def run_net(self, fold=0):
        fea1 = tf.placeholder(tf.float32, shape=(None, 63), name='fea1')
        fea2 = tf.placeholder(tf.float32, shape=(None, 117), name='fea2')
        fea3 = tf.placeholder(tf.float32, shape=(None, 153), name='fea3')
        fea4 = tf.placeholder(tf.float32, shape=(None, 63), name='fea4')
        fea = tf.placeholder(tf.float32, shape=(None, 396), name='fea5')
        label = tf.placeholder(tf.int32, name='label')
        lr = tf.placeholder(tf.float32, name='lr')
        is_training = tf.placeholder(tf.bool, name='is_training')

        fea_1 = self.model1(fea1, is_training)
        fea_2 = self.model2(fea2, is_training)
        fea_3 = self.model3(fea3, is_training)
        fea_4 = self.model4(fea4, is_training)

        pred = self.model(fea_1, fea_2, fea_3, fea_4, fea, is_training)
        cost = self.compute_loss(label, pred, is_training)
        accuracy = self.compute_accuracy(label, pred)

        global_step = tf.Variable(0, trainable=False)
        l_r = tf.train.exponential_decay(self.start_lr, global_step, 10000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(l_r, 0.9).minimize(cost, global_step)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # if update_ops:
        #    updates = tf.group(*update_ops)
        #    cost = control_flow_ops.with_dependencies([updates], cost)
        # init = tf.global_variables_initializer()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=50)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        variables_to_restore = slim.get_variables_to_restore()
        variable_restore_op = slim.assign_from_checkpoint_fn('./models/model_new/best_model.ckpt', variables_to_restore,
                                                             ignore_missing_vars=False)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            variable_restore_op(sess)

            num_te_batch = len(self.datasets.datasets)
            # num_te_batch = len(self.datasets.te_dataset)
            te_loss = np.zeros((num_te_batch), dtype=np.float32)
            te_acc = np.zeros((num_te_batch), dtype=np.float32)
            te_label = np.zeros((num_te_batch), dtype=np.float32)
            # te_logits  = np.zeros((num_te_batch), dtype=np.float32)
            y_true, y_pred = [], []
            if not os.path.exists(self.results_dir):
                os.mkdir(self.results_dir)
            for idx in range(num_te_batch):
                te_batch_fea, feas, te_batch_label, name = self.datasets.get_next_batch(idx, fold, False, 1, 0)

                te_loss[idx], te_acc[idx], te_logits = sess.run([cost, accuracy, pred],
                                                                feed_dict={fea1: te_batch_fea[0], fea2: te_batch_fea[1],
                                                                           fea3: te_batch_fea[2], fea4: te_batch_fea[3],
                                                                           fea: feas,
                                                                           label: te_batch_label, lr: 0.,
                                                                           is_training: False})
                y_true.append(np.argmax(te_batch_label))
                y_pred.append(np.argmax(te_logits))
                true = np.argmax(te_batch_label)
                preds = np.argmax(te_logits)
                cla = self.classes[preds]
                if true != preds:
                    with open(self.results_dir + '/wrong_' + str(name) + '.txt', 'a+') as f:
                        f.write(str(name) + '  ' + str(cla) + '  ' + str(preds) + '  ' + str(te_logits) + '\n')
                if true == preds:
                    with open(self.results_dir + '/right_' + str(name) + '.txt', 'a+') as f:
                        f.write(str(name) + '  ' + str(cla) + '  ' + str(preds) + '  ' + str(te_logits) + '\n')
            format_str = ('%s: test loss = %.4f, test accuracy = %.4f')
            te_loss = np.mean(te_loss)
            te_acc = np.mean(te_acc)
            # fpr, tpr, thresholds = roc_curve(te_label, te_logits)
            c_m = confusion_matrix(y_true=y_true, y_pred=y_pred)
            # te_auc = roc_auc_score(te_label, te_logits)
            print('Testing:')
            print(format_str % (datetime.now(), te_loss, te_acc))
            print(c_m)
            with open(self.results_dir + '/results.txt', 'w+') as f1:
                f1.write('rc best acc:' + str(te_acc) + '\n')
                f1.write('confusion matrix:\n' + str(c_m))
                # f1.write('rc best auc:'+ str(te_auc)+ '\n')
                # f1.write('confusion matrix:\n'+'tn:'+ str(tn)+' fp:' + str(fp)+' fn:' +str(fn)+' tp:' +str(tp)+'\n')
                # f1.write('data point:\n'+'x:'+ str(fpr)+'\n' + str(tpr))
                f1.close()

        return te_acc


if __name__ == '__main__':
    model = './models'
    seed = 22
    dataset = Dataset(seed)
    net = mi_net(dataset, model)
    acc = net.run_net()



