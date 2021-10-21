from datetime import datetime
import os, time, sys
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.metrics import confusion_matrix
from train_process import *
import matplotlib.pyplot as plt
np.set_printoptions(threshold=5)


class MiNet(object):
    def __init__(self, dataset, model):
        self.start_lr = 1e-3
        self.train_iters = 20000
        self.datasets = dataset

        self.dimension = 315
        self.display_step = 50
        self.snapshot = 200
        self.model_save_dir = model
        self.batch_size = 32
        self.weight_decay = 1e-3
        self.stddev = 0.1

        self.results_dir = './results/train'

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
            # net = slim.fully_connected(fc9, 23, activation_fn=None, normalizer_fn=None, scope='out1')

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
        # fea_5 = self.model5(fea5, is_training)

        pred = self.model(fea_1, fea_2, fea_3, fea_4, fea, is_training)

        cost = self.compute_loss(label, pred, is_training)
        accuracy = self.compute_accuracy(label, pred)

        global_step = tf.Variable(0, trainable=False)
        l_r = tf.train.exponential_decay(self.start_lr, global_step, 2000, 0.5, staircase=True)
        optimizer = tf.train.AdamOptimizer(l_r, 0.9).minimize(cost, global_step)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # if update_ops:
        #    updates = tf.group(*update_ops)
        #    cost = control_flow_ops.with_dependencies([updates], cost)
        # init = tf.global_variables_initializer()
        init = tf.initialize_all_variables()
        saver = tf.train.Saver(max_to_keep=50)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)
            best_te_loss = 100.
            step = 1
            while step < self.train_iters:

                batch_fea, feas, batch_label, name = self.datasets.get_next_batch(step, fold, True, self.batch_size)

                # print batch_fea, batch_label
                start_time = time.time()
                # load batch_data
                _, batch_loss, acc = sess.run([optimizer, cost, accuracy],
                                              feed_dict={fea1: batch_fea[0], fea2: batch_fea[1],
                                                         fea3: batch_fea[2], fea4: batch_fea[3],
                                                         fea: feas,
                                                         label: batch_label, is_training: True})
                duration = time.time() - start_time
                if step % self.display_step == 0:
                    examples_per_sec = self.batch_size / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step = %d, loss = %.5f ( %.1f example/sec; %.3f sec/batch)')
                    print(format_str % (datetime.now(), step, batch_loss, examples_per_sec, sec_per_batch))

                if step % self.snapshot == 0 or (step + 1) == self.train_iters:
                    # test
                    num_te_batch = len(self.datasets.datasets[fold]['test'])
                    # num_te_batch = len(self.datasets.te_dataset)

                    te_loss = np.zeros((num_te_batch), dtype=np.float32)
                    te_acc = np.zeros((num_te_batch), dtype=np.float32)
                    te_label = np.zeros((num_te_batch), dtype=np.float32)
                    te_logits = np.zeros((num_te_batch), dtype=np.float32)
                    for idx in range(num_te_batch):
                        te_batch_fea, feas, te_batch_label, name = self.datasets.get_next_batch(idx, fold, False, 1, 0)

                        te_loss[idx], te_acc[idx] = sess.run([cost, accuracy],
                                                             feed_dict={fea1: te_batch_fea[0], fea2: te_batch_fea[1],
                                                                        fea3: te_batch_fea[2], fea4: te_batch_fea[3],
                                                                        fea: feas,
                                                                        label: te_batch_label, lr: 0.,
                                                                        is_training: False})
                    format_str = ('%s: step = %d, test loss = %.4f, test accuracy = %.4f ')
                    te_loss = np.mean(te_loss)
                    te_acc = np.mean(te_acc)
                    # tn, fp, fn, tp = confusion_matrix(te_label, np.rint(te_logits)).ravel()
                    # te_auc = roc_auc_score(te_label, te_logits)
                    print('Testing:')
                    print(format_str % (datetime.now(), step, te_loss, te_acc))
                    '''
                    #train
                    num_tr_batch = len(self.datasets.datasets[fold]['train'])
                    #num_tr_batch = len(self.datasets.tr_dataset[fold])
                    tr_loss = np.zeros((num_tr_batch), dtype=np.float32)
                    tr_acc = np.zeros((num_tr_batch), dtype=np.float32)
                    for idx in range(num_tr_batch):
                        tr_batch_fea,feas, tr_batch_label, name = self.datasets.get_next_batch(idx, fold,False,1,1)

                        tr_loss[idx], tr_acc[idx] = sess.run([cost, accuracy], feed_dict={fea1:tr_batch_fea[0],fea2:tr_batch_fea[1],
                                                                           fea3:tr_batch_fea[2],fea4:tr_batch_fea[3],fea5:tr_batch_fea[4],fea:feas,
                                                                           label:tr_batch_label, lr:0., is_training:False})
                    format_str = ('%s: step = %d, train loss = %.4f, train accuracy = %.4f ')
                    tr_loss = np.mean(tr_loss)
                    tr_acc = np.mean(tr_acc)
                    print 'Training:'
                    print format_str % (datetime.now(), step, tr_loss, tr_acc)
                    '''
                    # save model
                    if te_loss < best_te_loss:
                        best_te_loss = te_loss
                        saver.save(sess, self.model_save_dir + '/best_model.ckpt')
                        print('Model Saved')

                step = step + 1
            print('Optimization Finished!')

            # compute test accuracy of best model
            saver.restore(sess, self.model_save_dir + '/best_model.ckpt')
            num_te_batch = len(self.datasets.datasets[fold]['test'])
            # num_te_batch = len(self.datasets.te_dataset)
            te_loss = np.zeros((num_te_batch), dtype=np.float32)
            te_acc = np.zeros((num_te_batch), dtype=np.float32)
            te_label = np.zeros((num_te_batch), dtype=np.float32)
            # te_logits  = np.zeros((num_te_batch), dtype=np.float32)
            y_pred, y_true = [], []
            for idx in range(num_te_batch):
                te_batch_fea, feas, te_batch_label, name = self.datasets.get_next_batch(idx, fold, False, 1, 0)

                te_loss[idx], te_acc[idx], te_logits = sess.run([cost, accuracy, pred],
                                                                feed_dict={fea1: te_batch_fea[0], fea2: te_batch_fea[1],
                                                                           fea3: te_batch_fea[2], fea4: te_batch_fea[3],
                                                                           fea: feas,
                                                                           label: te_batch_label,
                                                                           lr: 0., is_training: False})
                true = np.argmax(te_batch_label)
                preds = np.argmax(te_logits)
                y_true.append(true)
                y_pred.append(preds)
                if true != preds:
                    with open(self.results_dir + '/wrong_' + str(true) + '.txt', 'a+') as f:
                        f.write(str(name) + '  ' + str(true) + '  ' + str(preds) + '\n')
                if true == preds:
                    with open(self.results_dir + '/right_' + str(true) + '.txt', 'a+') as f:
                        f.write(str(name) + '  ' + str(true) + '  ' + str(preds) + '\n')
            format_str = ('%s: step = %d, test loss = %.4f, test accuracy = %.4f')
            te_loss = np.mean(te_loss)
            te_acc = np.mean(te_acc)
            # fpr, tpr, thresholds = roc_curve(te_label, te_logits)
            cm = confusion_matrix(y_pred=y_pred, y_true=y_true)
            # te_auc = roc_auc_score(te_label, te_logits)
            print(cm)
            print('Testing:')
            print(format_str % (datetime.now(), step, te_loss, te_acc))

            with open(self.results_dir + '/results.txt', 'w+') as f1:
                f1.write('rc best acc:' + str(te_acc) + '\n')
                # f1.write('rc best auc:'+ str(te_auc)+ '\n')
                f1.write('confusion matrix:\n' + str(cm) + '\n')
                # f1.write('data point:\n'+'x:'+ str(fpr)+'\n' + str(tpr))
                f1.close()

        return te_acc


if __name__ == '__main__':
    model = './models/model_new'
    seed = [22 + i * 5 for i in range(1)]
    acc = np.zeros((1, 1))
    # auc = np.zeros((1, 1))
    for idx in range(1):
        dataset = Dataset(seed[idx])
        print('Successfully loading data\n')
        for fold in range(1):
            print('run=', idx, ' fold=', fold)
            net = MiNet(dataset, model)
            acc[idx][fold] = net.run_net(fold)
            tf.reset_default_graph()