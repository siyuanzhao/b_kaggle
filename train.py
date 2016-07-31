import tensorflow as tf
import data_helper
from bimbo import BimboRNN
import time
import numpy as np
import datetime
import math
import pandas as pd

tf.flags.DEFINE_integer('embedding_size', 20, 'Dimensionality of product embedding')
tf.flags.DEFINE_integer('batch_size', 5000, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 400, 'Number of training epochs')
tf.flags.DEFINE_integer('hidden_size', 50, 'Number of hidden units')
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_integer("max_grad_norm", 1.0, "Maximum gradient norm. 40.0")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ===================================

print 'loading data...'

data_file = 'train_pivot.csv'

product_l, data = data_helper.read_data(data_file)

data_size = data.shape[0]
product_num = len(product_l)

epoch_steps = data_size / FLAGS.batch_size
# turn to pandas dataframe
product_l = pd.DataFrame(product_l)
product_l.columns = ['Producto_ID']
product_l['index1'] = product_l.index

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
    sess = tf.Session(config=session_conf)
    rnn = BimboRNN(FLAGS.batch_size, FLAGS.embedding_size, product_num, FLAGS.hidden_size, 6)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        timestamp = str(int(time.time()))
        decay_lr = tf.train.exponential_decay(0.001, global_step, 3000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(decay_lr)
        # gradient pipeline
        grads_and_vars = optimizer.compute_gradients(rnn.rmse)

        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                          for g, v in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             name="train_op",
                                             global_step=global_step)
        # Initialize all variables
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(tf.all_variables())

        for j in range(FLAGS.num_epochs):
            total_loss = 0
            # shuffle data
            # data = shuffle(data)
            data = data.iloc[np.random.permutation(len(data))]
            for i in range(epoch_steps):

                data_batch = data.iloc[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                demand_nums = data_batch[['3','4','5','6','7','8','9']].as_matrix()
                demand_nums[:,1] = demand_nums[:,0] + demand_nums[:,1]
                demand_nums[:,2] = demand_nums[:,1] + demand_nums[:,2]
                demand_nums[:,3] = demand_nums[:,2] + demand_nums[:,3]
                demand_nums[:,4] = demand_nums[:,3] + demand_nums[:,4]
                demand_nums[:,5] = demand_nums[:,4] + demand_nums[:,5]
                demand_nums[:,6] = demand_nums[:,5] + demand_nums[:,6]
                product_ids = []
                products = data_batch['Producto_ID']
                products_index = pd.merge(data_batch, product_l, how='left')
                products_index.fillna(product_num, inplace=True)
                product_ids = products_index['index1'].as_matrix()

                #tweak_nums = np.concatenate((demand_nums[:,0:1], demand_nums[:,1:-1]), axis=1)
                tweak_nums = demand_nums[:, 0:-1]
                feed_dict = {rnn.demand_nums: demand_nums[:, 1:], rnn.product_ids: product_ids, rnn.tweak_nums: tweak_nums}

                _, step, loss, pred, o_w = sess.run([train_op, global_step, rnn.loss, rnn.pred, rnn.o_w], feed_dict)
                total_loss += loss**2
                time_str = datetime.datetime.now().isoformat()
                if i % 200 == 0:
                    print '{} -- Epoch {}, Step {}, loss: {}'.format(time_str, j, i,loss)
                    with open('pred_log', 'a') as f:
                        f.write(pred)
                        f.write('\n')
            total_loss = math.sqrt(total_loss/epoch_steps)
            print 'Epoch {} overall loss: {}'.format(j, total_loss)
            try:
                with open('result_log', 'a') as f:
                    f.write('Epoch {} overall loss: {}'.format(j, total_loss))
                    f.write('\n')
                if j>0 and j%50 == 0:
                    print 'saving model...'
                    saver.save(sess, 'checkpoint_epoch_{}'.format(j))
            except:
                print 'Unpected error.'
