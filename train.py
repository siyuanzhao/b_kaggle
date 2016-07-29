import tensorflow as tf
import data_helper
from bimbo import BimboRNN
import time
import numpy as np
import datetime
import math
from sklearn.utils import shuffle

tf.flags.DEFINE_integer('embedding_size', 100, 'Dimensionality of product embedding')
tf.flags.DEFINE_integer('batch_size', 500, 'Batch size')
tf.flags.DEFINE_integer('num_epochs', 400, 'Number of training epochs')
tf.flags.DEFINE_integer('hidden_size', 200, 'Number of hidden units')
tf.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.flags.DEFINE_integer("max_grad_norm", 40.0, "Maximum gradient norm. 40.0")

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
product_l = product_l.tolist()

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)
    sess = tf.Session(config=session_conf)
    rnn = BimboRNN(FLAGS.batch_size, FLAGS.embedding_size, product_num, FLAGS.hidden_size, 7)
    with sess.as_default():
        global_step = tf.Variable(0, name="global_step", trainable=False)
        timestamp = str(int(time.time()))
        decay_lr = tf.train.exponential_decay(0.01, global_step, 10000, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(decay_lr)
        # gradient pipeline
        grads_and_vars = optimizer.compute_gradients(rnn.loss)

        grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                          for g, v in grads_and_vars if g is not None]
        train_op = optimizer.apply_gradients(grads_and_vars,
                                             name="train_op",
                                             global_step=global_step)
        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        for j in range(FLAGS.num_epochs):
            total_loss = 0
            # shuffle data
            # data = shuffle(data)
            data.iloc[np.random.permutation(len(data))]
            for i in range(epoch_steps):

                data_batch = data.iloc[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]
                demand_nums = data_batch[['3','4','5','6','7','8','9']].as_matrix()
                product_ids = []
                products = data_batch['Producto_ID'].as_matrix()

                for p in products:
                    if p in product_l:
                        product_ids.append(product_l.index(p))
                    else:
                        product_ids.append(product_num)

                ones = np.ones([FLAGS.batch_size, 1])

                tweak_nums = np.concatenate((ones, demand_nums[:,:-1]), axis=1)

                feed_dict = {rnn.demand_nums: demand_nums, rnn.product_ids: product_ids, rnn.tweak_nums: tweak_nums}
                _, step, loss = sess.run([train_op, global_step, rnn.loss], feed_dict)
                total_loss += loss**2
                time_str = datetime.datetime.now().isoformat()
                if i % 200 == 0:
                    print '{} -- Epoch {}, Step {}, loss: {}'.format(time_str, j, i,loss)
            total_loss = math.sqrt(total_loss/epoch_steps)
            print 'Epoch {} overall loss: {}'.format(j, total_loss)
            with open('result_log', 'a') as f:
                f.write('Epoch {} overall loss: {}'.format(j, total_loss))
                f.write('\n')
