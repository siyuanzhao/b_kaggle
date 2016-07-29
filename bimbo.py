import tensorflow as tf


class BimboRNN(object):
    # RNN to solve Bimbo problem

    def __init__(self,
                 batch_size, embedding_dim, product_num, hidden_size, num_weeks):
        self.time_step = 11
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._product_num = product_num
        self._hidden_size = hidden_size
        self._num_weeks = num_weeks
        self.build_input()
        # build the model
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embeddings for product
            self.W = tf.get_variable(
                'product_embeddings', [self._product_num+1, self._embedding_dim],
                initializer=tf.contrib.layers.xavier_initializer())
            pro_embeddings = tf.nn.embedding_lookup(self.W, self.product_ids)
        # gru cell
        gru_cell = tf.nn.rnn_cell.GRUCell(self._hidden_size)

        # build inputs [batch_size, num_weeks, embedding_dim]
        pro_embeddings_expanded = tf.expand_dims(pro_embeddings, [1])
        demand_nums_expanded = tf.expand_dims(self.demand_nums, [-1])
        inputs = pro_embeddings_expanded * demand_nums_expanded
        reshaped_inputs = tf.reshape(inputs, [-1, self._embedding_dim])
        split_inputs = tf.split(0, self._num_weeks, reshaped_inputs)
        
        outputs, states = tf.nn.rnn(gru_cell, split_inputs, dtype=tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, self._hidden_size])
        sigmoid_w = tf.get_variable('sigmoid_w', [self._hidden_size, 1],
                                    initializer=tf.contrib.layers.xavier_initializer())
        sigmoid_b = tf.get_variable('sigmoid_b', [1],
                                    initializer=tf.truncated_normal_initializer())
        pred = tf.nn.relu(tf.matmul(output, sigmoid_w) + sigmoid_b)
        pred = tf.reshape(pred, [-1, self._num_weeks])
        # loss function
        loss = tf.sqrt(tf.reduce_mean(tf.square(tf.log(pred+1)-tf.log(self.demand_nums+1))))

        self.loss = loss
        self.pred = pred

    def build_input(self):
        self.product_ids = tf.placeholder(tf.int32, [self._batch_size])
        self.demand_nums = tf.placeholder(tf.float32, [self._batch_size, self._num_weeks])
        self.tweak_nums = tf.placeholder(tf.float32, [self._batch_size, self._num_weeks])
