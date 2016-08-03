import tensorflow as tf


class BimboRNN(object):
    # RNN to solve Bimbo problem

    def __init__(self,
                 batch_size, embedding_dim, product_num, hidden_size, num_weeks):
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim
        self._product_num = product_num
        self._hidden_size = hidden_size
        self._output_size = 15
        self._num_weeks = num_weeks
        self.build_input()
        # build the model
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # embeddings for product
            self.W = tf.get_variable(
                'product_embeddings', [self._product_num+1, self._embedding_dim],
                initializer=tf.truncated_normal_initializer())
            pro_embeddings = tf.nn.embedding_lookup(self.W, self.product_ids)
        # gru cell
        gru_cell1 = tf.nn.rnn_cell.GRUCell(self._hidden_size)
        gru_cell2 = tf.nn.rnn_cell.GRUCell(self._output_size)
        gru_cell = tf.nn.rnn_cell.MultiRNNCell([gru_cell1, gru_cell2])
        # build inputs [batch_size, num_weeks, embedding_dim]
        pro_embeddings_expanded = tf.expand_dims(pro_embeddings, [1])
        ones = tf.ones([self._num_weeks, self._embedding_dim])
        pro_embeddings_expanded = pro_embeddings_expanded * ones
        tweak_nums_expanded = tf.expand_dims(self.tweak_nums, [-1])
        #inputs = pro_embeddings_expanded * tweak_nums_expanded
        inputs = tf.concat(2, [pro_embeddings_expanded, tweak_nums_expanded])
        self.inputs = inputs
        #reshaped_inputs = tf.reshape(inputs, [-1, self._embedding_dim])
        reshaped_inputs = tf.reshape(inputs, [-1, self._embedding_dim+1])
        split_inputs = tf.split(0, self._num_weeks, reshaped_inputs)
        
        outputs, states = tf.nn.rnn(gru_cell, split_inputs, dtype=tf.float32)
        output = tf.reshape(tf.concat(1, outputs), [-1, self._output_size])
        #output = tf.reshape(tf.concat(1, outputs), [-1, self._num_weeks])
        # sigmoid_w = tf.get_variable('sigmoid_w', [self._hidden_size, self._output_size],
        #                            initializer=tf.truncated_normal_initializer())
        #sigmoid_b = tf.get_variable('sigmoid_b', [self._output_size],
        #                            initializer=tf.truncated_normal_initializer())
        o_w = tf.get_variable('o_w', [self._output_size, 1],
                              initializer=tf.truncated_normal_initializer(mean=20, stddev=5))
        o_b = tf.get_variable('o_b', [1], initializer=tf.truncated_normal_initializer(mean=10))
        #o = tf.nn.relu(tf.matmul(output, sigmoid_w) + sigmoid_b)

        pred = tf.sigmoid(tf.matmul(output, o_w) + o_b)
        #pred = output
        pred = tf.reshape(pred, [-1, self._num_weeks])
        # loss function
        #rmse = tf.sqrt(tf.reduce_sum(tf.square(pred-self.demand_nums)))
        loss = tf.sqrt(tf.reduce_mean(tf.square(pred-tf.log(self.demand_nums+1)), 0))
        pred = tf.exp(pred)
        rmse = loss
        self.loss = loss
        self.pred = pred
        self.rmse = rmse
        self.o_w = o_w

    def build_input(self):
        self.product_ids = tf.placeholder(tf.int32, [self._batch_size])
        self.demand_nums = tf.placeholder(tf.float32, [self._batch_size, self._num_weeks])
        self.tweak_nums = tf.placeholder(tf.float32, [self._batch_size, self._num_weeks])
