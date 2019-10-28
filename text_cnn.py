import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, num_quantized_chars, sequence_char_length, vocab_size, vocab_entity_size, vocab_pos_size,
    embedding_size, embedding_entity_size, embedding_pos_size, embedding_char_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

            # Placeholders for input, output and dropout
            self.input_x_entity = tf.placeholder(tf.int32, [None, sequence_length], name="input_entityvector")
            self.input_x_pos = tf.placeholder(tf.int32, [None, sequence_length], name="input_posvector")
            self.input_x_hand = tf.placeholder(tf.float32, [None, 1], name="input_x_hand")
            self.input_bag_of_entity = tf.placeholder(tf.float32, [None, 1000], name="input_bag_of_entity")
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_char = tf.placeholder(tf.int32, [None, sequence_char_length], name="input_char")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

            # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(l2_reg_lambda)

            # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W", trainable= True)

                # self.place_w = tf.placeholder(tf.float32, shape=(vocab_size, embedding_size))
                # self.set_W = tf.assign(self.W, self.place_w, validate_shape=False)

                self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

            # pos embedding layer
            # with tf.device('/cpu:0'), tf.name_scope("pos-embedding"):
            #     self.W_pos = tf.Variable(tf.random_uniform([vocab_entity_size, embedding_entity_size], -1.0, 1.0),
            #                              name="W_pos", trainable=True)
            #     self.embedded_tags = tf.nn.embedding_lookup(self.W_pos, self.input_x_entity)
            #
            # # concatenate word embedding and pos embedding
            # self.embedded_concat = tf.concat([self.embedded_chars, self.embedded_tags], 2)
            # concat_dim = embedding_size + embedding_entity_size
            #
            # # expand to the 4th dimension
            # self.embedded_concat = tf.expand_dims(self.embedded_concat, -1)


            # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    conv = tf.layers.batch_normalization(inputs=conv, momentum=0.997, epsilon=1e-5, center=True, scale=True)

                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(pooled_outputs, 3)
            self.h_pool_utt_cnn = tf.reshape(self.h_pool, [-1, num_filters_total])

            ####################################################################################
            ####################################################################################

            text_length = self._length(self.input_char)
            # Embedding Lookup 16
            with tf.name_scope("embedding_char"):
                self.embedding_character = tf.get_variable(name='lookup_Wchar',
                                                        shape=[num_quantized_chars, embedding_char_size],
                                                        initializer=tf.keras.initializers.he_uniform(), trainable=True)
                self.embedded_characters = tf.nn.embedding_lookup(self.embedding_character, self.input_char)

            # Create a convolution + maxpool layer for each filter size
            with tf.name_scope("rnn_char"):
                with tf.variable_scope('rnn_chr') as scope:
                    cell = self._get_cell(sequence_char_length, 'gru')
                    cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                    # cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, sequence_char_length)
                    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
                                                                             cell_bw=cell_with_attention,
                                                                             inputs=self.embedded_characters,
                                                                             sequence_length=text_length,
                                                                             dtype=tf.float32)

                    output_fw, output_bw = outputs
                    all_outputs = tf.concat([output_fw, output_bw], 2)
                    self.h_pool_flat_char = self.last_relevant(all_outputs, sequence_char_length)

            ####################################################################################
            ####################################################################################

            text_length = self._length(self.input_x_entity)
            # Embedding Lookup 16
            with tf.name_scope("embedding_entiy"):
                self.embedding_entity = tf.get_variable(name='lookup_Went',
                                                        shape=[vocab_entity_size, embedding_entity_size],
                                                        initializer=tf.keras.initializers.he_uniform(), trainable=True)
                self.embedded_entity = tf.nn.embedding_lookup(self.embedding_entity, self.input_x_entity)

            # Create a convolution + maxpool layer for each filter size
            with tf.name_scope("rnn_charent"):
                with tf.variable_scope('rnn_chrent') as scope:
                    cell = self._get_cell(sequence_length, 'gru')
                    cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                    cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, sequence_length)
                    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
                                                                             cell_bw=cell_with_attention,
                                                                             inputs=self.embedded_entity,
                                                                             sequence_length=text_length,
                                                                             dtype=tf.float32)

                    output_fw, output_bw = outputs
                    all_outputs = tf.concat([output_fw, output_bw], 2)
                    self.h_pool_flat_entity = self.last_relevant(all_outputs, sequence_length)

            ####################################################################################
            ####################################################################################

            text_length = self._length(self.input_x_pos)
            # Embedding Lookup 16
            with tf.name_scope("embedding_pos"):
                self.embedding_pos = tf.get_variable(name='lookup_W1pos',
                                                        shape=[vocab_pos_size, embedding_pos_size],
                                                        initializer=tf.keras.initializers.he_uniform(), trainable=True)
                self.embedded_pos = tf.nn.embedding_lookup(self.embedding_pos, self.input_x_pos)

            # Create a convolution + maxpool layer for each filter size
            with tf.name_scope("rnn_charpos"):
                with tf.variable_scope('rnn_chrpos') as scope:
                    cell = self._get_cell(sequence_length, 'gru')
                    cell_with_attention = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
                    cell_with_attention = tf.contrib.rnn.AttentionCellWrapper(cell_with_attention, sequence_length)
                    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_with_attention,
                                                                             cell_bw=cell_with_attention,
                                                                             inputs=self.embedded_pos,
                                                                             sequence_length=text_length,
                                                                             dtype=tf.float32)

                    output_fw, output_bw = outputs
                    all_outputs = tf.concat([output_fw, output_bw], 2)
                    self.h_pool_flat_pos = self.last_relevant(all_outputs, sequence_length)

            ####################################################################################
            ####################################################################################

            self.h_pool2_entity_hand = tf.concat([ self.input_bag_of_entity, self.h_pool_flat_entity], 1)

            with tf.name_scope("NN_entity_hand"):
                with tf.variable_scope('FCNN_entity_hand') as scope:
                    W = tf.get_variable(
                        "W_entity_hand",
                        shape=[self.h_pool2_entity_hand.get_shape()[1], 100],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[100]), name="b_entity_hand")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.dense_entity_hand = tf.nn.xw_plus_b(self.h_pool2_entity_hand, W, b, name="scores_entity_hand")

            with tf.name_scope("dropout_entity_hand"):
                self.h_drop_entity_BOE = tf.nn.dropout(self.dense_entity_hand, self.dropout_keep_prob)


            #normalize the tensor to a unite vector
            # self.h_drop_entity_hand  = tf.truediv(self.h_drop_entity_hand, tf.sqrt(tf.reduce_sum(tf.square(self.h_drop_entity_hand), axis=1, keepdims=True)))
            ####################################################################################
            ####################################################################################

            self.h_pool_utt_cnn_flat = tf.concat([self.h_pool_utt_cnn, self.input_x_hand, self.h_pool_flat_char, self.h_pool_flat_pos], 1)

            with tf.name_scope("NN_h_pool_flat"):
                with tf.variable_scope('FCNN_h_pool_flat') as scope:
                    W = tf.get_variable(
                        "W_h_pool_flat",
                        shape=[self.h_pool_utt_cnn_flat.get_shape()[1], 100],
                        initializer=tf.contrib.layers.xavier_initializer())
                    b = tf.Variable(tf.constant(0.1, shape=[100]), name="b_h_pool_flat")
                    l2_loss += tf.nn.l2_loss(W)
                    l2_loss += tf.nn.l2_loss(b)
                    self.dense_utt_cnn = tf.nn.xw_plus_b(self.h_pool_utt_cnn_flat, W, b, name="scores_h_pool_flat")

            with tf.name_scope("dropout_h_pool_flat"):
                self.h_drop_cnn_dense_textual = tf.nn.dropout(self.dense_utt_cnn, self.dropout_keep_prob)

            ####################################################################################
            ####################################################################################

            # self.dot_product = tf.reduce_sum( tf.multiply( self.h_drop_entity_BOE, self.h_drop_cnn_dense_textual ), 1, keepdims=True )
            self.cosine_distance = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(self.h_drop_entity_BOE, 0), tf.nn.l2_normalize(self.h_drop_cnn_dense_textual, 0)),  1, keepdims=True )


            ####################################################################################
            ####################################################################################

            # Concatination of both features to a Tensor
            self.h_pool2_flat_concat = tf.concat([self.h_drop_cnn_dense_textual, self.cosine_distance, self.h_drop_entity_BOE], 1)

            # Adding handcrafted features to a Tensor
            self.h_pool2_flat = tf.concat([self.h_pool2_flat_concat], 1)

            ####################################################################################
            ####################################################################################

            with tf.variable_scope('FCNN') as scope:


                W = tf.get_variable(
                    "W",
                    shape=[self.h_pool2_flat.get_shape()[1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.dense = tf.nn.xw_plus_b(self.h_pool2_flat, W, b, name="scores")

            ####################################################################################
            ####################################################################################

            # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.dense, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[self.dense.get_shape()[1], num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                self.true_labels = tf.argmax(self.input_y, 1)
                correct_predictions = tf.equal(self.predictions, self.true_labels)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    # @staticmethod
    def _get_cell(self,hidden_size, cell_type):
        if cell_type == "vanilla":
            return tf.nn.rnn_cell.BasicRNNCell(hidden_size)
        elif cell_type == "lstm":
            return tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
        elif cell_type == "gru":
            return tf.nn.rnn_cell.GRUCell(hidden_size)
        else:
            print("ERROR: '" + cell_type + "' is a wrong cell type !!!")
            return None

    # Length of the sequence data
    # @staticmethod
    def _length(self, seq):
        relevant = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(relevant, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    # @staticmethod
    def last_relevant(self, seq, length):
        batch_size = tf.shape(seq)[0]
        max_length = int(seq.get_shape()[1])
        input_size = int(seq.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (length - 1)
        flat = tf.reshape(seq, [-1, input_size])
        return tf.gather(flat, index)