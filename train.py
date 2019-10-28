import tensorflow as tf
import numpy as np
import os, warnings, sys
import time , collections
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import cPickle, time, json
from embedding import Word2Vec
from sklearn.metrics import classification_report as cr

# Parameters
# ==================================================
#ss

root = os.path.dirname(os.getcwd())

root = root + '/ConCET'

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.5, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("Training_Data", root + "/datasets/Spotlight/final_version/self_dialogue/train_self_dialogue_adan_spotlight.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("Test_data_entity", root + "/datasets/Spotlight/final_version/self_dialogue/test_self_dialogue_adan_spotlight.txt", "Contains utts + entities")


# Model Hyperparameters
# "/Users/aliahmadvand/Desktop/HowTo/Semantic-Clustering/GoogleNews-vectors-negative300.bin"
tf.flags.DEFINE_string("word2vec", "/Users/aliahmadvand/Desktop/HowTo/Semantic-Clustering/GoogleNews-vectors-negative300.bin", "Word2vec file with pre-trained embeddings (default: None)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding")
tf.flags.DEFINE_integer("embedding_entity_dim", 32, "Dimensionality of entity embedding (default: 16)")
tf.flags.DEFINE_integer("embedding_pos_dim", 16, "Dimensionality of pos embedding (default: 16)")
tf.flags.DEFINE_integer("embedding_char_dim", 16, "Dimensionality of entity embedding (default: 16)")
tf.flags.DEFINE_integer("num_quantized_chars", 40, "num_quantized_chars")
tf.flags.DEFINE_string("filter_sizes", "2,3,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 25, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 10, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, x_char, y_text, handcraft, bag_of_entity, vocab, w2v, data_pos, data_entity, pos_vocab, entity_vocab, train_test_dev, class_label = data_helpers.load_data_and_labels(FLAGS.Training_Data, FLAGS.Test_data_entity)

# Build vocabulary
# max_document_length = max([len(x.split(" ")) for x in x_text])

max_document_length = 39
char_length = 50

vocab_processor = np.zeros([len(x_text), max_document_length+1])
x = data_helpers.fit_transform(x_text, vocab_processor, vocab)

# Build vocabulary
vocab_processor_pos = np.zeros([len(data_pos), max_document_length+1])
x_pos = data_helpers.fit_transform_pos(data_pos, vocab_processor_pos)


# Build vocabulary
vocab_processor_entity = np.zeros([len(data_entity), max_document_length+1])
x_entity = data_helpers.fit_transform_pos(data_entity, vocab_processor_entity)


x_shuf, x_char_shuf, y_shuf, handcraft_shuf, bag_of_entity_shuf, x_pos_shuf, x_entity_shuf = x, x_char, y_text, handcraft, bag_of_entity, x_pos, x_entity

offset = int(x_shuf.shape[0] * 0)
x_shuffled, x_char_shuffled, y_shuffled, handcraft_shuffled, x_pos_shuffled, x_entity_shuffled, bag_of_entity_shuffled = x_shuf[offset:], x_char_shuf[offset:], y_shuf[offset:], handcraft_shuf[offset:], x_pos_shuf[offset:], x_entity_shuf[offset:], bag_of_entity_shuf[offset:]

# # Split train/test set
# dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_shuffled)))
dev_sample_index = (-1*( len(x) - train_test_dev))
print dev_sample_index
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
x_char_train, x_char_dev = x_char_shuffled[:dev_sample_index], x_char_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
handcraft_train, handcraft_dev = np.array(handcraft_shuffled[:dev_sample_index]), np.array(handcraft_shuffled[dev_sample_index:])
bag_of_entity_train, bag_of_entity_dev = np.array(bag_of_entity_shuffled[:dev_sample_index]), np.array(bag_of_entity_shuffled[dev_sample_index:])
x_pos_train, x_pos_dev = x_pos_shuffled[:dev_sample_index], x_pos_shuffled[dev_sample_index:]
x_entity_train, x_entity_dev = x_entity_shuffled[:dev_sample_index], x_entity_shuffled[dev_sample_index:]


print("Vocabulary Size: {:d}".format(len(vocab)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=17,
            num_quantized_chars=FLAGS.num_quantized_chars,
            sequence_char_length=char_length,
            vocab_size=len(vocab),
            vocab_entity_size=len(entity_vocab)+1,
            vocab_pos_size= len(pos_vocab) + 1,
            embedding_size=FLAGS.embedding_dim,
            embedding_entity_size=FLAGS.embedding_entity_dim,
            embedding_pos_size = FLAGS.embedding_pos_dim,
            embedding_char_size=FLAGS.embedding_char_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_data = dict()
        vocab_data['data'] = vocab
        vocab_data['max_doc_len'] = max_document_length+1
        vocab_path =  os.path.join(out_dir, "vocab.json")
        handel = open(vocab_path, 'wb')
        json.dump(vocab_data,handel)
        handel.close()



        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        count = 0
        vocabulary = dict()
        if FLAGS.word2vec:
            # w2v.load_model('Word2Vector.model')
            print "\n====>len Vocab after all these {}".format(len(vocab))
            initW = np.random.uniform(-0.25, 0.25, (len(vocab), FLAGS.embedding_dim))

            for word in vocab:
                # start = time.time()
                initW[vocab[word]] = w2v.word_vectors[word]

            # sess.run(cnn.set_W, feed_dict={cnn.place_w: initW})
            sess.run(cnn.W.assign(initW))

        def train_step(x_batch, handcraft_batch, bag_of_entity_batch, y_batch, x_pos_batch, x_entity_batch, x_char_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_char: x_char_batch,
                cnn.input_x_pos: x_pos_batch,
                cnn.input_x_entity: x_entity_batch,
                cnn.input_x_hand: handcraft_batch,
                cnn.input_bag_of_entity: bag_of_entity_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            # train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, hand_batch, bag_of_entity_dev, y_batch, x_pos_batch, x_entity_batch, x_char_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_char: x_char_batch,
                cnn.input_x_pos: x_pos_batch,
                cnn.input_x_entity: x_entity_batch,
                cnn.input_x_hand: hand_batch,
                cnn.input_bag_of_entity: bag_of_entity_dev,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, predictions, true_labels, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.predictions, cnn.true_labels, cnn.accuracy], feed_dict)
            compute_prf(predictions, true_labels, class_label)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                a = 0
                # writer.add_summary(summaries, step)

            return accuracy


        def compute_prf(predictions, true_labels, class_label):

            # file_pred = open('wrong_pred.txt', 'w')
            reverse_label = {}
            for key in class_label:
                reverse_label[class_label[key]] = key

            new_predictions = []
            new_true_labels = []
            try:
                for i in range(len(predictions)):
                    new_predictions.append(reverse_label[predictions[i]])
                    new_true_labels.append(reverse_label[true_labels[i]])
            except:
                pass

            utts = x_text[dev_sample_index:]
            #
            # for i, text in enumerate(utts):
            #     new_label = posttopicmerging(text)
            #     if len(new_label) > 0:
            #         new_predictions[i] = new_label

            # for i, text in enumerate(utts):
            #     if new_predictions[i] != new_true_labels[i]:
            #         file_pred.write(text + '\t' + new_predictions[i] + '\t' + new_true_labels[i])
            #         file_pred.write('\n')

            print cr(new_true_labels, new_predictions, digits=3)


        # Generate batches
        batches = data_helpers.batch_iter(list(zip(x_train, handcraft_train, bag_of_entity_train, y_train, x_pos_train, x_entity_train, x_char_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        start = time.time()
        all_acc = []
        for batch in batches:
            x_batch, hand_batch, bag_of_entity_batch, y_batch, x_pos_batch, x_entity_batch, x_char_batch = zip(*batch)

            # try:
            train_step(x_batch, hand_batch, bag_of_entity_batch, y_batch, x_pos_batch, x_entity_batch, x_char_batch)
            # except:
            #     h =0
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                accuracy = dev_step(x_dev, handcraft_dev, bag_of_entity_dev, y_dev, x_pos_dev, x_entity_dev, x_char_dev, writer=dev_summary_writer)
                all_acc.append((accuracy))
                print "The best so far:  ", max(all_acc)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
                print time.time() - start
