import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell, seq2seq
import numpy as np
import random
import json
import sys
import argparse

# Config
word2idx = json.load(open("./word2idx.json", 'r'))
idx2word = dict((k, v) for v, k in word2idx.iteritems())
model_dir = "./model/"
model_path = model_dir + "model.ckpt"
training_data = "./train_val_data.json"
testing_data = "./testing_data.json"
output_path = "./prediction.json"
training_feature_dir = "./train_features/"
testing_feature_dir = "./test_features/"
nclass = len(idx2word)
number_of_frames = 80
#max_sequence_length = 9 + 1 # 1 for _PAD original(36 + 1)
max_sequence_length = number_of_frames
rnn_dim = 512
learning_rate = 0.001
momentum = 0.5
batch_size = 100
nepoch = 200
image_dim = 4096
acoustic_dim = 300
word_dim = 300
acoustic_dim = 300
save_per_epoch = 100
decay_per_epoch = 0.5*save_per_epoch
decay_prob = 0.1

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, required=True)
parser.add_argument("--check_overfitting", type=bool, default=False)
parser.add_argument("--keep_training", type=bool, default=False)
args = parser.parse_args()
tf.app.flags.DEFINE_boolean('train', args.train, 'If False, load the pretrained model and test')
tf.app.flags.DEFINE_boolean('check_overfitting', args.check_overfitting, 'If True, load a batch of training data to check the model is overfitted or not')
tf.app.flags.DEFINE_boolean('keep_training', args.keep_training, 'If True, load pretrained model and keep training. Or start training with fresh variables')
FLAGS = tf.app.flags.FLAGS
# End of config

# Define model inputs
sess = tf.InteractiveSession()
image_feature = tf.placeholder(tf.float32, [None, number_of_frames, image_dim])
acoustic_feature = tf.placeholder(tf.float32, [None, number_of_frames, acoustic_dim])
groundtruth = [tf.placeholder(tf.float32, [None, nclass]) for _ in xrange(max_sequence_length)]
decoder_sequence_weights = tf.placeholder(tf.float32, [None, max_sequence_length])
schedule_sample = tf.placeholder(tf.bool, [None, max_sequence_length])

# Linearly transform image features to rnn_dim features
encode_W = tf.Variable(tf.random_uniform([image_dim+acoustic_dim, rnn_dim], -0.1, 0.1))
encode_b = tf.Variable(tf.zeros([rnn_dim]))
input_sequence = tf.concat(2, [image_feature, acoustic_feature])
input_sequence = tf.matmul(tf.reshape(input_sequence, [-1, image_dim+acoustic_dim]), encode_W) + encode_b
input_sequence = tf.reshape(input_sequence, [-1, number_of_frames, rnn_dim])
input_sequence = tf.transpose(input_sequence, [1, 0, 2])

# Bidirectional encoder of image frames
with tf.variable_scope("BiLSTM_encoder"):
    input_sequence = tf.split(0, number_of_frames, tf.reshape(input_sequence, [-1, rnn_dim]))
    encoder_cell_image_forward = rnn_cell.BasicLSTMCell(rnn_dim)
    encoder_cell_image_backward = rnn_cell.BasicLSTMCell(rnn_dim)
    biLSTM_encoder_outputs_image, encoder_state_image_forward, encoder_state_image_backward = rnn.bidirectional_rnn(encoder_cell_image_forward, encoder_cell_image_backward, input_sequence, dtype=tf.float32)
    # biLSTM_encoder_outputs_image [[batch_size, 2*rnn_dim], ..., ]

# Sequence to sequence : encoder
with tf.variable_scope("seq2seq_encoder"):
    seq2seq_encoder_cell = rnn_cell.BasicLSTMCell(rnn_dim)
    seq2seq_encoder_outputs, seq2seq_encoder_state = rnn.rnn(seq2seq_encoder_cell, biLSTM_encoder_outputs_image, dtype=tf.float32)

# Sequence to sequence : decoder
with tf.variable_scope("seq2seq_decoder"):
    seq2seq_decoder_cell = rnn_cell.BasicLSTMCell(rnn_dim)
    W_output = tf.Variable(tf.random_uniform([rnn_dim, nclass], -0.1, 0.1))
    b_output = tf.Variable(tf.random_uniform([nclass], -0.1, 0.1))
    word_transform = tf.Variable(tf.random_normal([nclass, word_dim]))
    state = seq2seq_encoder_state
    sample = tf.transpose(schedule_sample, [1, 0])
    prediction = []
    loss = 0
    for i in xrange(0, max_sequence_length):
        if i > 0: tf.get_variable_scope().reuse_variables()

        if i == 0: current_embedding = tf.matmul(tf.zeros_like(groundtruth[0]), word_transform)
        else:
            if FLAGS.train == True:
                current_embedding = tf.matmul(tf.select(sample[i], groundtruth[i-1], tf.one_hot(tf.argmax(logit, 1), nclass)), word_transform)
            else: current_embedding = tf.matmul(tf.one_hot(tf.argmax(logit, 1), nclass), word_transform)

        decoder_output, state = seq2seq_decoder_cell(current_embedding, state)

        logit = tf.matmul(decoder_output, W_output) + b_output
        prediction.append(logit)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit, groundtruth[i])
        #cross_entropy = tf.reshape(cross_entropy, [-1, 1])

        #if i == 0: loss = cross_entropy
        #else: loss = tf.concat(1, [loss, cross_entropy])
        loss += tf.reduce_sum(cross_entropy*decoder_sequence_weights[:, i])

loss = loss/tf.reduce_sum(decoder_sequence_weights)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
#train_step = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(loss)
#train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# End of model definition

# Read input
def preprocess_data(data, feature_dir):
    image_feature = []
    acoustic_feature = []
    captions = []
    caption_weights = []

    for item in data:
        # Image feature
        image = np.load(feature_dir + item["id"] + ".vgg.npy")
        image_feature.append(image)
        # Acoustic feature
        acoustic = np.load(feature_dir + item["id"] + ".audio.npy")
        acoustic_feature.append(acoustic)
        # Convert sentence to 1-hot
        temp = np.zeros((max_sequence_length, nclass))
        # Random one caption from captions
        caption = item["caption"][int(len(item["caption"])*random.random())]
        caption = caption.replace(',', '').replace('.', '').lower().split()
        for i, word in enumerate(caption):
            if (word in word2idx) and (i<max_sequence_length-1): temp[i][int(word2idx[word])] = 1
            elif (not (word in word2idx)) and (i<max_sequence_length-1): temp[i][int(word2idx["_UNK"])] = 1
        # Add "_PAD" at the end of the sentence
        if len(caption) > max_sequence_length-1: temp[max_sequence_length-1][0] = 1
        else: temp[len(caption)][0] = 1
        captions.append(temp)
        # Record the length
        temp = np.zeros((max_sequence_length))
        for i in xrange(0, len(caption)):
            if i < max_sequence_length-1: temp[i] = 1
        if len(caption) > max_sequence_length-1: temp[max_sequence_length-1] = 1
        else: temp[len(caption)] = 1
        caption_weights.append(temp)
    return image_feature, acoustic_feature, captions, caption_weights

def transform_to_list(batch_size, vector_y):
    # Transform vector y from [?, max_sequence_length, nclass] to [max_sequence_length, ?, nclass]
    input_y = []
    for i in xrange(0, max_sequence_length):
        y = np.zeros((batch_size, nclass))
        for j in xrange(0, len(vector_y)):
            y[j] = vector_y[j][i]
        input_y.append(y)
    return input_y

def schedule_sampling(batch_size, max_sequence_length, prob):
    sampling = []
    for i in xrange(batch_size):
        temp = []
        for j in xrange(max_sequence_length):
            if random.random() < prob: temp.append(True)
            else: temp.append(False)
        sampling.append(temp)
    return sampling

if FLAGS.train:
    data = json.load(open(training_data, 'r'))
elif FLAGS.train == False and FLAGS.check_overfitting == False:
    data = json.load(open(testing_data, 'r'))
elif FLAGS.train == False and FLAGS.check_overfitting == True:
    data = json.load(open(training_data, 'r'))
    data = data[:batch_size]
# Read validation data
#valid_data = json.load(open(valid_data, 'r'))

# =================================================================================
if FLAGS.train:
    # Train
    if FLAGS.keep_training == False: tf.initialize_all_variables().run()
    else:
        tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
        print("Successfully load model")

    epoch = step = 0
    prob = 1
    #while 1:
    while epoch < nepoch:
        # Preprocess data
        if (step + batch_size) < len(data):
            data_batch = data[step: step+batch_size]
        else:
            data_batch = data[step:]

        # Schedule sampling
        if epoch % decay_per_epoch == 0: prob -= decay_prob
        sampling = schedule_sampling(len(data_batch), max_sequence_length, 1) # Setting the last argument be 1 will always feed ground truths

        input_image_feature, input_acoustic_feature, input_captions, input_caption_weight = preprocess_data(data_batch, training_feature_dir)
        input_captions = transform_to_list(len(input_image_feature), input_captions)

        feed_dict = {image_feature: input_image_feature, acoustic_feature: input_acoustic_feature, decoder_sequence_weights: input_caption_weight, schedule_sample: sampling}
        feed_dict.update({groundtruth[t]: input_captions[t] for t in xrange(max_sequence_length)})
        train_step.run(feed_dict)

        step += len(data_batch)
        print("Running step " + str(step) + " / " + str(len(data)))
        sys.stdout.flush()
        if step == len(data):
            step = 0
            epoch += 1
            training_loss = loss.eval(feed_dict)
            # Test on validation data
            #valid_feature, valid_captions, valid_caption_weights = preprocess_data(valid_data)
            #valid_captions = transform_to_list(len(valid_feature), valid_captions)
            #sampling = schedule_sampling(len(valid_feature), max_sequence_length, 0)
            #feed_dict = {image_feature: valid_feature, decoder_sequence_weights: valid_caption_weights, schedule_sample: sampling}
            #feed_dict.update({groundtruth[t]: valid_captions[t] for t in xrange(max_sequence_length)})
            #valid_loss = loss.eval(feed_dict)

            #print("End of epoch " + str(epoch) + " : training_loss = " + str(training_loss) + ", valid_loss = " + str(valid_loss))
            print("End of epoch " + str(epoch) + " : training_loss = " + str(training_loss))
            sys.stdout.flush()
            # Shuffle data
            random.shuffle(data)

            # Save model
            if epoch % save_per_epoch == 0:
                tf.train.Saver().save(sess, model_path, global_step=epoch)
                print("Model save in file: " + model_path + "-" + str(epoch))
                sys.stdout.flush()

else:
# =================================================================================
    # Test
    # Restore model
    tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
    print("Successfully load model")

    if FLAGS.check_overfitting == True: testing_feature_dir = training_feature_dir
    input_image_feature, input_acoustic_feature, input_captions, input_caption_weight = preprocess_data(data, testing_feature_dir)
    input_captions = transform_to_list(len(input_image_feature), input_captions)

    feed_dict = {image_feature: input_image_feature, acoustic_feature: input_acoustic_feature, decoder_sequence_weights: input_caption_weight}
    feed_dict.update({groundtruth[t]: input_captions[t] for t in xrange(max_sequence_length)})
    prediction = sess.run(prediction, feed_dict)

    # Transform prediction
    output = []
    for i in xrange(0, len(data)):
        temp = np.zeros((max_sequence_length, nclass))
        for j in xrange(0, len(prediction)):
            temp[j] = prediction[j][i]
        output.append(temp)

    output_table = []
    for i, sentence in enumerate(output):
        string_predict = ""
        string_groundtruth = ""
        for j, word in enumerate(sentence):
            string_predict += idx2word[np.argmax(word)] + ' '
            string_groundtruth += idx2word[np.argmax(input_captions[j][i])] + ' '
        string_predict = string_predict.replace("_PAD", '').rstrip()
        string_groundtruth = string_groundtruth.replace("_PAD", '').rstrip()
        print(string_predict)
        output_table.append({"id": data[i]["id"], "prediction": string_predict, "groundtruth": string_groundtruth})
    json.dump(output_table, open(output_path, 'w'), indent=4)
    testing_loss = loss.eval(feed_dict)
    print("Output saved in file: " + output_path)
    print("Testing loss = " + str(testing_loss))
