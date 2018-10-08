import tensorflow as tf
#from tensorflow.contrib import rnn
import numpy as np
import random
import json
import sys
import argparse
# Config
word2idx= json.load(open("./word_table.json", 'r'))
idx2word = dict((k, v) for v, k in word2idx.iteritems())
model_dir = "./model/"
model_path = model_dir + "model.ckpt"
training_data = "./train_val_data.json"
testing_data = "./testing_data.json"
valid_data = "./testing_data.json"
feature_dir = "/home_local/alex82528/video_caption/msrvtt/train_features/"
feature_test_dir = "/home_local/alex82528/video_caption/msrvtt/test_features/"
audio_feature_dir = "/home_local/alex82528/video_caption/msrvtt/train_features/"
audio_test_feature_dir = "/home_local/alex82528/video_caption/msrvtt/test_features/"
output_path = "./prediction.json"
nclass = len(idx2word)
number_of_frames = 80
#max_sequence_length = 9 + 1
max_sequence_length = number_of_frames
batch_size = 100
rnn_dim = 256
learning_rate = 0.001
image_dim = 4096
word_dim = rnn_dim
acoustic_dim = 300
save_per_epoch = 100
nepoch = 60000
decay_per_epoch = 0.5*save_per_epoch
decay_prob = 0.1

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--train', type=bool, required=True)
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--check_overfitting', type=bool, default=False)
parser.add_argument('--keep_training', type=bool, default=False)
args = parser.parse_args()
tf.app.flags.DEFINE_boolean('train', args.train, 'if false, load the pretrained model and test')
tf.app.flags.DEFINE_boolean('attention', args.attention, 'if false, load the pretrained model and test')
tf.app.flags.DEFINE_boolean('check_overfitting', args.check_overfitting, 'If True, load a batch of training data to check the model is overfitted or not')
tf.app.flags.DEFINE_boolean('keep_training', args.keep_training, 'If True, load pretrained model and keep training. Or start training with fresh variables')
FLAGS = tf.app.flags.FLAGS
# End of config
# Define model inputs

#sess = tf.InteractiveSession()

#def get_attention(state1_list, state2):
#	similarity=[]
#	for a_state1 in state1_list :
#	# normalize in L2 method and compute cosine_similarity with subtraction
#		distance=tf.losses.cosine_distance(tf.nn.l2_normalize(a_state1,dim=0),tf.nn.l2_normalize(state2,dim=0),dim=0)
#		a_similarity = 1-distance
#		similarity.append(a_similarity)
#	return similarity

with tf.variable_scope(tf.get_variable_scope()) as scope :

	sess = tf.InteractiveSession()
	image_feature = tf.placeholder(tf.float32, [None, number_of_frames, image_dim])
	audio_feature = tf.placeholder(tf.float32, [None, number_of_frames, acoustic_dim])
	groundtruth = [tf.placeholder(tf.float32, [None, nclass]) for _ in xrange(max_sequence_length)]
	groundtruth_weights = tf.placeholder(tf.float32, [None, max_sequence_length])
	schedule_sample = tf.placeholder(tf.bool, [None, max_sequence_length])
	# Linear transform image features to rnn_dim features
	encode_image_W = tf.Variable(tf.random_uniform([image_dim, rnn_dim], -0.1, 0.1))
	encode_image_b = tf.Variable(tf.zeros([rnn_dim]))
	input_image = tf.nn.xw_plus_b(tf.reshape(image_feature, [-1, image_dim]), encode_image_W, encode_image_b)
	input_image_before_trans = tf.reshape(input_image, [-1, number_of_frames, rnn_dim])
	input_image = tf.transpose(input_image_before_trans, [1, 0, 2])
	# Linear transform audio features to rnn_dim features
	encode_audio_W = tf.Variable(tf.random_uniform([acoustic_dim, rnn_dim], -0.1, 0.1))
	encode_audio_b = tf.Variable(tf.zeros([rnn_dim]))
	input_audio = tf.nn.xw_plus_b(tf.reshape(audio_feature, [-1, acoustic_dim]), encode_audio_W, encode_audio_b)
	input_audio_before_trans = tf.reshape(input_audio, [-1, number_of_frames, rnn_dim])
	input_audio = tf.transpose(input_audio_before_trans, [1, 0, 2])
        # Define RNN cells
	LSTMCell1 = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_dim) # for image frame
	LSTMCell2 = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_dim)
	LSTMCell3 = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(rnn_dim) # for audio frame
	state_zeros = tf.matmul(tf.zeros_like(groundtruth[0]), tf.zeros([nclass, rnn_dim])) # [?, rnn_dim]
	state1 = tuple([tf.zeros_like(state_zeros), tf.zeros_like(state_zeros)])
	state2 = tuple([tf.zeros_like(state_zeros), tf.zeros_like(state_zeros)])
        state3 = tuple([tf.zeros_like(state_zeros), tf.zeros_like(state_zeros)])
        # Encoding stage
	#word_padding = tf.matmul(tf.zeros_like(state_zeros), tf.zeros([rnn_dim, word_dim])) # [?, word_dim]
	word_padding = tf.zeros_like(state_zeros)


	#state1_collector=[]
	for i in xrange(number_of_frames):
		if i > 0:
			tf.get_variable_scope().reuse_variables()
		with tf.variable_scope("LSTM1"):
			output1, state1 = LSTMCell1(input_image[i], state1)
		#state1_collector.append(state1[1]) #enable if attention are utilized
		with tf.variable_scope("LSTM3"):
			output3, state3 = LSTMCell3(input_image[i], state3)
		with tf.variable_scope("LSTM2"):
			output2, state2 = LSTMCell2(tf.concat(axis=1, values=[word_padding, output1,output3]), state2)
	# Decoding stage
	prediction = []
	sample = tf.transpose(schedule_sample, [1, 0])
	word_transform = tf.Variable(tf.random_uniform([nclass, word_dim], -0.1, 0.1))
	W_output = tf.Variable(tf.random_uniform([rnn_dim, nclass], -0.1, 0.1))
	b_output = tf.Variable(tf.zeros([nclass]))
	loss = 0
	for i in xrange(max_sequence_length):
		if i == 0:
			current_embedding = tf.zeros_like(word_padding)
			LSTM1_input = tf.zeros_like(input_image[0])
		else:
			# Always feed the correct answer
			if FLAGS.train == True:
				current_embedding = tf.matmul(tf.where(sample[i], groundtruth[i-1], tf.one_hot(tf.argmax(logit, 1), nclass)), word_transform)
		#First arg:<tf.where(condition,x,y)>: if condition==True,take x. if condition=False ,take y.
			else:
				current_embedding = tf.matmul(tf.one_hot(tf.argmax(logit, 1), nclass), word_transform)
	#	# TODO
	#	# attention for lstm1 input
	#	if FLAGS.attention== True:
	#	#attention = get_attention(state1_collector,state2)
	#		#sate1_collector=tf.transpose(tf.reshape(state1_collector,[-1,rnn_dim]))
	#		similarity=[]
	#		accumulate = tf.zeros_like(state2[1])
	#		nor_state2 = tf.nn.l2_normalize(state2[1],1) # (?,256)
	#		for a_state in state1_collector:
	#			nor_a_state1 = tf.nn.l2_normalize(a_state,1) #(?,256)
	#			tmp=tf.multiply(nor_state2,nor_a_state1) # (?,256)
	#			red_tmp=tf.reduce_sum(tmp,1) # (?,)
	#			similarity.append(red_tmp) # 80,  (?,)
	#		#accumulate=tf.reduce_sum(accumulate,1) # (?,)
	#		weighted_input = tf.zeros_like(input_image[0])
	#		for z, time_step2 in enumerate(similarity):
	#			a_expand = tf.matmul(tf.reshape(time_step2,[-1,1]),tf.ones([1,256])) # (?,256)
	#			td_weighted_input = tf.multiply(a_expand , input_image[z])
	#			weighted_input += td_weighted_input
	#		LSTM1_input = weighted_input
        #        #LSTM1_input = tf.zeros_like(input_image[0])
	#	else:
		LSTM1_input = tf.zeros_like(input_image[0])
		LSTM3_input = tf.zeros_like(input_audio[0])

		tf.get_variable_scope().reuse_variables()
		with tf.variable_scope("LSTM1"):
			output1, state1 = LSTMCell1(LSTM1_input, state1)
		with tf.variable_scope("LSTM3"):
			output3, state3 = LSTMCell1(LSTM3_input, state3)
                with tf.variable_scope("LSTM2"):
			output2, state2 = LSTMCell2(tf.concat(axis=1, values=[current_embedding, output1,output3]), state2)

		logit = tf.nn.xw_plus_b(output2, W_output, b_output)
		prediction.append(logit)
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=groundtruth[i])
		#cross_entropy = tf.reshape(cross_entropy, [-1, 1])

		#if i == 0: loss = cross_entropy
		#else: loss = tf.concat(1, [loss, cross_entropy])
		loss += tf.reduce_sum(cross_entropy*groundtruth_weights[:, i])

	loss = loss/tf.reduce_sum(groundtruth_weights)
#loss = tf.reduce_mean(tf.reduce_mean(tf.mul(loss, groundtruth_weights), 1), 0)
# Clip gradient
#optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
#optimizer = tf.train.AdamOptimizer(learning_rate)
#gvs = optimizer.compute_gradients(loss)
#capped_gvs = [(tf.clip_by_value(grad, -0.5, 0.5), var) for grad, var in gvs]
#train_step = optimizer.apply_gradients(capped_gvs)
#with tf.variable_scope(tf.get_variable_scope()) as scope :
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
# End of model definition

# Read input
def preprocess_data(data,feat_dir):
	feature = []
	captions = []
	caption_weights = []

	for item in data:
		# Image feature
		image = np.load(feat_dir + item["id"] + ".vgg.npy")
		feature.append(image)
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
	return feature, captions, caption_weights

def get_audio_frame(data,feat_dir):
        feature = []
        for item in data :
            # acoustic feature
            acoustic = np.load(feat_dir+item["id"]+".audio.npy")
            feature.append(acoustic)
        return feature

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
valid_data = json.load(open(valid_data, 'r'))

# =================================================================================
if FLAGS.train:
	# Train
	if FLAGS.keep_training == False: tf.global_variables_initializer().run()
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
                # image feature preprocessing
		input_feature, input_captions, input_caption_weight = preprocess_data(data_batch,feature_dir)
		input_feature = np.stack(input_feature)
		input_captions = transform_to_list(len(input_feature), input_captions)
                # audio feature preprocessing
                audio_input = get_audio_frame(data_batch,audio_feature_dir)
                audio_input = np.stack(audio_input)
                feed_dict = {image_feature: input_feature, audio_feature : audio_input, groundtruth_weights: input_caption_weight, schedule_sample: sampling}
		#feed_dict = {image_feature: input_feature, groundtruth_weights: input_caption_weight}
		feed_dict.update({groundtruth[t]: input_captions[t] for t in xrange(max_sequence_length)})
		sys.stdout.flush()
		train_step.run(feed_dict)

		step += len(data)
		if step == len(data):
			step = 0
			epoch += 1
			training_loss = loss.eval(feed_dict)
			# Test on validation data
			valid_feature, valid_captions, valid_caption_weights = preprocess_data(valid_data,feature_test_dir)
			valid_captions = transform_to_list(len(valid_feature), valid_captions)
                        valid_audio = get_audio_frame(valid_data,audio_test_feature_dir)
			sampling = schedule_sampling(len(valid_feature), max_sequence_length, 1) #last arg=1 ,no schedule sampling
                        feed_dict = {image_feature: valid_feature,audio_feature:valid_audio, groundtruth_weights: valid_caption_weights, schedule_sample: sampling}
			feed_dict.update({groundtruth[t]: valid_captions[t] for t in xrange(max_sequence_length)})
			valid_loss = loss.eval(feed_dict)

			print("End of epoch " + str(epoch) + " : training_loss = " + str(training_loss) ) + ", valid_loss = " + str(valid_loss)
			sys.stdout.flush()
			# Shuffle data
			random.shuffle(data)

			# Save model
			#if epoch % save_per_epoch == 0:
			if epoch % 6000 == 0:
				tf.train.Saver(max_to_keep=10).save(sess, model_path, global_step=epoch)
				print("Model save in file: " + model_path + "-" + str(epoch))
				sys.stdout.flush()

else:
# =================================================================================
	# Test
	# Restore model
	tf.train.Saver().restore(sess, tf.train.latest_checkpoint(model_dir))
	print("Successfully load model")

	input_feature, input_captions, input_caption_weight = preprocess_data(data,feature_test_dir)
	input_captions = transform_to_list(len(input_feature), input_captions)
        audio_input = get_audio_frame(data, audio_test_feature_dir )
        feed_dict = {image_feature: input_feature,audio_feature: audio_input, groundtruth_weights: input_caption_weight}
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
