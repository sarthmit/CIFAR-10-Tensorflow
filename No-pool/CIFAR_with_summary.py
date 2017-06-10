import tensorflow as tf
import load_data
import numpy as np
import os

current_path = os.path.dirname(os.path.realpath(__file__))
log_path = current_path+"/logs"
data_path = current_path+"/cifar-10-batches-py"

if tf.gfile.Exists(log_path+'/train'):
	tf.gfile.DeleteRecursively(log_path+'/train')
	tf.gfile.DeleteRecursively(log_path+'/test')
	tf.gfile.DeleteRecursively(log_path+'/projector')
tf.gfile.MkDir(log_path + '/projector')
tf.gfile.MkDir(log_path + '/train')
tf.gfile.MkDir(log_path + '/test')

test_dict = load_data.unpickle("cifar-10-batches-py/test_batch")
test_dict["data"] = load_data.GCN(test_dict["data"])
test_dict["data"] = load_data.zca_whitening(test_dict["data"])
test_dict["labels"] = load_data.convert_one_hot(test_dict["labels"],10)

dict = load_data.create_data()
dict["data"] = load_data.GCN(dict["data"])
dict["data"] = load_data.zca_whitening(dict["data"])
dict["labels"] = load_data.convert_one_hot(dict["labels"],10)

def initializer(shape, type, wd=0.001):
	if type==0:
		return tf.Variable(tf.fill(shape,0.1))
	elif type==1:
		dev = np.sqrt(shape[0])
	else:
		dev = np.sqrt(shape[0]*shape[1]*shape[2])
	initial = tf.Variable(tf.random_normal(shape, stddev=0.05))
	tf.add_to_collection("losses",tf.multiply(tf.nn.l2_loss(initial),wd))
	return initial

def conv2d(x,W,padding):
	return tf.nn.conv2d(x , W , [1,1,1,1] , padding=padding)
def conv2dstr2(x,W,padding):
	return tf.nn.conv2d(x, W, [1,2,2,1], padding=padding)

def conv_pool_layer(x,length,width,input_channels,output_channels,layer_name,act=tf.nn.relu,padding="SAME"):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = initializer([length,width,input_channels,output_channels],2)
			if weights:
				tf.summary.histogram(layer_name+"_weights",weights)
		with tf.name_scope('biases'):
			biases = initializer([output_channels],0)
			if biases:
				tf.summary.histogram(layer_name+"_biases",biases)
		with tf.name_scope('activations'):
			activations = act(conv2dstr2(x,weights,padding) + biases)
			tf.summary.histogram(layer_name+"_activations",activations)
		return activations

def conv_layer(x,length,width,input_channels,output_channels,layer_name,act=tf.nn.relu,padding="SAME"):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = initializer([length,width,input_channels,output_channels],2)
			if weights:
				tf.summary.histogram(layer_name+"_weights",weights)
		with tf.name_scope('biases'):
			biases = initializer([output_channels],0)
			if biases:
				tf.summary.histogram(layer_name+"_biases",biases)
		with tf.name_scope('activations'):
			activations = act(conv2d(x,weights,padding) + biases)
			tf.summary.histogram(layer_name+"_activations",activations)
		return activations

def average_pool(data):
	return tf.nn.avg_pool(data,ksize=[1,8,8,1],strides=[1,8,8,1],padding="VALID")

def learning_rate():
	epoch1 = 200
	epoch2 = 250
	epoch3 = 300

	learning_decay = 0.1

	batch_per_epoch = 50000/128

	decay1 = int(batch_per_epoch*epoch1)
	decay2 = int(batch_per_epoch*epoch2)
	decay3 = int(batch_per_epoch*epoch3)

	learning_rate = 0.05

	decayed_learning_rate1 = tf.train.exponential_decay(learning_rate, global_step, decay1, learning_decay, staircase=True)
	decayed_learning_rate2 = tf.train.exponential_decay(decayed_learning_rate1, global_step, decay2, learning_decay, staircase=True)
	decayed_learning_rate3= tf.train.exponential_decay(decayed_learning_rate2, global_step, decay3, learning_decay, staircase=True)
	tf.summary.scalar("learning_rate",decayed_learning_rate3)
	return decayed_learning_rate3

def dropout(data,keep_prob):
	return tf.nn.dropout(data,keep_prob)

def loss(logits,labels):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
	tf.add_to_collection('losses',cost)

	return tf.add_n(tf.get_collection('losses'),"total_loss")

def acc(prediction,labels):
	correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
	return tf.reduce_mean(tf.cast(correct_prediction,tf.float32),0)

#with tf.device('/cpu:0'):
#	embedding = tf.Variable(tf.stack(mnist.test.images[:num_images], axis=0), trainable=False,name='embedding')

x = tf.placeholder(tf.float32, [None,3072])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
keep_prob_input = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)

x_image = tf.reshape(x,[-1,32,32,3])
tf.summary.image("Images",x_image,25)

h11 = conv_layer(x_image,3,3,3,96,"conv_11")
h12 = conv_layer(h11,3,3,96,96,"conv_12")
h13 = conv_pool_layer(h12,3,3,96,96,"conv_str2_1")

h1 = dropout(h13,keep_prob)

h21 = conv_layer(h1,3,3,96,192,"conv_21")
h22 = conv_layer(h21,3,3,192,192,"conv_22")
h23 = conv_pool_layer(h22,3,3,192,192,"conv_str2_2")

h2 = dropout(h23,keep_prob)

h31 = conv_layer(h2,3,3,192,192,"conv_31")
h32 = conv_layer(h21,1,1,192,192,"conv_32",padding="VALID")
h33 = conv_pool_layer(h22,1,1,192,10,"conv_str2_3",padding="VALID")

h_out = average_pool(h33)
h_out = tf.reshape(h_out,[-1,10])

cost = loss(h_out,y)
tf.summary.scalar("Loss", cost)

train = tf.train.MomentumOptimizer(learning_rate(),0.9).minimize(cost,global_step)
accuracy = acc(h_out,y)
tf.summary.scalar("Accuracy",accuracy)

merged_summary = tf.summary.merge_all()

test_accur = tf.placeholder(tf.float32)
summ = tf.summary.scalar("Test_Accuracy",test_accur)

saver = tf.train.Saver(max_to_keep=2)
	
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

saver.save(sess,'cifar-10')
train_writer = tf.summary.FileWriter(log_path+'/train',sess.graph)
test_writer = tf.summary.FileWriter(log_path+'/test')

def test_acc():
	sum=0.0
	for i in range(100):
		batch_x, batch_y = load_data.next_batch(test_dict,100,0,start=i*100)
		t = accuracy.eval(feed_dict={x:batch_x, y:batch_y, keep_prob:1.0,keep_prob_input:1.0})
		sum = sum + t
	return sum/100.0

for i in range(1000):
	batch_x, batch_y = load_data.next_batch(dict,128,1)
	feed = {x:batch_x, y:batch_y, keep_prob:0.5, keep_prob_input:0.8}
	if i%10==0:
		summary,training_cost,train_accuracy = sess.run([merged_summary,cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.0,keep_prob_input:1.0})
		train_writer.add_summary(summary,i)
		print("Step %d Training Accuracy: %f Cost: %f " %(i, train_accuracy, training_cost))
		saver.save(sess,current_path)
	if i%9 == 0:
		test_accuracy = test_acc()
		summary = sess.run(summ,feed_dict={test_accur:test_accuracy})
		test_writer.add_summary(summary,i)
	train.run(feed_dict=feed)

print("Test Accuracy: %f" %test_acc())