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
#test_dict["data"] = load_data.zca_whitening(test_dict["data"])
test_dict["labels"] = load_data.convert_one_hot(test_dict["labels"],10)

dict = load_data.create_data()
dict["data"] = load_data.GCN(dict["data"])
#dict["data"] = load_data.zca_whitening(dict["data"])
dict["labels"] = load_data.convert_one_hot(dict["labels"],10)

def initializer(shape, type, wd=0.0):
	if type==0:
		return tf.Variable(tf.fill(shape,0.1))
	elif type==1:
		dev = np.sqrt(shape[0])
	else:
		dev = np.sqrt(shape[0]*shape[1]*shape[2])
	initial = tf.Variable(tf.random_normal(shape, stddev=0.05))
	tf.add_to_collection("losses",tf.multiply(tf.nn.l2_loss(initial),wd))
	return initial

def conv2d(x,W):
	return tf.nn.conv2d(x , W , [1,1,1,1] , padding="SAME")
def max_pool(x):
	return tf.nn.max_pool(x, [1,3,3,1], [1,2,2,1], padding="SAME")

def normalization(x):
	return tf.nn.local_response_normalization(x,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

def dense_layer(x,input_size,output_size,layer_name,act=tf.nn.relu):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = initializer([input_size,output_size],1,0.04)
			tf.summary.histogram(layer_name+"_weights",weights)
		with tf.name_scope('biases'):
			biases = initializer([output_size],0,0.0)
			tf.summary.histogram(layer_name+"_biases",biases)
		with tf.name_scope('activations'):
			activations = act(tf.matmul(x,weights) + biases)
			tf.summary.histogram(layer_name+"_activations",activations)
		return activations

def conv_layer(x,length,width,input_channels,output_channels,layer_name,act=tf.nn.relu,padding="SAME"):
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			weights = initializer([length,width,input_channels,output_channels],2,0.0)
			if weights:
				tf.summary.histogram(layer_name+"_weights",weights)
		with tf.name_scope('biases'):
			biases = initializer([output_channels],0,0.0)
			if biases:
				tf.summary.histogram(layer_name+"_biases",biases)
		with tf.name_scope('activations'):
			activations = act(conv2d(x,weights) + biases)
			tf.summary.histogram(layer_name+"_activations",activations)
		return activations

def dropout(data,keep_prob):
	return tf.nn.dropout(data,keep_prob)

def loss(logits,labels):
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
	tf.add_to_collection('losses',cost)

	return tf.add_n(tf.get_collection('losses'),"total_loss")

def acc(prediction,labels):
	correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(labels,1))
	return tf.reduce_mean(tf.cast(correct_prediction,tf.float32),0)

x = tf.placeholder(tf.float32, [None,3072])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
keep_prob_input = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)

lr = tf.train.exponential_decay(0.1,global_step,100,0.993,staircase=True)

x_image = tf.reshape(x,[-1,32,32,3])
tf.summary.image("Images",x_image,25)

h1 = conv_layer(x_image,3,3,3,64,"conv_layer_1")
h1 = max_pool(h1)
h1 = normalization(h1)

h2 = conv_layer(h1,3,3,64,128,"conv_layer_2")
h2 = normalization(h2)
h2 = max_pool(h2)

h3 = tf.reshape(h2,[-1,8*8*128])
h3 = dropout(h3,keep_prob)

h4 = dense_layer(h3,8*8*128,1024,"dense_layer_1")
h4 = dropout(h4,keep_prob)

h5 = dense_layer(h4,1024,512,"dense_layer_2")
h5 = dropout(h5,keep_prob)

h_out = dense_layer(h5,512,10,"dense_output",tf.identity)

cost = loss(h_out,y)
tf.summary.scalar("Loss", cost)

train = tf.train.MomentumOptimizer(lr,0.9).minimize(cost,global_step)
tf.summary.scalar("Learning_Rate",lr)
accuracy = acc(h_out,y)
tf.summary.scalar("Accuracy",accuracy)

merged_summary = tf.summary.merge_all()

test_accur = tf.placeholder(tf.float32)
summ = tf.summary.scalar("Test_Accuracy",test_accur)
	
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

train_writer = tf.summary.FileWriter(log_path+'/train',sess.graph)
test_writer = tf.summary.FileWriter(log_path+'/test')

def test_acc():
	sum=0.0
	for i in range(100):
		batch_x, batch_y = load_data.next_batch(test_dict,100,0,start=i*100)
		t = accuracy.eval(feed_dict={x:batch_x, y:batch_y, keep_prob:1.0,keep_prob_input:1.0})
		sum = sum + t
	return sum/100.0

for i in range(20000):
	batch_x, batch_y = load_data.next_batch(dict,160,1)
	feed = {x:batch_x, y:batch_y, keep_prob:0.8, keep_prob_input:0.8}
	if i%10==0:
		summary,training_cost,train_accuracy = sess.run([merged_summary,cost,accuracy],feed_dict={x:batch_x,y:batch_y,keep_prob:1.0,keep_prob_input:1.0})
		train_writer.add_summary(summary,i)
		print("Step %d Training Accuracy: %f Cost: %f " %(i, train_accuracy, training_cost))
	elif i%9 == 0:
		test_accuracy = test_acc()
		summary = sess.run(summ,feed_dict={test_accur:test_accuracy})
		test_writer.add_summary(summary,i)
	sess.run(train,feed_dict=feed)

print("Test Accuracy: %f" %test_acc())