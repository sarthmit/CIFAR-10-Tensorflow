import tensorflow as tf
import load_data
import numpy as np

x = tf.placeholder(tf.float32, [None,3072])
y = tf.placeholder(tf.float32, [None,10])
keep_prob = tf.placeholder(tf.float32)
keep_prob_input = tf.placeholder(tf.float32)
global_step = tf.Variable(0, trainable=False)

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

test_dict = load_data.unpickle("cifar-10-batches-py/test_batch")
test_dict["data"] = load_data.GCN(test_dict["data"])
test_dict["data"] = load_data.zca_whitening(test_dict["data"])
test_dict["labels"] = load_data.convert_one_hot(test_dict["labels"],10)

def initializer(shape, type, wd=0.001):
	if type==0:
		dev = np.sqrt(shape[0])
	elif type==1:
		dev = np.sqrt((shape[0]+shape[1])/2.0)
	else:
		dev = np.sqrt((shape[0]*shape[1]*shape[2] + shape[3])/2.0)
	initial = tf.Variable(tf.random_normal(shape, stddev=1/dev))
	return initial

def conv2d(x,W,padding="SAME"):
	return tf.nn.conv2d(x , W , [1,1,1,1] , padding=padding)
def conv2dstr2(x,W,padding="SAME"):
	return tf.nn.conv2d(x, W, [1,2,2,1], padding=padding)

W_conv11 = initializer([3 , 3 , 3, 96] , 2)
b_conv11 = initializer([96] , 0)

W_conv12 = initializer([3 , 3 , 96 , 96] , 2)
b_conv12 = initializer([96] , 0)

W_conv13 = initializer([3 , 3 , 96 , 96] , 2)
b_conv13 = initializer([96] , 0)

W_conv21 = initializer([3 , 3 , 96, 192] , 2)
b_conv21 = initializer([192] , 0)

W_conv22 = initializer([3 , 3 , 192 , 192] , 2)
b_conv22 = initializer([192] , 0)

W_conv23 = initializer([3 , 3 , 192 , 192] , 2)
b_conv23 = initializer([192] , 0)

W_conv31 = initializer([3 , 3 , 192, 192] , 2)
b_conv31 = initializer([192] , 0)

W_conv32 = initializer([1 , 1 , 192 , 192] , 2)
b_conv32 = initializer([192] , 0)

W_conv33 = initializer([1 , 1 , 192 , 10] , 2)
b_conv33 = initializer([10] , 0)

reg_term = 0.001 * (tf.nn.l2_loss(W_conv11) + tf.nn.l2_loss(W_conv12) + tf.nn.l2_loss(W_conv13)
+ tf.nn.l2_loss(W_conv21) + tf.nn.l2_loss(W_conv22) + tf.nn.l2_loss(W_conv23)
+ tf.nn.l2_loss(W_conv31) + tf.nn.l2_loss(W_conv32) + tf.nn.l2_loss(W_conv33)
+ tf.nn.l2_loss(b_conv11) + tf.nn.l2_loss(b_conv12) + tf.nn.l2_loss(b_conv13)
+ tf.nn.l2_loss(b_conv21) + tf.nn.l2_loss(b_conv22) + tf.nn.l2_loss(b_conv23)
+ tf.nn.l2_loss(b_conv31) + tf.nn.l2_loss(b_conv32) + tf.nn.l2_loss(b_conv33))

x_image = tf.reshape(x , [-1,32,32,3])

x_image = tf.nn.dropout(x_image,keep_prob_input)

h11 = tf.nn.relu(conv2d(x_image , W_conv11) + b_conv11)
h12 = tf.nn.relu(conv2d(h11 , W_conv12) + b_conv12)
h13 = tf.nn.relu(conv2dstr2(h12 , W_conv13) + b_conv13)

h13 = tf.nn.dropout(h13,keep_prob)

h21 = tf.nn.relu(conv2d(h13 , W_conv21) + b_conv21)
h22 = tf.nn.relu(conv2d(h21 , W_conv22) + b_conv22)
h23 = tf.nn.relu(conv2dstr2(h22 , W_conv23) + b_conv23)

h23 = tf.nn.dropout(h23,keep_prob)

h31 = tf.nn.relu(conv2d(h23,W_conv31) + b_conv31)
h32 = tf.nn.relu(conv2d(h31,W_conv32,"VALID") + b_conv32)
h33 = tf.nn.relu(conv2d(h32,W_conv33,"VALID") + b_conv33)

h_out = tf.nn.avg_pool(h33,ksize=[1,8,8,1],strides=[1,8,8,1],padding="VALID")
h_out = tf.reshape(h_out,(-1,10))

cost = reg_term + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_out))

train = tf.train.MomentumOptimizer(decayed_learning_rate1,0.9).minimize(cost,global_step)
correct_pred = tf.equal(tf.argmax(h_out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)

dict = load_data.create_data()
dict["data"] = load_data.GCN(dict["data"])
dict["data"] = load_data.zca_whitening(dict["data"])
dict["labels"] = load_data.convert_one_hot(dict["labels"],10)

for i in range(30000):
	batch_x, batch_y = load_data.next_batch(dict,128,1)
	feed = {x:batch_x, y:batch_y, keep_prob:1.0, keep_prob_input:1.0}
	if(i%100 == 0):
		sum=0.0
		for j in range(100):
			batch_xs, batch_ys = load_data.next_batch(test_dict,100,0)
			sum = sum + accuracy.eval(feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0, keep_prob_input:1.0})
		train_accuracy = accuracy.eval(feed_dict=feed)
		training_cost = cost.eval(feed_dict=feed)
		print("Step %d Training Accuracy: %f Cost: %f  Test Accuracy: %f" %((i/100 + 1), train_accuracy, training_cost,(sum/100.0)))
	train.run(feed_dict={x:batch_x, y:batch_y, keep_prob:0.5, keep_prob_input:0.8})

sum=0.0
for i in range(100):
	batch_x, batch_y = load_data.next_batch(test_dict,100,0,start=i*100)
	t = accuracy.eval(feed_dict={x:batch_x, y:batch_y, keep_prob:1.0,keep_prob_input:1.0})
	sum = sum + t
	print("%f ---- %f" %(t,sum))
print("Test Accuracy: %f" %(sum/100.0))