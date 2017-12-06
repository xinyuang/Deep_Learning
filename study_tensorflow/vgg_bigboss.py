import tensorflow as tf
import getdata as data
import numpy as np
import os
from PIL import Image

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def costfunction(l,pred):
  mean_angle_error = 0
  total_pixels = 0
  zero = tf.constant(0, dtype=tf.float32)
  pred = tf.cast(pred,tf.float32)
  l = tf.cast(l,tf.float32)
  mask = l[:,:,:,0]
  pred = ((pred / 255.0) - 0.5) * 2
  l = ((l / 255.0) - 0.5) * 2
  total_pixels = tf.cast(tf.count_nonzero(mask),tf.float32)
  mask = tf.not_equal(mask,zero)
  a11 = tf.reduce_sum(pred * pred, axis=3)
  a11 = tf.boolean_mask(a11, mask)
  a22 = tf.reduce_sum(l * l, axis=3)
  a22 = tf.boolean_mask(a22,mask)
  a12 = tf.reduce_sum(pred * l, axis=3)
  a12 = tf.boolean_mask(a12,mask)
  cos_dist = a12 / tf.sqrt(a11 * a22)  
  clip1 = tf.ones(tf.shape(cos_dist))
  clip2 =-1*tf.ones(tf.shape(cos_dist))
  isnan = tf.is_nan(cos_dist)
  cos_dist = tf.where(isnan, clip2, cos_dist)
  ceil = tf.less(cos_dist,1)
  cos_dist = tf.where(ceil,cos_dist,clip1)
  floor = tf.greater(cos_dist,-1)
  cos_dist = tf.where(floor,cos_dist,clip2)
  angle_error = tf.acos(cos_dist)
  mae = tf.reduce_sum(angle_error)/total_pixels
  return cos_dist, angle_error, mae
#-----------------------scale 1-----------------------------------------------------
#-----------------------Graph--------------------------

x = tf.placeholder(tf.float32, shape=(None,128,128,3))
y_ = tf.placeholder(tf.float32, shape=(None,128,128,3))

keep_prob = tf.placeholder(tf.float32)
#--------------------Conv layer 1----------------------

W_conv1 = weight_variable([3, 3, 3, 64])
b_conv1 = bias_variable([64])

x_image1 = tf.reshape(x, [-1,128,128,3])
relu1 = tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)
print "output1 64 64 64"
print relu1.get_shape()

output1 = max_pool_2x2(relu1)
print output1.get_shape()

#--------------------Conv layer 2----------------------

W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])

x_image2 = tf.reshape(output1, [-1,64,64,64])
relu2 = tf.nn.relu(conv2d(x_image2, W_conv2) + b_conv2)
print "output2 32 32 128"
print relu2.get_shape()

output2 = max_pool_2x2(relu2)
print output2.get_shape()

#--------------------Conv layer 3----------------------

W_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])

x_image3 = tf.reshape(output2, [-1,32,32,128])
relu3 = tf.nn.relu(conv2d(x_image3, W_conv3) + b_conv3)
print "output3 16 16 256"
print relu3.get_shape()

output3 = max_pool_2x2(relu3)
print output3.get_shape()

#--------------------Conv layer 4----------------------

W_conv4 = weight_variable([3, 3, 256, 512])
b_conv4 = bias_variable([512])

x_image4 = tf.reshape(output3, [-1,16,16,256])
relu4 = tf.nn.relu(conv2d(x_image4, W_conv4) + b_conv4)
print "output4 8 8 512"
print relu4.get_shape()

output4 = max_pool_2x2(relu4)
print output4.get_shape()

#--------------------Conv layer 5----------------------

W_conv5 = weight_variable([3, 3, 512, 512])
b_conv5 = bias_variable([512])

x_image5 = tf.reshape(output4, [-1,8,8,512])
relu5 = tf.nn.relu(conv2d(x_image5, W_conv5) + b_conv5)
print "output5 4 4 512"
print relu5.get_shape()

output5 = max_pool_2x2(relu5)
print output5.get_shape()

#----------------------FC layer 1----------------------

W_fc1 = weight_variable([8192, 4096])
b_fc1 = bias_variable([4096])

x_image6 = tf.reshape(output5, [-1, 8192])
output6 = tf.nn.bias_add(tf.matmul(x_image6, W_fc1), b_fc1)
print "output6 1 4096"
print output6.get_shape()

#----------------------FC layer 2----------------------

W_fc2 = weight_variable([4096, 32768])
b_fc2 = bias_variable([32768])

x_image7 = tf.reshape(output6, [-1, 4096])
output7_1 = tf.nn.bias_add(tf.matmul(x_image7, W_fc2), b_fc2)
output7 = tf.reshape(output7_1, [-1, 32, 32, 32])
print "output7 32 32 32"
print output7.get_shape()

scale1_output = tf.div(tf.subtract(output7, tf.reduce_min(output7)), 
  tf.subtract(tf.reduce_max(output7), tf.reduce_min(output7))) * 255

#-----------------------scale 2-----------------------------------------------------

#-------------------- pre processing --------------------
W2_conv0 = weight_variable([3, 3, 3, 64])
b2_conv0 = bias_variable([64])

prescale2 = tf.nn.relu(conv2d(x, W2_conv0) + b2_conv0)
print "prescale2 128 128 64"
print prescale2.get_shape()

scale2_input = tf.concat([max_pool_2x2(max_pool_2x2(prescale2)), scale1_output], 3)
print "scale2 input shape: 32 32 96"
print scale2_input.get_shape()

#----------------------Conv layer 1------------------------
W2_conv1 = weight_variable([5, 5, 96, 64])
b2_conv1 = bias_variable([64])

conv2_1 = tf.nn.relu(conv2d(scale2_input, W2_conv1) + b2_conv1)
print "conv2_1 32 32 64"
print conv2_1.get_shape()

#----------------------Conv layer 2------------------------
W2_conv2 = weight_variable([5, 5, 64, 64])
b2_conv2 = bias_variable([64])

conv2_2 = tf.nn.relu(conv2d(conv2_1, W2_conv2) + b2_conv2)
print "conv2_2 32 32 64"
print conv2_2.get_shape()
#----------------------Conv layer 3------------------------
W2_conv3 = weight_variable([5, 5, 64, 64])
b2_conv3 = bias_variable([64])

conv2_3 = tf.nn.relu(conv2d(conv2_2, W2_conv3) + b2_conv3)
print "conv2_3 32 32 64"
print conv2_3.get_shape()

#----------------------Conv layer 4------------------------
W2_conv4 = weight_variable([5, 5, 64, 64])
b2_conv4 = bias_variable([64])

conv2_4 = tf.nn.relu(conv2d(conv2_3, W2_conv4) + b2_conv4)
print "conv2_4 32 32 64"
print conv2_4.get_shape()

#----------------------Conv layer 5------------------------
W2_conv5 = weight_variable([5, 5, 64, 3])
b2_conv5 = bias_variable([3])

conv2_5 = tf.nn.relu(conv2d(conv2_4, W2_conv5) + b2_conv5)
print "conv2_5 32 32 3"
print conv2_5.get_shape()

#----------------------upsample------------------------

output2_5 = tf.image.resize_bilinear(conv2_5, size=(64, 64))
print "scale2_output shape"
print output2_5.get_shape()

scale2_output = tf.div(tf.subtract(output2_5, tf.reduce_min(output2_5)), 
  tf.subtract(tf.reduce_max(output2_5), tf.reduce_min(output2_5))) * 255

#-----------------------scale 3-----------------------------------------------------

#-------------------- pre processing --------------------
W3_conv0 = weight_variable([3, 3, 3, 64])
b3_conv0 = bias_variable([64])

prescale3 = tf.nn.relu(conv2d(x, W3_conv0) + b3_conv0)
print "prescale3 128 128 64"
print prescale3.get_shape()

scale3_input = tf.concat([max_pool_2x2(prescale3), scale2_output], 3)
print "scale3 input: 128 128 67"
print scale3_input.get_shape()

#----------------------Conv layer 1------------------------
W3_conv1 = weight_variable([5, 5, 67, 64])
b3_conv1 = bias_variable([64])

conv3_1 = tf.nn.relu(conv2d(scale3_input, W3_conv1) + b3_conv1)
print "conv3_1 64 64 64"
print conv3_1.get_shape()

#----------------------Conv layer 2------------------------
W3_conv2 = weight_variable([5, 5, 64, 64])
b3_conv2 = bias_variable([64])

conv3_2 = tf.nn.relu(conv2d(conv3_1, W3_conv2) + b3_conv2)
print "conv3_2 64 64 64"
print conv3_2.get_shape()

#----------------------Conv layer 3------------------------
W3_conv3 = weight_variable([5, 5, 64, 64])
b3_conv3 = bias_variable([64])

conv3_3 = tf.nn.relu(conv2d(conv3_2, W3_conv3) + b3_conv3)
print "conv3_3 64 64 64"
print conv3_3.get_shape()

#----------------------Conv layer 4------------------------
W3_conv4 = weight_variable([5, 5, 64, 3])
b3_conv4 = bias_variable([3])

conv3_4 = tf.nn.relu(conv2d(conv3_3, W3_conv4) + b3_conv4)
print "conv3_4 64 64 3"
print conv3_4.get_shape()

outputfinal = tf.image.resize_bilinear(conv3_4, size=(128, 128))
#------------------------------------------------------

pred = tf.div(tf.subtract(outputfinal, tf.reduce_min(outputfinal)), 
  tf.subtract(tf.reduce_max(outputfinal), tf.reduce_min(outputfinal))) * 255

#------------------------------------------------------

norml, ang, loss = costfunction(l=y_,pred=pred)

train_step = tf.train.AdamOptimizer(5e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
img_batch, label_batch = data.train_read_and_decode("get_train_data.tfrecords")
test_img, test_mask= data.test_read_and_decode("get_test_data.tfrecords")
init = tf.global_variables_initializer()
restore = False
saver = tf.train.Saver()

if not restore:
  with tf.Session() as sess:
    sess.run(init)      
    threads = tf.train.start_queue_runners(sess=sess)
    for i in range(2000):
        if i%1 == 0:    
            val, l= sess.run([img_batch, label_batch])
            normlabel, angle, mae, predic = sess.run([norml, ang, loss, pred],feed_dict={x:val,y_:l})
            print("step %d, mae %f"%(i,mae))
            '''
            print "angle errors"
            print angle.shape
            print "cos dis"
            print normlabel
            print "angle error"
            print angle
            print "pred"
            print predic[0, 55, :, 1]
            '''
            train_step.run(feed_dict={x: val, y_: l})
    if not os.path.exists('model'):
        os.makedirs('model')  
    save_path = saver.save(sess,'model/model.ckpt')

else: 
  with tf.Session() as test:
    test.run(init)
    load_path = saver.restore(test,'model/model.ckpt')
    threads = tf.train.start_queue_runners(sess=test)
    test_val,test_m = test.run([test_img,test_mask])
    test_normal = test.run(pred,feed_dict={x:test_val})
    test_normal = test_normal * test_m
    test_normal = np.uint8(test_normal)
    for i in range(2000):
        #print(test_val[i,100,100,2])
        '''
        pic0 = Image.fromarray(test_m[i,:,:,:],'RGB')
        filename0 = 'test/normal/' + str(i) + 'o.png'
        pic0.save(filename0)
        #pic0.show()
        '''
        test_normal = np.uint8(test_normal)
        pic = Image.fromarray(test_normal[i,:,:,:],'RGB')
        filename = 'test/normal/' + str(i) + '.png'
        pic.save(filename)
        #pic.show()
