"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import facenet
import lfw
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import os
# Creates a graph.
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def focal_loss(onehot_labels, cls_preds,
                            alpha=1, gamma=2.0, name=None, scope=None):
    """Compute softmax focal loss between logits and onehot labels
    logits and onehot_labels must have same shape [batchsize, num_classes] and
    the same data type (float16, 32, 64)
    Args:
      onehot_labels: Each row labels[i] must be a valid probability distribution
      cls_preds: Unscaled log probabilities
      alpha: The hyperparameter for adjusting biased samples, default is 0.25
      gamma: The hyperparameter for penalizing the easy labeled samples
      name: A name for the operation (optional)
    Returns:
      A 1-D tensor of length batch_size of same type as logits with softmax focal loss
    """
    with tf.name_scope(scope, 'focal_loss', [cls_preds, onehot_labels]) as sc:
        logits = tf.convert_to_tensor(cls_preds)
        onehot_labels = tf.convert_to_tensor(onehot_labels)
        onehot_labels = tf.one_hot(onehot_labels,7)


        
        precise_logits = tf.cast(logits, tf.float32) if (
                        logits.dtype == tf.float16) else logits
        onehot_labels = tf.cast(onehot_labels, precise_logits.dtype)
        #predictions = tf.nn.sigmoid(logits)
        predictions = tf.nn.softmax(logits)
        #losses = tf.reduce_mean(-tf.reduce_sum(onehot_labels * tf.log(predictions), [1]))




        predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
         # add small value to avoid 0
        epsilon = 1e-8
        # alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
        # alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
        # fl = tf.multiply(alpha_t ,tf.pow(1. - predictions_pt, gamma)) 
        # test = fl-alpha_t
        power = tf.pow(1. - predictions_pt + 0.001, gamma)
        power = tf.cast(power, tf.float32)
        # ind = tf.ones([50,10],tf.float32)
        # test = power - ind
        losses = -tf.reduce_sum(power*onehot_labels * tf.log(predictions_pt+epsilon),[1],name='cross_entropy_per_example')
        # tf.pow(1. - predictions_pt, gamma) *
        #inx = tf.pow(1. - predictions_pt, gamma)
        losses = tf.reduce_mean(losses,name='cross_entropy')
    return losses

def main(args):
  
    network = importlib.import_module(args.model_def)

    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    facenet.store_revision_info(src_path, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    train_set = facenet.get_dataset(args.data_dir)
    if args.filter_filename:
        train_set = filter_dataset(train_set, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
    nrof_classes = len(train_set)

#-------------------------------------------------------------------------------------------------------------------    
    test_set = facenet.get_dataset(args.lfw_dir)
    if args.filter_filename:
        test_set = filter_dataset(test_set, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
    nrof_classes = len(test_set) 
#----------------------------------------------------------------------------------------------------------------
    image_test_list, label_test_list = facenet.get_image_paths_and_labels(test_set)
    assert len(image_test_list)>0, 'The dataset should not be empty'
#-----------------------------------------------------------------------------------------------------------------   
#------------------------------------------------------------------------------------------------------------------- 


    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pretrained_model = None
    if args.pretrained_model:
        pretrained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pretrained_model)
    
    if args.lfw_dir:
        print('LFW directory: %s' % args.lfw_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.lfw_pairs))
        # Get the paths for the corresponding images
        lfw_paths, actual_issame = lfw.get_paths(os.path.expanduser(args.lfw_dir), pairs, args.lfw_file_ext)
    
    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)
        
        # Get a list of image paths and their labels
        image_list, label_list = facenet.get_image_paths_and_labels(train_set)
        assert len(image_list)>0, 'The dataset should not be empty'
        
     
        # Create a queue that produces indices into the image_list and label_list 
        labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        range_size = array_ops.shape(labels)[0]
        index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
    
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')

        labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='labels')
        
        input_queue = data_flow_ops.FIFOQueue(capacity=100000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder], name='enqueue_op')
 #------------------------------------------------------------------------------------------------------------------------
        test_labels = ops.convert_to_tensor(label_test_list, dtype=tf.int32)
        test_range_size = array_ops.shape(test_labels)[0]
        test_index_queue = tf.train.range_input_producer(test_range_size, num_epochs=None,
                             shuffle=True, seed=None, capacity=32)
        
        test_index_dequeue_op = test_index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')
        
        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

        batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        
        test_image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='test_image_paths')

        test_labels_placeholder = tf.placeholder(tf.int64, shape=(None,1), name='test_labels')
        
        input_test_queue = data_flow_ops.FIFOQueue(capacity=8000000,
                                    dtypes=[tf.string, tf.int64],
                                    shapes=[(1,), (1,)],
                                    shared_name=None, name=None)
        vali_enqueue_op = input_test_queue.enqueue_many([test_image_paths_placeholder, test_labels_placeholder], name='vali_enqueue_op')  
#-----------------------------------------------------------------------------------------------------------------------     
        nrof_preprocess_threads = 4        
        images_and_labels = []
        for _ in range(nrof_preprocess_threads):
            filenames, label = input_queue.dequeue()
            images = []
            for filename in tf.unstack(filenames):
                file_contents = tf.read_file(filename)
                image = tf.image.decode_image(file_contents, channels=3)
                if args.random_rotate:
                    image = tf.py_func(facenet.random_rotate_image, [image], tf.uint8)
                if args.random_crop:
                    image = tf.random_crop(image, [args.image_size, args.image_size, 3])
                else:
                    image = tf.image.resize_image_with_crop_or_pad(image, args.image_size, args.image_size)
                if args.random_flip:
                    image = tf.image.random_flip_left_right(image)
    
                #pylint: disable=no-member
                image.set_shape((args.image_size, args.image_size, 3))
                images.append(tf.image.per_image_standardization(image))
            images_and_labels.append([images, label])

#--------------------------------------------------------------------------------------------------------------------------------
        test_images_and_labels = []
        test_filenames, test_label = input_test_queue.dequeue()
        test_images = []
        for test_filename in tf.unstack(test_filenames):
            test_file_contents = tf.read_file(test_filename)
            test_image = tf.image.decode_image(test_file_contents, channels=3)
            test_image.set_shape((args.image_size, args.image_size, 3))
            test_images.append(tf.image.per_image_standardization(test_image))
        test_images_and_labels.append([test_images, test_label])

        test_image_batch, test_label_batch = tf.train.batch_join(
            test_images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 *args.batch_size,
            allow_smaller_final_batch=True)
        test_image_batch = tf.identity(test_image_batch, 'test_image_batch')
        test_image_batch = tf.identity(test_image_batch, 'test_input')
        test_label_batch = tf.identity(test_label_batch, 'test_label_batch')
#---------------------------------------------------------------------------------------------------------------------------------
    
        image_batch, label_batch = tf.train.batch_join(
            images_and_labels, batch_size=batch_size_placeholder, 
            shapes=[(args.image_size, args.image_size, 3), ()], enqueue_many=True,
            capacity=4 * nrof_preprocess_threads * args.batch_size,
            allow_smaller_final_batch=True)
        print(image_batch)
        image_batch = tf.identity(image_batch, 'image_batch')
        image_batch = tf.identity(image_batch, 'input')
        label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Total number of classes: %d' % nrof_classes)
        print('Total number of examples: %d' % len(image_list))
        
        print('Building training graph')
        
        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)
        logits = tf.identity(logits, 'final_Logits')
        #trainable=False,

#----------------------------------------------------------------------------------------------------------------------------------

        test_prelogits, _ = network.inference(test_image_batch, 1, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay,reuse=True)
        test_logits = slim.fully_connected(test_prelogits, len(test_set), activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=True)
        with tf.name_scope('accuracy'):
          onehot_labels = tf.one_hot(test_label_batch,7)
          correct_prediction = tf.equal(tf.argmax(test_logits, 1), tf.argmax(onehot_labels, 1))
          correct_prediction = tf.cast(correct_prediction, tf.float32)
        accuracy_ = tf.cast(tf.reduce_mean(correct_prediction),tf.float32)
        #print(accuracy_)

        with tf.name_scope('confusion'):
            confusion = tf.contrib.metrics.confusion_matrix(test_label_batch, tf.argmax(test_logits, 1),num_classes=7)
            #print(confusion)
#-----------------------------------------------------------------------------------------------------------------------------------

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Add center loss
        if args.center_loss_factor>0.0:
            prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        # cross_entropy_mean = focal_loss(onehot_labels=label_batch, cls_preds=logits,
        #                     alpha=0.25, gamma=2, name=None, scope=None)


        tf.add_to_collection('losses', cross_entropy_mean)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        saver1 = tf.train.Saver(tf.trainable_variables()[0:-1], max_to_keep=200)
        saver = tf.train.Saver(max_to_keep=200)
        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                # init_0 = tf.global_variables_initializer()
                # sess.run(init_0)
                # var_names = []
                # name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()),tf.global_variables()))
                # with tf.variable_scope('',reuse=True):
                #     for var_name, saved_var_name in var_names[:-1]:
                #         curr_var = name2var[saved_var_name]
                #         var_shape = curr_var.get_shape().as_list()
                #         if var_shape == saved_shapes[saved_var_name]:
                #             restore_vars.append(curr_var) 
                # saver = tf.train.Saver(restore_vars,max_to_keep=3)
                saver1.restore(sess, pretrained_model)

            # Training and validation loop
            print('Running training')
            epoch = 0
            while epoch < args.max_nrof_epochs:
                step = sess.run(global_step, feed_dict=None)
                epoch = step // args.epoch_size
            #     print(step)
                # Train for one epoch
                train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                    learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                    total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)
                sum_ = 0
                confusion_m = tf.zeros((7,7))
                for i in range(7):
                    acc,m = test(confusion,accuracy_, args, sess, epoch, image_test_list, label_test_list, index_dequeue_op, vali_enqueue_op, test_image_paths_placeholder, test_labels_placeholder,
                            learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
                            total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file)
                    sum_ = sum_ + acc
                    confusion_m = confusion_m + m
                avg = sum_ / 7
                confusion_m = confusion_m/tf.reduce_sum(confusion_m,1)
                print(avg)
                np.set_printoptions(precision=4)  
                print(confusion_m.eval())





                summary = tf.Summary()
                #pylint: disable=maybe-no-member
                summary.value.add(tag='lfw/accuracy', simple_value=avg)
                #summary.value.add(tag='lfw/c_matrix', simple_value=m)
                summary_writer.add_summary(summary, step)
                with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
                    f.write('\n%d\t%.5f\t\n' % (step, avg))
                    f.write(np.array_str(confusion_m.eval()))
                    # Save variables and the metagraph if it doesn't exist already
                    save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, step)

                    # # Evaluate on LFW
                    # if args.lfw_dir:
                    #     evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
                    #         embeddings, label_batch, lfw_paths, actual_issame, args.lfw_batch_size, args.lfw_nrof_folds, log_dir, step, summary_writer)
    return model_dir
  
def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold
  
def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset
  
def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    index_epoch = sess.run(index_dequeue_op)
    label_epoch = np.array(label_list)[index_epoch]
    image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_epoch),1)
    image_paths_array = np.expand_dims(np.array(image_epoch),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    #print(labels_array)
    # Training loop
    train_time = 0
    while batch_number < args.epoch_size:
        start_time = time.time()
        feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
        if (batch_number % 100 == 0):
            err, _, step, reg_loss, summary_str = sess.run([loss, train_op, global_step, regularization_losses, summary_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=step)
        else:
            err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
              (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
        batch_number += 1
        train_time += duration
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, step)
    return step

def test(confusion,accuracy_, args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder, 
      learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, global_step, 
      loss, train_op, summary_op, summary_writer, regularization_losses, learning_rate_schedule_file):
#
    batch_number = 0
    
    if args.learning_rate>0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    # index_epoch = sess.run(index_dequeue_op)
    # label_epoch = np.array(label_list)[index_epoch]
    # image_epoch = np.array(image_list)[index_epoch]
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.array(label_list),1)
    image_paths_array = np.expand_dims(np.array(image_list),1)
    #print(labels_array)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    #print(np.shape(labels_array))
    # # Training loop
    # # train_time = 0
    # # while batch_number < args.epoch_size:
    # #     start_time = time.time()
    feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:False, batch_size_placeholder:args.lfw_batch_size}
    #     # if (batch_number % 100 == 0):
    #     #     err, _, step, reg_loss, summary_str,acc = sess.run([loss, train_op, global_step, regularization_losses, summary_op, accuracy], feed_dict=feed_dict)
    #     #     summary_writer.add_summary(summary_str, global_step=step)
    #     # else:
    #     #     err, _, step, reg_loss = sess.run([loss, train_op, global_step, regularization_losses], feed_dict=feed_dict)
    #     # duration = time.time() - start_time
    #     # print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tRegLoss %2.3f' %
    #     #       (epoch, batch_number+1, args.epoch_size, duration, err, np.sum(reg_loss)))
    acc,m = sess.run([accuracy_,confusion], feed_dict=feed_dict)
    print('accuracy',acc)
    print('confusion_m', m)
    #     # batch_number += 1
    #     # train_time += duration
    # # Add validation loss and accuracy to summary
    # summary = tf.Summary()
    # #pylint: disable=maybe-no-member
    # summary.value.add(tag='time/total', simple_value=train_time)
    # summary_writer.add_summary(summary, step)
    return acc,m

def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, 
        embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer):
    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')
    
    # Enqueue one epoch of image paths and labels
    labels_array = np.expand_dims(np.arange(0,len(image_paths)),1)
    image_paths_array = np.expand_dims(np.array(image_paths),1)
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array})
    
    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(actual_issame)*2
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for _ in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab] = emb
        
    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(emb_array, actual_issame, nrof_folds=nrof_folds)
    
    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    lfw_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='lfw/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='lfw/val_rate', simple_value=val)
    summary.value.add(tag='time/lfw', simple_value=lfw_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'lfw_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))

def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches. Multiple directories are separated with colon.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--lfw_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--lfw_file_ext', type=str,
        help='The file extension for the LFW dataset.', default='png', choices=['jpg', 'png'])
    parser.add_argument('--lfw_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--lfw_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--lfw_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
