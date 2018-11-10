from styx_msgs.msg import TrafficLight
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import numpy as np
import cv2

def tl_Model(x, train_flag):    
    #parameter
    mu = 0         #parameter of initial value
    sigma = 0.05   #parameter of initial value
    
    # Layer 1: Convolutional. Input = 400x300x3. Output = 390x290x6.
    W1 = tf.Variable(tf.truncated_normal([11, 11, 3, 6], mean=mu, stddev=sigma), name='W1')
    b1 = tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sigma), name='b1')
    strides = [1, 1, 1, 1]
    conv1 = tf.nn.conv2d(x, W1, strides=strides, padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b1)

    # ReLU    
    conv1 = tf.nn.relu(conv1)
        
    # Pooling. Input = 390x290x6. Output = 78x58x6.
    ksize = [1, 5, 5, 1]
    strides = [1, 5, 5, 1]
    pool1 = tf.nn.max_pool(conv1, ksize, strides=strides, padding='SAME')  
    
    # Layer 2: Convolutional. Output = 58x38x12.
    W2 = tf.Variable(tf.truncated_normal([21, 21, 6, 12], mean=mu, stddev=sigma), name='W2')
    b2 = tf.Variable(tf.truncated_normal([12], mean=mu, stddev=sigma), name='b2')
    strides = [1, 1, 1, 1]
    conv2 = tf.nn.conv2d(pool1, W2, strides=strides, padding='VALID')
    conv2 = tf.nn.bias_add(conv2, b2)   
        
    # ReLU    
    conv2 = tf.nn.relu(conv2)
    
    # Pooling. Input = 58x38x12. Output = 29x19x12.
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    pool2 = tf.nn.max_pool(conv2, ksize, strides=strides, padding='SAME')

    # Flatten. Input = 29x19x12. Output = 6612.
    flat1 = flatten(pool2)

    # dropout
    flat1_drop = tf.nn.dropout(flat1, keep_prob=0.9)     
    flat1 = tf.cond(train_flag, lambda: flat1_drop, lambda: flat1)        
    
    # Layer 4: Fully Connected. Input = 6612. Output = 120.
    W4 = tf.Variable(tf.truncated_normal([6612, 120], mean=mu, stddev=sigma), name='W4')
    b4 = tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sigma), name='b4')
    fcon1 = tf.add(tf.matmul(flat1, W4), b4)
    
    # ReLU    
    fcon1 = tf.nn.relu(fcon1)

    # dropout
    fcon1_drop = tf.nn.dropout(fcon1, keep_prob=0.9)     
    fcon1 = tf.cond(train_flag, lambda: fcon1_drop, lambda: fcon1)    
    
    # Layer 5: Fully Connected. Input = 120. Output = 84.
    W5 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma), name='W5')
    b5 = tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sigma), name='b5')
    fcon2 = tf.add(tf.matmul(fcon1, W5), b5)
        
    # ReLU    
    fcon2 = tf.nn.relu(fcon2)

    # dropout
    fcon2_drop = tf.nn.dropout(fcon2, keep_prob=0.9)     
    fcon2 = tf.cond(train_flag, lambda: fcon2_drop, lambda: fcon2)    
        
    # Layer 6: Fully Connected. Input = 84. Output = 3.
    W6 = tf.Variable(tf.truncated_normal([84, 3], mean=mu, stddev=sigma), name='W6')
    b6 = tf.Variable(tf.truncated_normal([3], mean=mu, stddev=sigma), name='b6')
    logits = tf.add(tf.matmul(fcon2, W6), b6)    
    
    return logits


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.x = tf.placeholder(tf.float32, (None, 400, 300, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.train_flag = tf.placeholder(tf.bool)
        
        self.logits = tl_Model(self.x, self.train_flag)
        self.pred = tf.argmax(self.logits, 1)            

        self.sess = tf.InteractiveSession()  
        saver = tf.train.Saver()
        saver.restore(self.sess, "./light_classification/tl_classifier_model")

        self.image_prediction = tf.argmax(self.logits, 1)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction        
        image = image[0:400,250:550]
        
        #image = (image - np.mean(image)) / np.std(image)        
        image = np.resize(image, (1, 400, 300, 3))        

        #light_color_index = self.sess.run(self.logits, feed_dict={self.x: image, self.y: np.array([0,1,2]), self.train_flag:False})    
        #print(light_color_index)

        light_color_pred = self.sess.run(self.pred, feed_dict={self.x: image, self.y: np.array([0,1,2]), self.train_flag:False})    
        #print(light_color_pred)
        return light_color_pred
#        return TrafficLight.GREEN
