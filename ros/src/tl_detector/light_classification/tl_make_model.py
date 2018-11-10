import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import cv2
import csv
import random
from collections import Counter

def AffineTransImage(image):

    #parameter
    angle_range = 5
    scale_range_min = 0.5
    scale_range = 1
    shift_range = 5
    
    rotation_angle = (np.random.rand() - 0.5) * angle_range
    scale          = (np.random.rand() * scale_range) + scale_range_min
    shiftx         = (np.random.rand() - 0.5) * shift_range
    shifty         = (np.random.rand() - 0.5) * shift_range

    #size of image
    size = (image.shape[1], image.shape[0])

    #center of rotation
    center = (int(size[0]/2), int(size[1]/2))

    #Rotation Matrix
    RotMat = cv2.getRotationMatrix2D(center, rotation_angle, scale) 

    #shift Matrix
    ShiftMat = np.array([[0, 0, shiftx], [0, 0, shifty]])

    #Affine Matrix
    AffineMat = RotMat + ShiftMat

    #Affine transformation(No padding)
    X_mod = cv2.warpAffine(image, AffineMat, size, flags=cv2.INTER_LINEAR) 

    return X_mod


## Model Architecture
EPOCHS = 35
BATCH_SIZE = 32

def tl_Model(x, train_flag):    
    #parameter
    mu = 0         #parameter of initial value
    sigma = 0.05   #parameter of initial value
    
    # Layer 1: Convolutional. Input = 400x300x3. Output = 390x290x6.
    W1 = tf.Variable(tf.truncated_normal([11, 11, 3, 6], mean=mu, stddev=sigma), name='W1', trainable=True)
    b1 = tf.Variable(tf.truncated_normal([6], mean=mu, stddev=sigma), name='b1', trainable=True)
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
    W2 = tf.Variable(tf.truncated_normal([21, 21, 6, 12], mean=mu, stddev=sigma), name='W2', trainable=True)
    b2 = tf.Variable(tf.truncated_normal([12], mean=mu, stddev=sigma), name='b2', trainable=True)
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
    W4 = tf.Variable(tf.truncated_normal([6612, 120], mean=mu, stddev=sigma), name='W4', trainable=True)
    b4 = tf.Variable(tf.truncated_normal([120], mean=mu, stddev=sigma), name='b4', trainable=True)
    fcon1 = tf.add(tf.matmul(flat1, W4), b4)
    
    # ReLU    
    fcon1 = tf.nn.relu(fcon1)

    # dropout
    fcon1_drop = tf.nn.dropout(fcon1, keep_prob=0.9)     
    fcon1 = tf.cond(train_flag, lambda: fcon1_drop, lambda: fcon1)    
    
    # Layer 5: Fully Connected. Input = 120. Output = 84.
    W5 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma), name='W5', trainable=True)
    b5 = tf.Variable(tf.truncated_normal([84], mean=mu, stddev=sigma), name='b5', trainable=True)
    fcon2 = tf.add(tf.matmul(fcon1, W5), b5)
        
    # ReLU    
    fcon2 = tf.nn.relu(fcon2)

    # dropout
    fcon2_drop = tf.nn.dropout(fcon2, keep_prob=0.9)     
    fcon2 = tf.cond(train_flag, lambda: fcon2_drop, lambda: fcon2)    
        
    # Layer 6: Fully Connected. Input = 84. Output = 3.
    W6 = tf.Variable(tf.truncated_normal([84, 3], mean=mu, stddev=sigma), name='W6', trainable=True)
    b6 = tf.Variable(tf.truncated_normal([3], mean=mu, stddev=sigma), name='b6', trainable=True)
    logits = tf.add(tf.matmul(fcon2, W6), b6)    
    
    return logits

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, train_flag:False})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


################### Main Sequence ###################
#def make_model():
#input image
images_list = []

#colors
colors_list = []
 
#read result&image file
f = open('./result.csv', 'r')
reader = csv.reader(f)

for row in reader:
    fileID = row[0]
    colors_list.append(row[1])

    img = cv2.imread('./img/' + str(fileID) + '.jpg')
    images_list.append(img)
    
    
#shuffle <---Time Series Data??
random.seed(0)
random.shuffle(colors_list)

random.seed(0)
random.shuffle(images_list)

colors = np.array(colors_list)
images = np.array(images_list)

#split train, test, valid
data_size = len(colors_list)
train_rate = 0.8
validation_rate = 0.1

train_size = int(data_size * train_rate)
validation_size = int(data_size * validation_rate)

train_x = images[0:train_size]
train_y = colors[0:train_size]
validation_x = images[train_size:train_size+validation_size]
validation_y = colors[train_size:train_size+validation_size]
test_x = images[train_size+validation_size:]
test_y = colors[train_size+validation_size:]

print(images.shape)
print(len(colors))

print(train_x.shape)
print(len(train_y))
print(np.sum(train_y=='0'))
print(np.sum(train_y=='1'))
print(np.sum(train_y=='2'))

print(validation_x.shape)
print(len(validation_y))
print(np.sum(validation_y=='0'))
print(np.sum(validation_y=='1'))
print(np.sum(validation_y=='2'))

print(test_x.shape)
print(len(test_y))
print(np.sum(test_y=='0'))
print(np.sum(test_y=='1'))
print(np.sum(test_y=='2'))

##Flattening the number of examples per label
counter = Counter(train_y)
max_sample = max(counter.values())

for i in range(len(train_y)):
    if i % 50 == 0:
        print(i)
        
    if counter[train_y[i]] < max_sample:
        adding = int((max_sample - counter[train_y[i]]) / counter[train_y[i]]) + 1
        for j in range(adding):
            train_x = np.append( train_x, np.array([train_x[i]]), axis=0 )
            train_y = np.append( train_y, train_y[i] )    
print(train_x.shape)
print(len(train_y))
print(np.sum(train_y=='0'))
print(np.sum(train_y=='1'))
print(np.sum(train_y=='2'))

#Pre-process the data set(Global Contrast Normalization)
#for i in range(len(train_y)):
#    train_x[i] = (train_x[i] - np.mean(train_x[i])) / np.std(train_x[i])

#for i in range(len(validation_y)):
#    validation_x[i] = (validation_x[i] - np.mean(validation_x[i])) / np.std(validation_x[i])

#for i in range(len(test_y)):
#    test_x[i] = (test_x[i]  - np.mean(test_x[i]))  / np.std(test_x[i])

print('\n------------------------------------------------')
print('-----------------Model Learning-----------------')

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 400, 300, 3))
y = tf.placeholder(tf.int32, (None))
train_flag = tf.placeholder(tf.bool)
one_hot_y = tf.one_hot(y, 3)

#learning rate
learning_rate = 0.0005

logits = tl_Model(x, train_flag)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(train_x)

    print("Training...\n")
    for i in range(EPOCHS):

        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = train_x[offset:end], train_y[offset:end]

            #Data Augmentation
            batch_x_mod = np.zeros_like(batch_x)
            for j in range(len(batch_x)):
                batch_x_mod[j] = AffineTransImage( batch_x[j] )
            
            sess.run(training_operation, feed_dict={x: batch_x_mod, y: batch_y, train_flag: True})

        train_accuracy      = evaluate(train_x, train_y)
        validation_accuracy = evaluate(validation_x, validation_y)

        print("EPOCH {} ...".format(i+1))
        print("Train Accuracy = {:.3f}".format(train_accuracy))
        print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))

        #stop at good validation accuracy model
        if validation_accuracy > 0.95:
            break

    modelname = './tl_classifier_model'    
    vars_to_train = tf.trainable_variables()
    saver = tf.train.Saver(vars_to_train)
    saver.save(sess, modelname)
    print("Model saved")
    print()
    print(one_hot_y)

