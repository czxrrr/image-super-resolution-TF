from PIL import Image
import numpy as np
from glob import glob
'''
This program aims to make image more clear by using convolutional neural network


Python 2.7 with PIL support, as far as I know PIL doesn't support python 3.0 at present
Tensorflow 1.0 and above

'''


import tensorflow as tf
sess = tf.InteractiveSession()

global file_list
global pointer
path= './*.jpg'
file_list = glob(path)
print file_list
pointer= 0
# get data from files
batch_num =1

def getImages(num):
    imgsx=[]
    imgsy=[]
    global pointer
    global file_list
    for i in range(num):
        if pointer==file_list.__len__(): pointer=0
        filename= file_list[pointer]
        #print filename
        img=Image.open(filename)
        imgsx.append(np.array(img.crop((0,0,256,256)).resize((128,128))))
        imgsy.append(np.array(img.crop((256,0,512,256))))
        pointer+=1
    return imgsx,imgsy

# x [128,128,3]   y[512,512,3]
# loss  L1 loss
x_image = tf.placeholder(tf.float32, shape=[batch_num, 128,128,3])
y_original = tf.placeholder(tf.float32, shape=[batch_num,256,256,3])



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def deconv2d(x, W,out_shape):
    return tf.nn.conv2d_transpose(x, W, out_shape,strides=[1, 2, 2, 1], padding='SAME')
# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
#                         strides=[1, 2, 2, 1], padding='SAME')
# def norm(x):
#     return tf.nn.local_response_normalization(x)

# Four convolutional layer


W_conv1 = weight_variable([5, 5, 3, 32])
h_conv1= conv2d(x_image,W_conv1)
# size :128 to  64


W_conv2 = weight_variable([5, 5, 32, 64])
h_conv2= conv2d(h_conv1,W_conv2)
# size :64 to 32

W_conv3 = weight_variable([5, 5, 64, 128])
h_conv3= conv2d(h_conv2,W_conv3)
# size :32 to 16

W_conv4 = weight_variable([5, 5, 128, 128])
h_conv4= conv2d(h_conv3,W_conv4)
# size :16 to 8


h_conv4_duplicated = tf.concat([h_conv4,h_conv4],3)
print h_conv4_duplicated.shape

# Four deconvolutional layers
W_conv4r = weight_variable([5, 5, 128, 256])
h_conv4r = deconv2d(h_conv4_duplicated,W_conv4r,[batch_num,16,16,128])
# size : 8 to 16


h_conv3_duplicated = tf.concat([h_conv4r,h_conv3],3)
W_conv3r = weight_variable([5, 5, 64, 256])
h_conv3r = deconv2d(h_conv3_duplicated,W_conv3r,[batch_num,32,32,64])
# size :16 to 32


h_conv2_duplicated = tf.concat([h_conv3r,h_conv2],3)
W_conv2r = weight_variable([5, 5, 32,128])
h_conv2r = deconv2d(h_conv2_duplicated,W_conv2r,[batch_num,64,64,32])
# size :32 to 64


h_conv1_duplicated = tf.concat([h_conv2r,h_conv1],3)
print h_conv1_duplicated.shape
W_conv1r = weight_variable([5, 5,3, 64])
h_conv1r = deconv2d(h_conv1_duplicated,W_conv1r,[batch_num,128,128,3])
# size :64 to 128


h_conv0_duplicated = tf.concat([h_conv1r,x_image],3)
print h_conv0_duplicated.shape
W_conv0r = weight_variable([5, 5, 3,6])
h_conv0r = deconv2d(h_conv0_duplicated,W_conv0r,[batch_num,256,256,3])


# the final result is h_conv1
L1_loss = tf.reduce_mean(tf.abs(h_conv0r-y_original))

# use L1 loss

train_step = tf.train.AdamOptimizer(1e-3).minimize(L1_loss)
init=tf.global_variables_initializer()
saver=tf.train.Saver()
sess.run(init)


for i in range(10000):
    batch=getImages(batch_num)
    #print batch[1]
    train_step.run(feed_dict={x_image: batch[0], y_original: batch[1]})
    if (i %1000 == 0):
        k=L1_loss.eval(feed_dict={x_image: batch[0], y_original: batch[1]})
        print k
saver.save(sess,'./model',global_step=i)



