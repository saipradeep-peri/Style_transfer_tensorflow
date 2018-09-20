# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 14:38:29 2018

@author: SaiPradeep
"""
import PIL.Image
import vgg19
import numpy as np
import cv2
import scipy.io
import scipy.misc
from IPython.display import Image, display
import tensorflow as tf

# mean value matrix computed from the image net used here to subtract the mean RGB value
MEAN_VALUES = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

#loading the image from a specified path and preprocessing the image by subtracting the mean values
def load_image(path):
    image = cv2.imread(path)
    res = cv2.resize(image,dsize = (512,512), interpolation=cv2.INTER_CUBIC)
    res = np.reshape(res, ((1,) + res.shape))
   # res = res - MEAN_VALUES
    return res
    
#preprocess the input image by subtracting the mean
def preprocess(img):
    VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    img = img - VGG_MEAN
    
    
#    img[:,:,:,0] -= VGG_MEAN[0]
#    img[:,:,:,1] -= VGG_MEAN[1]
#    img[:,:,:,2] -= VGG_MEAN[2]
    return img
    
def plot_image_big(image):
    image = np.clip(image,0.0,255.0)
    print(type(image))
    print(image.shape)
    image = image.astype(np.uint8)
    display(PIL.Image.fromarray(image[0]))
    
def deprocess_image(x):
    x = x.reshape((512, 512, 3))
    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def mean_squared_error(a,b):
    return tf.reduce_mean(tf.square(a-b))

# conetnt loss is calulated as mean square loss between the content image and output image
def content_loss(content,generated):
    m, H, W, C = generated.get_shape().as_list()   
    J_content = (1/(4*H*W*C)) * tf.reduce_sum(tf.square(content-generated))  
    return J_content
    
def Content_loss_total(sess,model,layer_id):
    return content_loss(sess.run(model[layer_id]),model[layer_id])
    
#gram matrix helps to capture the style of the image by cancelling out the edges and leaving the style behind
def gram_matrix(G):
    GM = tf.matmul(G, tf.transpose(G)) 
    return GM 

#the gram value difference between the generated image and style image is minimized
def style_loss(style,generated):
    m, H, W, C = generated.get_shape().as_list()
    style = tf.reshape(style, [H*W ,C])
    generated = tf.reshape(generated, [H*W ,C])

    gram_style = gram_matrix(tf.transpose(style))
    gram_generated = gram_matrix(tf.transpose(generated))

    J_style_layer = tf.reduce_sum(tf.square(gram_style - gram_generated)) / (4 * C**2 * (W * H)**2)
        
    return J_style_layer

# style loss for calculated for all the style layers
def style_loss_total(sess,style_layers,model):
    
    loss_style = [style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in style_layers]
    layer_weights = [layer_weight for _, layer_weight in style_layers]
    loss_total = sum([layer_weights[l] * loss_style[l] for l in range(len(style_layers))])
    return loss_total
 

#total loss is from both content and style loss and total loss is minimized
def total_cost(content_loss, style_loss, alpha = 10, beta = 10):
    
    total_loss = alpha * content_loss + beta * style_loss
    
    return total_loss

# A random image based on some noise is generated and is used for loss calculation and optimization    
def generate_noise_image(content_image, noise_ratio):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -20, 20,
            (1, 512, 512, 3)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image
    
content_image = load_image('Input/willy_wonka_old.jpg')
print("content_image : ")
plot_image_big(content_image)
content_image = preprocess(content_image)


generated = generate_noise_image(content_image,0.6)

style_image = load_image('Input/style9.jpg')
style_image = preprocess(style_image)
print("style_image : ")
plot_image_big(style_image)

# the weights of vgg19 model is downloaded from web and is used as pretrained weights
model = vgg19.load_vgg_model("imagenet-vgg-verydeep-19.mat")

#the content loss is calculated on conv4_1 layer in vgg model as the edges are detected more at deep layers
Content_Layers = 'conv4_1'

#For style 5 conv layers are used and some weights are used and multiplied as a low value for initial layers and some high values for higher layers
Style_Layers = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0),
]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(content_image))
    Content_Loss = Content_loss_total(sess,model,Content_Layers)
    sess.run(model['input'].assign(style_image))
    Style_Loss = style_loss_total(sess ,Style_Layers ,model)
    Total_loss = total_cost(Content_Loss,Style_Loss)
    optimizer = tf.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(Total_loss)
    sess.run(tf.global_variables_initializer())
    sess.run(model['input'].assign(generated))
    for i in range(2):
        sess.run(train_step)
        if i%1 == 0:
            print("loss at ",  i , " iterations : " , Total_loss.eval())
    mixed_image = sess.run(model['input'])
    #the mean values are added back
    deprocess_image(mixed_image)
    #image = mixed_image + MEAN_VALUES
    
    # Get rid of the first useless dimension, what remains is the image.
    image = mixed_image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    scipy.misc.imsave('Output/output.jpg', image)
    print("output_image : ")
    plot_image_big(image)
     

    
    
