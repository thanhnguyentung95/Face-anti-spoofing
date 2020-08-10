from tensorflow.keras.layers import Conv2D, BatchNormalization
from tensorflow.keras import Model
from tensorflow.math import square
import tensorflow as tf
import numpy as np


# assumed auto broadcasting

class RSGB(Model):
    def __init__(self, filters):
        super(RSGB, self).__init__(name='')
        self.conv = Conv2D(filters, 3, padding='same')
        self.pw_conv = Conv2D(filters, 1, padding='same')
        # 2 options, BatchNormalization and GroupNormalization. For now just stick to BN.
        self.gradient_norm = BatchNormalization() 
        self.layer_norm = BatchNormalization()


    def call(self, input_tensor, training=False):
        main_stream = self.conv(input_tensor)

        gradient_x = spatial_gradient_x(input_tensor)
        gradient_y = spatial_gradient_y(input_tensor)
        gradient_x = square(gradient_x)
        gradient_y = square(gradient_y)
        spatial_gradient = gradient_x + gradient_y
        spatial_gradient = self.gradient_norm(spatial_gradient)
        pw_spatial_gradient = self.pw_conv(spatial_gradient)
        
        outstream = tf.math.add(main_stream, pw_spatial_gradient)  # support broadcasting
        outstream = self.layer_norm(outstream)
        outstream = tf.nn.relu(outstream)

        return outstream
        

def spatial_gradient_x(input, name=''):
    sobel_plane_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
    sobel_plane_x = np.repeat(sobel_plane_x, input.get_shape().as_list()[-1], axis=-1)
    sobel_plane_x = np.expand_dims(sobel_plane_x, axis=-1)
    sobel_kernel_x = tf.constant(sobel_plane_x, dtype=tf.float32)

    Spatial_Gradient_x = tf.nn.depthwise_conv2d(input, filter=sobel_kernel_x, \
                                                strides=[1,1,1,1], padding='SAME', name=name+'/spatial_gradient_x')
    return Spatial_Gradient_x

def spatial_gradient_y(input, name=''):
    sobel_plane_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
    sobel_plane_y = np.repeat(sobel_plane_y, input.get_shape().as_list()[-1], axis=-1)
    sobel_plane_y = np.expand_dims(sobel_plane_y, axis=-1)
    sobel_kernel_y = tf.constant(sobel_plane_y, dtype=tf.float32)

    Spatial_Gradient_y = tf.nn.depthwise_conv2d(input, filter=sobel_kernel_y, \
                                                strides=[1,1,1,1], padding='SAME', name=name+'/spatial_gradient_y')
    return Spatial_Gradient_y