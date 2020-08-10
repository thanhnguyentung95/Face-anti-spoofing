# import the necessary packages
from .network_layers import RSGB

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, Flatten

class LivenessNet:
	@staticmethod
	def build_backbone(height, width, depth):
		input_ = Input(shape=(height, width, depth))
		x = RSGB(64)(input_)

		x = RSGB(128)(x)
		x = RSGB(192)(x)
		x = RSGB(128)(x)
		low_level_features = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

		x = RSGB(128)(low_level_features)
		x = RSGB(192)(x)
		x = RSGB(128)(x)
		mid_level_features = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

		x = RSGB(128)(mid_level_features)
		x = RSGB(192)(x)
		x = RSGB(128)(x)
		high_level_features = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

		FEATURE_DIMENSION = height // 8
		features1 = tf.compat.v1.image.resize_bilinear(low_level_features, size = (FEATURE_DIMENSION, FEATURE_DIMENSION))
		features2 = tf.compat.v1.image.resize_bilinear(mid_level_features, size = (FEATURE_DIMENSION, FEATURE_DIMENSION))
		concated_features = tf.concat([features1, features2, high_level_features], axis=-1)

		x = RSGB(128)(concated_features)
		x = RSGB(64)(x)
		depth_single = Conv2D(1, 3, strides=(1, 1), padding='same', activation='relu')(x)

		model = Model(inputs=[input_], outputs=[depth_single])
		return model

	@staticmethod
	def build_classifier(backbone, nclasses):
		# Freeze backbone
		for layer in backbone.layers:
			layer.trainable = False

		depth_single = backbone.output
		flatten_depth = Flatten()(depth_single)

		x = Dense(256, activation='relu')(flatten_depth)
		x = Dropout(0.5)(x)
		x = Dense(64, activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(32, activation='relu')(x)
		x = Dropout(0.5)(x)
		preds = Dense(nclasses, activation='softmax')(x)
		model = Model(inputs=[backbone.input], outputs=[preds])
		
		return model