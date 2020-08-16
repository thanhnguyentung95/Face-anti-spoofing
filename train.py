# USAGE
# python train.py --dataset dataset --model liveness.model --le le.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import tensorflow as tf
from pyimagesearch.livenessnet import LivenessNet
from pyimagesearch.network_loss import EDL
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
from utils import load_config, get_run_logdir


CONFIG = load_config()

INPUT_HEIGHT = CONFIG['input_size']['height']
INPUT_WIDTH = CONFIG['input_size']['width']
INPUT_DEPTH = CONFIG['input_size']['depth']

DEPTH_MAP_WIDTH = INPUT_WIDTH // 8
DEPTH_MAP_HEIGHT = INPUT_HEIGHT // 8
DEPTH_MAP_DEPTH = 1

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = float(CONFIG['initial_learning_rate'])
BS = CONFIG['training_param']['batch_size']
NUM_EPOCHS_PHASE1 = CONFIG['training_param']['num_epochs_phase1']
NUM_EPOCHS_PHASE2 = CONFIG['training_param']['num_epochs_phase2']

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
realImagePaths = list(paths.list_images(CONFIG['real_face_path']))
proofImagePaths = list(paths.list_images(CONFIG['fake_face_path']))
data = []
labels = {}
depth_labels = np.array([])
class_labels = []
# loop over all image paths

if CONFIG['debugging']:
	realImagePaths = realImagePaths[:10]
	proofImagePaths = proofImagePaths[:10]

for imagePath in realImagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio	


	image = cv2.imread(imagePath)
	image = cv2.resize(image, (INPUT_HEIGHT, INPUT_WIDTH))

	class_label = 'real'
	depthPath = imagePath.replace('real', 'depth', 1)
	depth_label = cv2.imread(depthPath, cv2.IMREAD_GRAYSCALE) / 255
	depth_label = cv2.resize(depth_label, (DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH))
	depth_label = np.expand_dims(depth_label, axis=(0, 3))

	# update the data and labels lists, respectively
	data.append(image)
	class_labels.append(class_label)
	if depth_labels.shape[0] == 0:
		depth_labels = depth_label
	else:
		depth_labels = np.vstack((depth_labels, depth_label))

for imagePath in proofImagePaths:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio	
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (INPUT_HEIGHT, INPUT_WIDTH))

	class_label = 'fake'
	depth_label = np.zeros((1, DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH, DEPTH_MAP_DEPTH))

	# update the data and labels lists, respectively
	data.append(image)
	class_labels.append(class_label)
	if depth_labels.shape[0] == 0:
		depth_labels = depth_label
	else:
		depth_labels = np.vstack((depth_labels, depth_label))


# convert the data into a NumPy array, then preprocess it by scaling
# all pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0

# encode the labels (which are currently strings) as integers and then
# one-hot encode them
le = LabelEncoder()
class_labels = le.fit_transform(class_labels)
class_labels = to_categorical(class_labels, 2)

labels['depth_label'] = depth_labels
labels['class_label'] = class_labels


# initialize the optimizer and model
print("[INFO] compiling model...")
opt = CONFIG['optimizer']['name']
if opt == 'adam':
	opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS_PHASE1)

###### train the network
print("[INFO] training network for {} epochs...".format(NUM_EPOCHS_PHASE1))

######### Phase 1 ############
(trainX, testX, trainY, testY) = train_test_split(data, labels['depth_label'],
	test_size=CONFIG['training_param']['test_size'], random_state=42)

model = LivenessNet.build_backbone(height=INPUT_HEIGHT, width=INPUT_WIDTH, depth=INPUT_DEPTH)
model.compile(loss=EDL, optimizer=opt, metrics=EDL)
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir())
H = model.fit(x=trainX, y=trainY, batch_size=BS, 
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=NUM_EPOCHS_PHASE1, callbacks=[tensorboard_cb])

# Phase 2

(trainX, testX, trainY, testY) = train_test_split(data, labels['class_label'],
	test_size=CONFIG['training_param']['test_size'], random_state=42)
model = LivenessNet.build_classifier(model, 2)
model.compile(loss=["categorical_crossentropy"], optimizer=opt, metrics='accuracy')
tensorboard_cb = tf.keras.callbacks.TensorBoard(get_run_logdir())
H = model.fit(x=trainX, y=trainY, batch_size=BS, 
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=NUM_EPOCHS_PHASE2, callbacks=[tensorboard_cb])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(x=testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))

# save the network to disk
print("[INFO] serializing network to '{}'...".format(CONFIG['model_path']))
model.save(CONFIG['model_path'])

# save the label encoder to disk
f = open(CONFIG['encoder_path'], "wb")
f.write(pickle.dumps(le))
f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, NUM_EPOCHS_PHASE2), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, NUM_EPOCHS_PHASE2), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, NUM_EPOCHS_PHASE2), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, NUM_EPOCHS_PHASE2), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(CONFIG['output_plot_path'])
