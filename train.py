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

INPUT_WIDTH = 128
INPUT_HEIGHT = 128
INPUT_DEPTH = 3

DEPTH_MAP_WIDTH = INPUT_WIDTH // 8
DEPTH_MAP_HEIGHT = INPUT_HEIGHT // 8

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
	help="path to input dataset", default='dataset')
ap.add_argument("-m", "--model", type=str, required=False,
	help="path to trained model", default='liveness.model')
ap.add_argument("-l", "--le", type=str, required=False, 
	help="path to label encoder", default= 'le.pickle')
ap.add_argument("-p", "--plot", type=str,
	help="path to output loss/accuracy plot", default="plot.png")
args = vars(ap.parse_args())

# initialize the initial learning rate, batch size, and number of
# epochs to train for
INIT_LR = 1e-4
BS = 2
EPOCHS = 2

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
print("[INFO] loading images...")
realImagePaths = list(paths.list_images(args["dataset"] + '/real'))
proofImagePaths = list(paths.list_images(args["dataset"] + '/fake'))
data = []
labels = {}
depth_labels = []
class_labels = []
# loop over all image paths

for imagePath in realImagePaths[:10]:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio	


	image = cv2.imread(imagePath)
	image = cv2.resize(image, (INPUT_HEIGHT, INPUT_WIDTH))

	class_label = 'real'
	depthPath = imagePath.replace('real', 'depth', 1)
	depth_label = cv2.imread(depthPath, cv2.IMREAD_GRAYSCALE) / 255

	# update the data and labels lists, respectively
	data.append(image)
	class_labels.append(class_label)
	depth_labels.append(depth_label)

for imagePath in proofImagePaths[:10]:
	# extract the class label from the filename, load the image and
	# resize it to be a fixed 32x32 pixels, ignoring aspect ratio	
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (INPUT_HEIGHT, INPUT_WIDTH))

	class_label = 'fake'
	depth_label = np.zeros((DEPTH_MAP_HEIGHT, DEPTH_MAP_WIDTH))

	# update the data and labels lists, respectively
	data.append(image)
	class_labels.append(class_label)
	depth_labels.append(depth_label)


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
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

###### train the network
print("[INFO] training network for {} epochs...".format(EPOCHS))

######### Phase 1 ############
(trainX, testX, trainY, testY) = train_test_split(data, labels['depth_label'],
	test_size=0.5, random_state=42)

trainY = np.array(trainY)
testY = np.array(testY)

model = LivenessNet.build_backbone(height=INPUT_HEIGHT, width=INPUT_WIDTH, depth=INPUT_DEPTH)
model.compile(loss=EDL, optimizer=opt, metrics=EDL)

H = model.fit(x=trainX, y=trainY, batch_size=BS, 
	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS)

## Phase 2

# (trainX, testX, trainY, testY) = train_test_split(data, labels['class_label'],
# 	test_size=0.5, random_state=42)

# model.compile(loss=["binary_crossentropy"], optimizer=opt, metrics='accuracy')SS
# H = model.fit(x=trainX, y=trainY, batch_size=BS, 
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS)

# # evaluate the network
# print("[INFO] evaluating network...")
# pred_depth, pred_class = model.predict(x=testX, batch_size=BS)
# print(classification_report(testY.argmax(axis=1),
# 	predictions.argmax(axis=1), target_names=le.classes_))

# # save the network to disk
# print("[INFO] serializing network to '{}'...".format(args["model"]))
# model.save(args["model"], save_format="h5")

# # save the label encoder to disk
# f = open(args["le"], "wb")
# f.write(pickle.dumps(le))
# f.close()

# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy on Dataset")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])
