# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

#import needed to load the image set and preprocess it
from sklearn.preprocessing import LabelEncoder
import skimage
from skimage import data #https://stackoverflow.com/questions/32876962/python-module-import-why-are-components-only-available-when-explicitly-importe
from skimage import transform #used to resize the image to a standard size 
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#import needed to build the model
from keras.layers.convolutional import Convolution2D #create a convolutional layer
from keras.layers.convolutional import MaxPooling2D #create a pooling layer
from keras.layers.core import Activation #activation functions of the neurons
from keras.layers.core import Flatten #flatten multi-dimensional volumes from a convolution layer to a 1D array that will fed a fully connected layer
from keras.layers.core import Dropout #random disconnection of nodes to reduce overfitting
from keras.layers.core import Dense #create a fully connected layer
from keras.models import Sequential #CNN are feedforward sequential networks
import datetime #let measure the time taken by the training
from keras.optimizers import SGD #use the stochastic gradient optimizer

#import needed to plot accuracy
from pyimagesearch.callbacks import trainingMonitor
import matplotlib
matplotlib.use("Agg") # set the matplotlib backend so figures can be saved in the background

#import needed to create the classification report
from sklearn.metrics import classification_report

import numpy as np
import os
import csv

datasetPath = "\\\\192.168.1.37\\Multimedia\\datasets\\watches_categories"
categories = ["Women watches", "Balls-like hours ticks", "Balls and digits hours ticks", "Bars and Roman digits", "unknown 1",
				 "Bars hours ticks", "Bars and digits", "unknown 2", "Ball and Roman digits", "Big digits", 
				 "Colored", "Connected", "Digital", "Diver", "Ice", "Jewel", "Multi screens", "No hours",
				 "original", "pilot", "rectangular", "robust", "roman", "sport"]
outputBaseDirectory = "output" #the output subdirectory 


def loadWatchesData(data_directory, NB_CATEGORIES):
	#This function gets images and their corresponding categories from the images directory
	#It generates a CSV file displaying the number of images per categories for data analysis purposes
	fileType= '.jpg' #We want to use only the images in JPG format. 
	
	dataAnalysisReportFile = os.path.join(outputBaseDirectory, "categoriesAnalysis.csv") #creates the path of the CSV file

	#get the list categories from the image directory
	directories = [d for d in os.listdir(data_directory) 
	               if os.path.isdir(os.path.join(data_directory, d))] #parse the subdirectories in the directory of images
	labels = [] #initialize the list that will contain the image categories
	images = [] #initialize the list that will contain the images themselves.
	print("Found {} categories".format(len(directories))) #Feed back the user to how many categories of images are available

	with open(dataAnalysisReportFile, "w") as output: 
		writer=csv.writer(output, lineterminator=os.linesep) #prepare the generation of the CSV file

		for d in directories: #Loop on each subdirectory (=each categories of images)
			label_directory = os.path.join(data_directory, d) #construct the path of the sub-directory that will be explored
			file_names = [os.path.join(label_directory, f) for f in os.listdir(label_directory) if f.endswith(str(fileType))] #gather paths of all jpg images from that category in a list
			writer.writerow([d, len(file_names)]) #add a row in the CSV file for taht category.
			print("Found {} files in directory {}".format(len(file_names), d))

			for f in file_names: #for each images in that caetegory
				images.append(skimage.data.imread(f)) #make use of the skimage library to read the image and normalize the pixel intensities to values between 0 and 1
				labels.append(d) #the subdirectory name reflect the category name of the image in human readable format.
			print("[INFO] - Loaded {} files from disk".format(len(images))) #Feed back to the user on how many images are found in the category.
		output.close() #close the CSV file when all images are loaded.
	print("[INFO] - Categories Analysis report created in: {}".format(dataAnalysisReportFile)) #Feed back to the user on where to find the report file.
	
	return images, labels #return the images and their categories

def normalizeData(images, WIDTH, HEIGHT):

		#images are now normalized to a standard size of 60 rows on 45 columns
	print("[INFO] - Resizing images")
	images=[transform.resize(image, (HEIGHT, WIDTH)) for image in images] #transform uses (row,cols) resize params
	images = np.array(images) #https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html

	return images
	
def splitDataset(images, labels):
	
	TEST_SIZE = 0.33 #this is the percentage of images from the dataset taht will be used as testing set.

	#and the dataset is split into training set and test set, the test set size is 33% of the total set and the training set is 67%
	print("[INFO] : Splitting data") #Feed back to the user 
	X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=TEST_SIZE, random_state=42) #usage of the sk_learn function to split
	print("Training size: {} Training label size: {}".format(X_train.shape, len(y_train))) #we now have a training set of images and their categories
	print("Testing size: {} Testing label size: {}".format(X_test.shape, len(y_test))) #And a testing set of images ant their categories
	
	return y_train, X_train, y_test, X_test

def encodeLabels(y_train, y_test, categories):
	# transform the training and testing labels into vectors in the range
	# [0, numClasses] -- this generates a vector for each label, where the
	# index of the label is set to `1` and all other entries to `0`
	print("[INFO] : Encoding labels...") #feed back to the user

	#init the label encoder with the categories values from the directory list.
	le = LabelEncoder() #The LabelEncoder class of sklearn package is used
	le.fit(categories) #encode the categories as integer
	y_train=le.transform(y_train) #transform the categories name (=directory names) to integers
	y_test=le.transform(y_test) #transform the categories name (=directory names) to integers
	
	y_train = np_utils.to_categorical(y_train, len(categories)) #build the array of vectors
	y_test = np_utils.to_categorical(y_test, len(categories))

	return y_train, y_test

def initializeModelParameters(imageHeight, imageWidth):
	#for ease of use all model parameters are stored here. An alternative is to create a model object and intialize in its constructor.
	#returns a dict of parameters of the model
	modelParameters={}

	modelParameters["MODEL_TYPE"] = "Sequential" #https://keras.io/getting-started/sequential-model-guide/
	modelParameters["IMAGE_CHANNELS"] = 3 #input images are in 3 channels, RGB
	modelParameters["IMAGE_HEIGHT"] = imageHeight #height of the input images
	modelParameters["IMAGE_WIDTH"] = imageWidth #widthof the input images
	
	
	modelParameters["DROPOUT"] = True #make usage of dropout after each convolutional layer

	modelParameters["K_FILTERS_1"] = 20 #the first layer will contain 20 filter of size 5x5
	modelParameters["FILTER_SIZE_1"] = 5
	modelParameters["POOL_SIZE_1"] = (2,2) #It will be followed by a pooling that reduces the size of the image by 2
	modelParameters["POOL_STRIDE_1"] = (2,2)
	modelParameters["DROPOUT_1"] = 0.25 #finally a dropout of 25% of the nodes will be applied before moving to next layer
	
	modelParameters["K_FILTERS_2"] = 50 #the second layer is made of 50 filter of size 5x5
	modelParameters["FILTER_SIZE_2"] = 5
	modelParameters["POOL_SIZE_2"] = (2,2) #It will be followed by a pooling that reduces the size of the image by 2
	modelParameters["POOL_STRIDE_2"] = (2,2)
	modelParameters["DROPOUT_2"] = 0.50 #finally a dropout of 50% of the nodes will be applied before moving to next layer

	modelParameters["FC_SIZE"] = 500 #the first fully connected layer will contain 500 nodes, 
	modelParameters["NB_CATEGORIES"] = len(categories) #the softmas will drive them down to the whished nb of categories
	
	modelParameters["ACTIVATION"] = "tanh" #and the activation function is always tanh.

	modelParameters["NB_EPOCH"] = 20 #the dataset will be fed 20 times to the network.


	return modelParameters

def buildModel(modelParameters):
	if modelParameters["MODEL_TYPE"] == "Sequential":
		model = Sequential()
		print("[INFO] - Started construction of a sequential neural model")

	else:
		print("[ERROR] - Model type unrecognized")
		return none

	#Add the first layer
	addLayer(model, modelParameters["K_FILTERS_1"], modelParameters["FILTER_SIZE_1"],
						modelParameters["IMAGE_HEIGHT"], modelParameters["IMAGE_WIDTH"],
						 modelParameters["IMAGE_CHANNELS"], modelParameters["ACTIVATION"],
						 modelParameters["POOL_SIZE_1"], modelParameters["POOL_STRIDE_1"],
						 modelParameters["DROPOUT_1"])
	#Add the second layer
	addLayer(model, modelParameters["K_FILTERS_2"], modelParameters["FILTER_SIZE_2"],
						modelParameters["IMAGE_HEIGHT"], modelParameters["IMAGE_WIDTH"],
						 modelParameters["IMAGE_CHANNELS"], modelParameters["ACTIVATION"],
						 modelParameters["POOL_SIZE_2"], modelParameters["POOL_STRIDE_2"],
						 modelParameters["DROPOUT_2"])

	#Add the first fully connected layer
	model.add(Flatten())
	model.add(Dense(modelParameters["FC_SIZE"]))
	model.add(Activation(modelParameters["ACTIVATION"]))
	print("[INFO] - Added fully connected layer of {} size and {} activation function".format(modelParameters["FC_SIZE"], modelParameters["ACTIVATION"] ))


	#add the final layer with softmax
	model.add(Dense(modelParameters["NB_CATEGORIES"]))
	model.add(Activation("softmax"))

	print("[INFO] - Added final fully connected layer of {} size and softmax activation function".format(modelParameters["NB_CATEGORIES"]))

	print("[INFO] - LENET NEURAL NETWORK CREATED")

	return model


def addLayer(model, nbFilters, sizeFilters, imageHeight, imageWidth, imageNbChannels, activation, poolSize, poolStride, dropoutRatio):
	model.add(Convolution2D(nbFilters , sizeFilters, sizeFilters, 
							border_mode="same", 
							input_shape=(imageHeight, imageWidth, imageNbChannels)))
	print("[INFO] - Added filter layer of {} filters, of size {},{}".format(nbFilters,sizeFilters,sizeFilters))


	model.add(Activation(activation))
	print("[INFO] - Using {} activation function for layer".format(activation))

	model.add(MaxPooling2D(pool_size=poolSize , strides=poolStride))
	print("[INFO] - Added pool layer of size {} , with stride {}".format(poolSize, poolStride))

	if dropoutRatio != 0:
		model.add(Dropout(dropoutRatio))
		print("[INFO] - Using {} dropout for layer".format(dropoutRatio))

	return model

#Load images and labels from disk to memory
images, labels = loadWatchesData(datasetPath, len(categories))
RESIZE_WIDTH = 45
RESIZE_HEIGHT = 60
images = normalizeData(images, RESIZE_WIDTH, RESIZE_HEIGHT)
y_train, X_train, y_test, X_test = splitDataset(images, labels)
y_train, y_test = encodeLabels(y_train, y_test, categories)

print("[INFO] - Size of training set {} / Size of test set: {}".format(X_train.shape, X_test.shape))

#Now that we have our dataset ready, let's start building the network
# Build the model using SGD
print("[INFO] - compiling model...")
time_start= datetime.datetime.now() #Let's start a timer to check how long it takes to train

#get model parameters 
modelParams = initializeModelParameters(RESIZE_HEIGHT, RESIZE_WIDTH)

#Initialize the model
model = buildModel(modelParams)

#select the SGD as optimizer and compile it
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#construct the plot callbacks 
#at each epoch, a measure of the testing and training set accuracy will be taken and saved as a picture and json file
figPath=os.path.sep.join([outputBaseDirectory, "{}.png".format(os.getpid())]) #the file name is taken from the process id
jsonPath = os.path.sep.join([outputBaseDirectory, "{}.json".format(os.getpid())]) #the file name is taken from the process id
callbacks = [trainingMonitor(figPath, jsonPath=jsonPath)] #Use the trainingMonitor Keras function to build the graph after each epoch

 
# start the training process
print("[INFO] - starting training...")
model.fit(X_train, y_train, batch_size=32, validation_data=(X_test, y_test),
	nb_epoch=modelParams["NB_EPOCH"], callbacks=callbacks, verbose=1)

# show the accuracy on the testing set
(loss, accuracy) = model.evaluate(X_test, y_test,batch_size=32, verbose=1)
print("[INFO] - accuracy: {:.2f}%".format(accuracy * 100))

#compute and print duration
timeSpent = (datetime.datetime.now()-time_start).total_seconds()
print("[INFO] - duration in seconds: {}".format(timeSpent))

#save the model to disk
print("[INFO] - dumping architecture and weights to file...")
outputModelPath=os.path.join(outputBaseDirectory, "watches_lenet.hdf5")
model.save(outputModelPath)

