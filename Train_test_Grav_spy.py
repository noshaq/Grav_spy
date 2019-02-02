import math
import os.path
import numpy as np
import pandas as pd
from keras import optimizers
from keras.models import load_model
from conff_matrix import cm_analysis
from learning_curve_sklearn import *
from sklearn.utils import class_weight
from keras.callbacks import TensorBoard
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau,LearningRateScheduler



# Neural network: VGG16

from VGG_Grav_spy import VGG


# Read the preprocesed files and generate the batches to train. 
# We only scale the pixel values. Other options wi

#  Numper of images loaded in each iteration
print("Importing the VGG16")

batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
	shuffle=True,
        seed=42)

validation_generator = valid_datagen.flow_from_directory(
        './data/valid',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical', shuffle=True,
    	seed=42)

test_generator = test_datagen.flow_from_directory(
        './data/test',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
	shuffle=False,
    	seed=42)



print("--------------------------------------------------\n")



nb_classes = train_generator.num_classes
print("Number of classes = ",nb_classes)


channels = test_generator.image_shape[2]

print("Number of channels =",channels)


print("--------------------------------------------------\n")

#########  Create the folder to store weights

if os.path.isdir("./models")== False:
         os.mkdir("./models")



#########  Create the folder to store metrics

if os.path.isdir("./performance_plots")== False:
         os.mkdir("./performance_plots")


# Verify if exists a trained model


print("-------------------------\n")

# Path to save the full model for later usage
bst_model_path = './models/GravSpy.h5'


###################### The VGG16 ################################################
# Arguments:
# 		 train_generator: Generate batches of tensor image from the train folder with 
#        	             real-time data augmentation (Only normalization in this case)
# 			             The data will be looped over (in batches). Same for validation_generator & test_generator
#        channels: Specify if the image is grayscale (1) or RGB (3)
#        nb_epoch: Number of epochs
#        nb_classes: Number of classes for classification
#
################################################################################################
if os.path.isfile(bst_model_path) == False:
	print("Pretrained model not found, training VGG16\n")


#######  This is an imbalanced problem, computing the class wegths

	number_of_examples = len(train_generator.filenames)
	number_of_generator_calls = math.ceil(number_of_examples / (1.0 * train_generator.batch_size )) 

	train_labels = []

	for i in range(0,int(number_of_generator_calls)):
	    train_labels.extend(np.array(train_generator[i][1]))

	y_true_train=np.argmax(train_labels,axis=1)

	weights_per_class= class_weight.compute_class_weight('balanced',np.unique(y_true_train), y_true_train)   

	class_weight_dict = dict(enumerate(weights_per_class))
                                                 


	model = VGG(train_generator, validation_generator, test_generator, nb_classes, channels)

	################## Set the loss function, the optimizer and the main metric #####################################3

	sgd = optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss = 'categorical_crossentropy',
	                    optimizer=sgd,
	                    metrics=['accuracy','mse'])


	# Reduce the learning rate when the accuracy is the same for 3 consecutive epochs

	reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1,
	                                         mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.00001)

	# Avoid overfiting by setting a  boundary in the precision change:

	stop = EarlyStopping(monitor='val_acc',
	                            min_delta=0.001,
	                            patience=30,
	                            verbose=0,
	                            mode='auto')


	model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False, mode='max', monitor='val_acc')


	# Enable Tensorboard visualization
	tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

	print("--------------------------------------------------\n")

	print("Now training, please wait \n")

	# Step size for the iterators

	STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
	STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size


	estimator = model.fit_generator(generator=train_generator,
	                    steps_per_epoch=STEP_SIZE_TRAIN,
	                    validation_data=validation_generator,
	                    validation_steps=STEP_SIZE_VALID,
	                    epochs=25,
	                    verbose=1,
	                    class_weight=class_weight_dict,
	                    callbacks=[stop, tensor_board,model_checkpoint,reduceLROnPlat])

	print("Done!...... Succesfully????")


	learing_curve_keras(estimator,VGG)

else:
	print("Pretrained model finded, loading:")
	model = load_model('./models/GravSpy.h5')

	model.summary()



############ Check the quality of the trained model by the use of ROC curves, and confussion matrix ###########


##################### ROC curve for test dataset #########################################

print("Performance of the model... Making some plots")

#test_generator.reset()
# Predict the probability of belong to one class:
number_of_examples = len(test_generator.filenames)
number_of_generator_calls = math.ceil(number_of_examples / (1.0 * test_generator.batch_size )) 

predicted_prob=model.predict_generator(test_generator, steps = number_of_generator_calls ,verbose=1)

#predicted_prob=model.predict_generator(test_generator,verbose=1)


test_labels = []

for i in range(0,int(number_of_generator_calls)):
    test_labels.extend(np.array(test_generator[i][1]))


roc_all = plot_ROC_multiclass(np.array(test_labels),np.array(predicted_prob),nb_classes,"VGG16")



# Confussion Matrix

"""
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
predicted_class_indices=np.argmax(predicted_prob,axis=1)
y_true=np.argmax(test_labels,axis=1)

labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
labels_encoded = list(labels.keys())

predictions = [labels[k] for k in predicted_class_indices]


filename = "conf_matrix_gravspy"

cm_analysis(y_true, predicted_class_indices, filename, labels_encoded, ymap=None, figsize=(20,20))

# Print the classification report

report = classification_report(y_true, predicted_class_indices, target_names=labels.values())
print(report)
with open('./performance_plots/quality_report.txt', 'w') as f:
    f.write(report)
    f.close()

# Dump the filenames and predicted label
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("./performance_plots/results.csv",index=False)

