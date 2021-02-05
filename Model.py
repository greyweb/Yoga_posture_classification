#Importing libraries
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.applications import VGG19
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import splitfolders
import shutil


#Extracting the zip file
zip_ref = zipfile.ZipFile('yogadataset.zip', 'r')
zip_ref.extractall("/yoga")
zip_ref.close()

#Understanding Data

labels = []
train_counts = []
for dirname in os.listdir('yoga/DATASET/TRAIN'):
    labels.append(dirname)
    image_count = 0
    for img in os.listdir(os.path.join('yoga/DATASET/TRAIN',dirname)):
        image_count +=1
    train_counts.append(image_count)


#Printing labels and no of Train images in each class
#If necessary
#print(labels)
#print(train_counts)



#Splitting Train -> Train & Validation (80:20) split

#  ! Run the below code only once:

# Using split_folders 

os.makedirs("DATA")
input_folder = 'yoga/DATASET/TRAIN'
output_folder = 'DATA'
splitfolders.ratio(input_folder, output=output_folder, seed=1337, ratio=(.8, .2), group_prefix=None)
os.makedirs("DATA/test")

# Moving test files in to a new folder
source = 'yoga/DATASET/TEST'
destination= 'DATA/test'
dest = shutil.move(source, destination)  

# !

# Using Image Generator and augmenting train images

train_datagen = ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory("DATA/train",
                                                   batch_size=16,
                                            class_mode='categorical',
                                            target_size=(150, 150),
                                                    shuffle=True)


valid_generator = valid_datagen.flow_from_directory("DATA/val",
                                                    batch_size=16,
                                            class_mode='categorical',
                                                target_size=(150, 150),
                                                    shuffle=False)


test_datagen=ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('DATA/test/TEST',
                                                batch_size=1,
                                                class_mode='categorical',
                                                target_size=(150, 150),
                                                shuffle=False)

#Assigning weights to classes
class_weights = []
total_samples = train_generator.samples
total_classes = len(train_generator.class_indices)
for ele in train_counts:
    result = round(total_samples / (total_classes * ele),2)
    class_weights.append(result)

class_weights = dict(zip(train_generator.class_indices.values(),class_weights))

#print(class_weights)


# Clearing session
tf.keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)


# Model - VGG19

def custom_model():
    
    
    cus_model = VGG19(input_shape = (150,150,3), 
                         weights='imagenet', 
                         include_top=False,)
    
   #Using pre-trained weights from imagenet 
    for layer in cus_model.layers:
        layer.trainable = False

    # Adding layers in a sequential manner
    x = layers.Flatten()(cus_model.output)

    x = layers.Dense(512, activation='relu')(x)
    
    x = layers.Dropout(0.2)(x)
 
    x = layers.Dense(5, activation='softmax')(x)

    model = Model(cus_model.input,x)
    
    return model


#Structure of the model

model = custom_model()
model.summary()

#Callbacks

#Exponential Decay:

def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)

    return exponential_decay_fn


exponential_decay_fn = exponential_decay(lr0=0.0009, s=5)

lr_scheduler_ed = keras.callbacks.LearningRateScheduler(exponential_decay_fn)



early_stopping_m = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

#Compiling the model

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Fitting the model

history = model.fit(train_generator,
                    validation_data=valid_generator,
                    epochs=30,
                    batch_size=32,
                    callbacks=[checkpoint_cb, lr_scheduler_ed, early_stopping_m],
                    verbose=1
                    )

#Saving Models

model.save("model.h5")

#Plotting  graphs

# Training vs Validation Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(history.epoch, acc, 'r', label='Training accuracy')
plt.plot(history.epoch, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.grid(True)
plt.figure()

# Training vs Validation Loss

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(history.epoch, loss, 'r', label='Training Loss')
plt.plot(history.epoch, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.grid(True)
plt.show()

#Learning rate
plt.plot(history.epoch, history.history["lr"], "o-")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title(" exponential_decay", fontsize=14)
plt.grid(True)
plt.show()

#Plotting all history values in the same graph

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0, 1)




