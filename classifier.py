'''
Run training of image classification algorithm
'''

#import modules
import numpy as np
import pandas as pd 
import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
print(os.listdir("input"))

#Define constants
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#Read in the data, categorize as dog or cat
fileNames = os.listdir('input/train')
labels=[]
for f in fileNames:
    if f.split('.')[0]=='dog':
        labels.append(1)
    else:
        labels.append(0)

#Prepare the input data
df = pd.DataFrame({'fileName':fileNames, 'category':labels})
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

def create_model():#layers=best_params['layers'], nodes=best_params['nodes'], activation='LeakyReLU', regularizer=None):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Dense(2, activation='softmax'))

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['AUC', 'accuracy'])
    return model

model=create_model()#KerasClassifier(build_fn=create_model, verbose=1) #Build model
#result = model.fit(train_df['fileName'], train_df['category'], epochs=5) #Train the model

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15

train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "train", 
    x_col='fileName',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "input/train", 
    x_col='fileName',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

train_generator = train_datagen.flow_from_dataframe(train_df,
                                                    "./input/train/",x_col='fileName',y_col='category',
                                                    target_size=IMAGE_SIZE,
                                                    class_mode='categorical',
                                                    batch_size=batch_size)

epochs=3 
history = model.fit_generator(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size
    #callbacks=callbacks
)

model.model.save("models/keras_model_"+outDir+".h5") #Save the model
