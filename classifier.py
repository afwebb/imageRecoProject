'''
Run training of image classification algorithm
'''

#import modules
import numpy as np
import pandas as pd 
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
FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

#Read in the data, categorize as dog or cat
fileNames = os.listdir('input/train/train')
labels=[]
for f in fileNames:
    if f.split('.')[0]=='dog':
        labels.append(1)
    else:
        labels.append(0)

#Prepare the input data
df = pd.DataFrame({'fileName':fileNames, 'label':labels})
df["label"] = df["label"].replace({0: 'cat', 1: 'dog'})

train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

def create_model(layers=best_params['layers'], nodes=best_params['nodes'], activation='LeakyReLU', regularizer=None):
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

    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['AUC', 'accurcey'])
    return model

model=KerasClassifier(build_fn=create_model, verbose=1) #Build model
result = model.fit(trainDF['fileName'], trainDF['label'], epochs=5) #Train the model

model.model.save("models/keras_model_"+outDir+".h5") #Save the model
