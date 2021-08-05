#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())


# In[1]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# In[201]:


import numpy as np
import pandas as pd
from pathlib import Path
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from time import perf_counter
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
from IPython.display import Markdown, display
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D,MaxPooling2D,ZeroPadding2D,LSTM, AveragePooling2D,Input, Flatten , Dropout , BatchNormalization,Embedding,GRU,GlobalMaxPool2D,TimeDistributed
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
import cv2 
import random
from keras.models import Model
from imutils import paths
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.regularizers import l2,l1
from sklearn.utils import class_weight
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


# In[13]:


dir_ = Path('/Users/Admin/chest_xray - Copy/train')
train_file_paths = list(dir_.glob(r'**/*.jpeg'))

dir_ = Path('/Users/Admin/chest_xray - Copy/test')
test_file_paths = list(dir_.glob(r'**/*.jpeg'))

dir_ = Path('/Users/Admin/chest_xray - Copy/val')
val_file_paths = list(dir_.glob(r'**/*.jpeg'))


# In[4]:


def proc_img(filepath):
    labels = [str(filepath[i]).split("\\")[-2]               for i in range(len(filepath))]

    filepath = pd.Series(filepath, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')

    # Concatenate filepaths and labels
    df = pd.concat([filepath, labels], axis=1)

    # Shuffle the DataFrame and reset index
    df = df.sample(frac=1,random_state=0).reset_index(drop = True)
    
    return df

train_df = proc_img(train_file_paths)
val_df = proc_img(val_file_paths)
train_df = pd.concat([train_df,val_df]).reset_index(drop = True)
test_df = proc_img(test_file_paths)


print(f'Number of pictures in the training set: {train_df.shape[0]}')
print(f'Number of pictures in the test set: {test_df.shape[0]}')
print(f'Number of pictures in the validation set: {val_df.shape[0]}\n')


print(f'Number of different labels: {len(train_df.Label.unique())}\n')
print(f'Labels: {train_df.Label.unique()}')

# The DataFrame with the filepaths in one column and the labels in the other one
train_df.head(5)


# In[5]:


vc = train_df['Label'].value_counts()
plt.figure(figsize=(6,5))
sns.barplot(x = sorted(vc.index), y = vc, palette = "magma")
plt.title("Each Category Size in the Training Set", fontsize = 10)
plt.show()


# In[6]:


fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(12, 8),
                        subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(train_df.Filepath[i]))
    ax.set_title(train_df.Label[i], fontsize = 12)
plt.tight_layout(pad=0.5)
plt.show()


# In[305]:


def create_gen():
    # Load the Images with a generator and Data Augmentation
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        validation_split=0.1
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(rescale=1./255,
        dataframe=train_df, x_col='Filepath', y_col='Label',target_size=(224, 224),color_mode='rgb', class_mode='categorical',
        batch_size=16, shuffle=True,seed=0, subset='training',shear_range=0.2,
        zoom_range=0.2,channel_shift_range=5,horizontal_flip=True,vertical_flip=False,fill_mode='constant',cval=200
    )

    val_images = train_generator.flow_from_dataframe(rescale=1./255,
        dataframe=train_df, x_col='Filepath', y_col='Label', target_size=(224, 224),color_mode='rgb',class_mode='categorical',
        batch_size=16,shuffle=True, seed=0,subset='validation',shear_range=0.2,
        zoom_range=0.2,channel_shift_range=5,horizontal_flip=True,vertical_flip=False,fill_mode='constant',cval=200
    )

    test_images = test_generator.flow_from_dataframe(rescale=1./255,
        dataframe=test_df,x_col='Filepath', y_col='Label',target_size=(224, 224), color_mode='rgb', class_mode='categorical',
        batch_size=16, shuffle=False
    )
    
    return train_generator,test_generator,train_images,val_images,test_images


# In[306]:


train_generator,test_generator,train_images,val_images,test_images=create_gen()


# In[307]:


model = Sequential()
momentum = .7

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same",input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(BatchNormalization(momentum=momentum))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.1))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(BatchNormalization(momentum=momentum))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.1))


model.add(SeparableConv2D(filters=128, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
model.add(BatchNormalization(momentum=momentum))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Dropout(rate=0.1))

model.add(TimeDistributed(Flatten()))
model.add(LSTM(64))

model.add(Dense(256, activation='relu'))
model.add(Dense(units=3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[308]:


metrics = ['accuracy',tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=metrics)


# In[309]:


class_weights = class_weight.compute_class_weight("balanced",
                                                 np.unique(train_images.classes),
                                                 train_images.classes)
class_weight_dict = dict(enumerate(class_weights))


# In[310]:


epochs = 50

history = model.fit_generator(
           train_images, steps_per_epoch=25, 
           epochs=epochs, validation_data=val_images, 
           validation_steps=15,class_weight=class_weight_dict)


# In[311]:


print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(
        1,epochs,max(history.history['accuracy']),max(history.history['val_accuracy']) ))
pred = model.predict(test_images)


# In[313]:


loss, acc,pre,rec = model.evaluate(train_images)


# In[312]:


loss, acc,pre,rec = model.evaluate(test_images)


# In[314]:


y_pre=[]
for i in range (pred.shape[0]):
        y_pre.append(np.argmax(pred[i]))


# In[315]:


labels= ['BACTERIA', 'VIRUS', 'NORMAL']
cm  = confusion_matrix(test_images.classes, y_pre)
plt.figure()
plot_confusion_matrix(cm,figsize=(8,5),cmap=plt.cm.Blues)
plt.xticks(range(3), labels, fontsize=10)
plt.yticks(range(3), labels, fontsize=10)
plt.xlabel('Predicted Label',fontsize=10)
plt.ylabel('True Label',fontsize=10)
plt.show()


# In[316]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax = ax.ravel()

for i, met in enumerate(['accuracy', 'loss']):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history['val_' + met])
    ax[i].set_title('Model {}'.format(met))
    ax[i].set_xlabel('epochs')
    ax[i].set_ylabel(met)
    ax[i].legend(['train', 'val'])


# In[317]:


from plot_keras_history import plot_history
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10,10))
plot_history(history, path="standard.png")
plt.show()


# In[318]:


def print_results(y_test, y_pred):
    print('Accuracy   : {:.5f}'.format(accuracy_score(y_pred , y_test))) 
    print('Precision  : {:.5f}'.format(precision_score(y_test , y_pred,average="macro")))
    print('Recall     : {:.5f}'.format(recall_score(y_test , y_pred,average="macro")))
    print('F1         : {:.5f}'.format(f1_score(y_test , y_pred,average="macro")))
    print('Confusion Matrix : \n', confusion_matrix(y_test, y_pred))


# In[319]:


print_results(test_images.classes, y_pre)


# In[320]:


from PIL import Image

incorrect = np.nonzero(test_images.classes != y_pre)[0]

fig, ax = plt.subplots(4, 3, figsize=(12,12))
ax = ax.ravel()
plt.subplots_adjust(wspace=0.25, hspace=0.75)
plt.tight_layout()
i = 0
for c in range (12):
    im=Image.open(test_df["Filepath"][c])
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].imshow(im)
    ax[i].set_title('Predicted Class: {}, Actual Class: {}'.format(y_pre[c], test_images.classes[c]))
    i += 1 

