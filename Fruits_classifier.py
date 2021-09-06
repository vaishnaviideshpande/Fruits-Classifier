#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix


# In[2]:


datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   preprocessing_function=preprocess_input)


# In[3]:


training_set = datagen.flow_from_directory(r'D:\Desktop\Projects\Fruits_131\fruits-360_dataset\fruits-360\Training',
                                           target_size = [256,256],
                                           shuffle = False,
                                           batch_size =16)


# In[4]:


val_set = datagen.flow_from_directory(r'D:\Desktop\Projects\Fruits_131\fruits-360_dataset\fruits-360\Test',
                                           target_size = [256,256],
                                           shuffle = False,
                                           batch_size =16)


# In[5]:


resnet = ResNet50(input_shape=[256,256,3],weights='imagenet',include_top=False)
for layer in resnet.layers:
    layer.trainable=False
    
x = tf.keras.layers.Flatten()(resnet.output)

x = tf.keras.layers.Dense(units=131,activation='softmax')(x)

model = Model(inputs=resnet.input, outputs=x)
model.summary()


# In[ ]:


model.compile(optimizer='RMSprop',loss='categorical_crossentropy',metrics =['accuracy'])
history = model.fit(training_set,validation_data=val_set,epochs=5,batch_size=16)


# In[ ]:


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


# In[ ]:


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")


# In[ ]:




