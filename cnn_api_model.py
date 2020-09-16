# Convolutional Neural Network
import keras
from keras.preprocessing.image import ImageDataGenerator
import os

#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import sys
from keras.utils import plot_model, multi_gpu_model
from keras.models import Model
from keras.layers import Input, Dropout, Activation, merge
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2
from keras import callbacks, optimizers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import time 
import json 

start = time.time()

DEV = False
argvs = sys.argv
argc = len(argvs)

if argc > 1 and (argvs[1] == "--development" or argvs[1] == "-d"):
  DEV = True
if DEV:
  epochs = 2

else:
  epochs = 20000


train_data_path = '/home/toto/COVID_RESEARCH/Data-split/training/'
validation_data_path = '/home/toto/COVID_RESEARCH/Data-split/validation/'

"""
parameters
"""
img_width, img_height = 224,224
batch_size = 32
train_samples = 100
validation_samples = 10

nb_filters1=32
nb_filters2=64


pools_size = 2
classes_num = 2

lr = 0.00001


visible = Input(shape=(img_width,img_height,3))
conv1 = Conv2D(32, kernel_size=7, padding='same', activation='relu')(visible)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(32, kernel_size=5, padding='same', activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(32, kernel_size=7, padding='same', activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(32, kernel_size=5, padding='same', activation='relu')(pool3)
pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
conv5 = Conv2D(64, kernel_size=7, padding='same', activation='relu')(pool4)
pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
#conv6 = Conv2D(64, kernel_size=5, padding='same', activation='relu')(pool5)
#pool6 = MaxPooling2D(pool_size=(2, 2))(conv6)
#conv7 = Conv2D(64, kernel_size=7, padding='same', activation='relu')(pool6)
#pool7 = MaxPooling2D(pool_size=(2, 2))(conv7)


flat = Flatten()(pool5)            
hidden1 = Dense(1024, kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),activation='relu')(flat)
dropout1 = Dropout(0.3)(hidden1)
#hidden2 = Dense(512, activation = 'relu')(dropout1)
hidden2 = Dense(512, kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),activation='relu')(dropout1)
dropout2 = Dropout(0.3)(hidden2)
#hidden3 = Dense(256, activation = 'relu')(dropout2)
hidden3 = Dense(256, kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01),activation='relu')(dropout2)
dropout3 = Dropout(0.3)(hidden3)

output = Dense(classes_num, activation='softmax')(dropout3)
model = Model(inputs=visible, outputs=output)

# Run on Parallel GPU
#==================================================================================================================
parallel_model = multi_gpu_model(model,3)
parallel_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])
#===================================================================================================================

filepath = "weight.best2.covid.CNN757.hdf5"

checkpoint = ModelCheckpoint(filepath,monitor="val_acc",verbose=1,save_best_only=True,mode='max')
callbacks_list = [checkpoint]

#model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=lr),metrics=['accuracy'])


#print(parallel_model.summary())
print(model.summary())



train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=.5,vertical_flip=True, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
                  train_data_path,
                  target_size=(img_width,img_height),
                  batch_size=batch_size,
                  class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
                  validation_data_path,
                  target_size=(img_width,img_height),
                  batch_size=batch_size,
                  class_mode='categorical')


history = parallel_model.fit_generator(
          train_generator,
          samples_per_epoch = train_samples, 
          epochs = epochs,
          validation_data = validation_generator,
          callbacks=callbacks_list,
          validation_steps = validation_samples)  


target_dir = './models/'
if not os.path.exists(target_dir):
   os.mkdir(target_dir)

model.save('./models/model2_Covid_757.h5')
model.save_weights('./models/weight2_Covid_757.h5')

print(history.history.keys())



with open('./models/history2_Covid_757.json','w') as f:
   json.dump(history.history,f)


end = time.time()
dur = end-start

file = open('./models/runtime2_Covid_757.txt','w')
file.write(str(dur))
file.close()

if dur<60:
   print("Execution time:",dur,"seconds")
elif dur>60 and dur<3600:
  
   dur=dur/60
   print("Execution time:",dur,"minutes")

else:
   dur = dur/(60*60)
   print("Execution time:",dur,"hours" )























# plot graph
#plot_model(model, to_file='convolutional_neural_network.png')
