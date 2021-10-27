import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import matplotlib.pyplot as plt
import time

ruta_dataset_images='\Images' #Path where the distance maps are located.
ruta_guardado_NN='\Models' #Path where the models will be saved.

#-----------------------------------------------------------
#We define a 'callback' to save the network as soon as a hit 
#rate > 92% is obtained.
#-----------------------------------------------------------
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.92) and (logs.get('val_loss')<0.2):
            print("\nReached 92% accuracy so cancelling training!")
            print('Validation accuracy: {}'.format(logs.get('val_accuracy')))
            print('Train accuracy: {}'.format(logs.get('accuracy')))
            print('Validation loss: {}'.format(logs.get('val_loss')))
            print('Train_loss: {}'.format(logs.get('loss')))
            self.model.save(ruta_guardado_NN+'red_V03_{}.tf'.format(str(logs.get('val_loss'))))

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

#-----------------------------------------------------------
#Preparation of the train, validation and test datasets
#-----------------------------------------------------------
train_dir = os.path.join(ruta_dataset_images, 'train')
validation_dir = os.path.join(ruta_dataset_images, 'validation')
test_dir = os.path.join(ruta_dataset_images, 'test')

# Directory 'train' with agonist, antagonist and inverse agonist images. 
train_Agonista_dir=os.path.join(train_dir, 'Agonista')
train_Antagonista_dir=os.path.join(train_dir, 'Antagonista')

# Directory 'validation' with agonist, antagonist and inverse agonist images. 
validation_Agonista_dir=os.path.join(validation_dir, 'Agonista')
validation_Antagonista_dir=os.path.join(validation_dir, 'Antagonista')

# Directory 'test' with agonist, antagonist and inverse agonist images. 
test_Agonista_dir=os.path.join(test_dir, 'Agonista')
test_Antagonista_dir=os.path.join(test_dir, 'Antagonista')

#-----------------------------------------------------------
#Construction and training of the  convolutional neural 
#network
#-----------------------------------------------------------
callbacks = myCallback()
time_callback = TimeHistory()

model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
          optimizer='rmsprop',
          metrics=['accuracy'])

history=model.fit(train_generator,
                 validation_data=validation_generator,
                 epochs=30,
                 callbacks=[callbacks,time_callback])

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch.
#-----------------------------------------------------------
acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]

epochs   = range(len(acc)) # Get number of epochs

#-----------------------------------------------------------
# Plot training and validation accuracy per epoch.
#-----------------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()
plt.show()

#-----------------------------------------------------------
# Plot training and validation loss per epoch.
#-----------------------------------------------------------
plt.plot  ( epochs,     loss )
plt.plot  ( epochs, val_loss )
plt.title ('Training and validation loss'   )
plt.show()

