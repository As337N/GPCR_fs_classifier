import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt

ruta_guardado_NN='\Models' #Path where the models will be saved.

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

def load_dataset(path, dim):
    datagen = ImageDataGenerator(rescale=1/255.)
    generator = datagen.flow_from_directory(path,
                                            class_mode='binary',
                                            target_size=(dim,dim))
    return generator

def model(seed, input_shape):
    tf.random.set_seed(seed)
    inputs = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation='relu')(inputs)
    y = MaxPooling2D(2)(y)
    y = Conv2D(32, (3, 3), activation='relu')(y)
    y = MaxPooling2D(2, 2)(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D(2, 2)(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(32, activation='relu')(y)
    y = Dense(16, activation='relu')(y)
    y = Dropout(0.45)(y)
    outputs = Dense(1, activation='sigmoid')(y)
    model = Model(inputs, outputs)
    return model

def model(seed, input_shape):
    tf.random.set_seed(seed)
    inputs = Input(shape=input_shape)
    y = Conv2D(32, (3, 3), activation='relu')(inputs)
    y = MaxPooling2D(2)(y)
    y = Conv2D(64, (3, 3), activation='relu')(y)
    y = MaxPooling2D(2, 2)(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(16, activation='relu')(y)
    y = Dropout(0.45)(y)
    outputs = Dense(1, activation='sigmoid')(y)
    model = Model(inputs, outputs)
    return model

def train_model(model, train_data, validation_data, epochs):
    callbacks = myCallback()
    time_callback = TimeHistory()

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    history = model.fit(train_data,
                        validation_data=validation_data,
                        epochs=epochs,
                        callbacks=[callbacks, time_callback])

    return history

def calculate_results(y_true, y_pred):
    '''
    Calculates model accuracy, precision, recall and f1 score of a binary classification
    '''

    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    # Calculate model precision, recall and f1-score using weighted average
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true,
                                                                                 y_pred,
                                                                                 average='weighted')
    model_results = {'accuracy': model_accuracy,
                     'precision': model_precision,
                     'recall': model_recall,
                     'f1': model_f1}
    return model_results

def plot_results(val_data,data, epochs, title):
    plt.plot(epochs, data)
    plt.plot(epochs, val_data)
    plt.title(title)
    plt.figure()
    plt.show()
