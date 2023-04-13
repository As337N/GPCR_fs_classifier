import os
import CNN
import tensorflow as tf

ruta_dataset_images='.../GPCR_classification/Scripts/Dataset_files/' #Path where files are stored


#-----------------------------------------------------------
#Preparation of the train, validation and test datasets
#-----------------------------------------------------------
train_dir = os.path.join(ruta_dataset_images, 'train')
validation_dir = os.path.join(ruta_dataset_images, 'validation')
test_dir = os.path.join(ruta_dataset_images, 'test')

train_dataset = CNN.load_dataset(train_dir, 150)
validation_dataset = CNN.load_dataset(validation_dir, 150)
test_dataset = CNN.load_dataset(test_dir, 150)

model_0 = CNN.model(seed=42,
                  input_shape=(150,150,3))

history_0 = CNN.train_model(model=model_0,
                            train_data=train_dataset,
                            validation_data=validation_dataset,
                            epochs=30)

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch.
#-----------------------------------------------------------
acc = history_0.history['accuracy']
val_acc = history_0.history['val_accuracy']
loss = history_0.history['loss']
val_loss = history_0.history['val_loss']

epochs = range(len(acc)) # Get number of epochs

CNN.plot_results(val_acc,acc,epochs,'Training and validation accuracy')
CNN.plot_results(val_loss,loss,epochs,'Training and validation loss')

model_preds = model_0.predict(test_dataset)
preds=tf.round(model_preds)

model_results = CNN.calculate_results(y_true=test_dataset[0][1],
                                   y_pred=preds)

print(model_results)
