import os
import keras
from keras.preprocessing.image import load_img, img_to_array 
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import itertools

"""This function translate the binary predicted label, 'i', into categorical
labels."""
def Nombres(i):
    if np.all(i == np.array([1.])):
        pred='Antagonist'
    elif np.all(i == np.array([0.])):
        pred='Agonist'
    else:
        pred='NA'
    return(pred)

"""This function predicts categorical labels for distance maps
stored in 'path' and returns a dictionary with the name of the PDB and the
predicted activity of the ligand that binds to the protein. """ 
def prueba(ruta):
    num=[]
    predicciones={}
    for file in os.listdir(ruta):
        if file[-4:] =='.png':
            img = load_img(ruta+'\\'+file)
            img = img.resize((150, 150))
            img = img_to_array(img)
            img = img.reshape( -1,150, 150,3)
            pred=new_model.predict(img)
            num.append(np.round(pred[0]))
            predicciones[file[:-4]]=Nombres(np.round(pred[0]))
    return predicciones

"""This function make the confussion matrix for the predicted values. It takes the true labels ('y_true'), the predicted 
labels (y_pred). Also the function can take a list with the names of the classes ('classes') that has a default value of 
'None', a 2 element tupple with the dimensions for the image ('figsize'), that by default has a value of (10,10) and a 
integer with the size of the text elements, except for the names of the classes, ('text_size')."""
def make_confusion_matrix(y_true, y_pred, classes=None,figsize=(10,10),text_size=15):
    #Create the confussion matrix
    cm=confusion_matrix(y_true,tf.round(y_pred))
    cm_norm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]#Normalize confusion matrix
    n_classes=cm.shape[0]

    #Let's prettify it
    fig,ax=plt.subplots(figsize=figsize)
    #Create a matrix plot
    cax=ax.matshow(cm,cmap=plt.cm.Blues)
    cb=fig.colorbar(cax)
    cb.set_label(label='Frequency', size=text_size)

    #Set labels to be classes
    if classes:
        labels=classes
    else:
        labels=np.arange(cm.shape[0])

    #Label the axes
    ax.set(xlabel='Predicted label',
          ylabel='True label',
          xticks=np.arange(n_classes),
          yticks=np.arange(n_classes),
          xticklabels=labels,
          yticklabels=labels)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)

    #Set x-axis labels to bottom
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    #Adjust label size
    ax.yaxis.label.set_size(text_size)
    ax.xaxis.label.set_size(text_size)
    #ax.title.set_size(text_size)

    #Set threshold for different colors
    threshold = (cm.max()+cm.min())/2.

    #Plot the text on each cell
    for i, j in itertools.product (range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,f'{cm[i,j]}({cm_norm[i,j]*100:.1f}%)',
        horizontalalignment='center',
        color='white' if cm[i,j]>threshold else 'black',
                size=text_size)

    fig.savefig(Saving_route)

Saving_route='' #Path where the confusion matrix will be saved.
ruta_NN='' #Path where the model is saved.
new_model = keras.models.load_model(ruta_NN)

ruta_dataset_images='' #Path where the distance maps we want to classify are stored.
test_dir = os.path.join(ruta_dataset_images, 'test')
# 'test' directory with the agonist and antagonist images. 
test_Agonista_dir=os.path.join(test_dir, 'Agonist')
test_Antagonista_dir=os.path.join(test_dir, 'Antagonist')

#We make and visualize the predictions of the model.
print(prueba(test_Agonista_dir),prueba(test_Antagonista_dir))

#In the following lines we load the distance maps to the model, with this we can make predictions on the distace maps.
ruta_test=os.path.join(ruta_dataset_images,'test')#Path of the 'test' dataset
test_datagen=ImageDataGenerator(rescale=1/255.)

test_generator=test_datagen.flow_from_directory(ruta_test,
                                                     class_mode='binary',
                                                     target_size=(150,150))

#We split the dataset into the features, X_test, and the labels, Y_test.
X_test=test_generator[0][0]
Y_test=test_generator[0][1]

y_pred=new_model.predict(X_test)#Make the predictions

#We evaluate the performance of the model in the 'test' set.
new_model.evaluate(X_test,Y_test)

#At last, we make the confusion matrix.
make_confusion_matrix(Y_test, y_pred, classes=['Antagonist','Angonist'],figsize=(10,10),text_size=23)
