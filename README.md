# GPCR_fs_classifier
## Introduction
GPCR_fs_classifier is a convolutional neural network (CNN) that can classify functional states for G protein-coupled receptors (GPCRs) from the rhodopsin family (class A).

This algorithm uses distance maps, obtained from PDB structural information, to represent the structural conformation of the GPCRs. The CNN takes the distance maps and returns a binary classification, agonist or antagonist, related to the activity of the ligand bound to the GPCR. The accuracy of the model reached 94.12% in the validation dataset and 93.33% in the test set, with losses of 0.18 and 1.23 respectively.

With this activity we can 

## Provided material description

This GitHub repository contains all of the materials needed to replicate this project.You can find the following files in this repository:

### Files with code

 * Data_preparation.py - This script prepares the training data, obtains the distance maps from the selected PDB files, and separates the images into the respective classes.
 * CNN.py - In this script, the CNN is prepared and trained. If a model obtains an accuracy greater than 92.0% and a loss smaller than 0.2, the model is saved.
 * PDB_classifier - Here you can find the code to use the saved model and predict the ligand activity classification on the test set. Moreover, you can obtain a confussion matrix to evaluate the model performance.

### Suplementary files

 * Name_files.zip - In this zip you can find the names of all the PDB structural files that we used in this proyect, they are in txt files separated by their classification. We only use the agonist and antagonist ligand activities.
 * red_V02_0.18.tf.zip - This zip includes a stored model that we obtained from CNN and that we used for the classifications.

### Dataset

 * Train_Agonista.zip - The training files for the agonist class.
 * Train_Antagonista.zip - The training files for the antagonist class.
 * validation.zip - The validation files. They are already classified into agonist and antagonist.
 * test.zip - The test files are already classified into agonist and antagonist.

You need to unzip the dataset in the following way:

       Data--->train--->Agonist
            |        |->Antagonist
            |
            |->test--->Agonist
            |       |->Antagonist
            |
            |->validation--->Agonist
                          |->Antagonist


Also, you can use Data_preparation.py and you are going to obtain the same files. 
