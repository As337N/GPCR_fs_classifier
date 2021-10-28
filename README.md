# GPCR_fs_classifier
GPCR_fs_classifier is a convolutional neural network (CNN) that can classify functional states for G protein-coupled receptors (GPCRs) from the rhodopsin family (class A).

This algorithm uses distance maps to represent the structural conformation of the GPCRs. The CNN take the distance maps and return a binary classification related to the activity of the ligand bound to the GPCR.

In this GitHub we provide all the material required to replicate this proyect. You can find the following files in this repository:

  Code files
  * Data_preparation.py - This script prepare the training data, obtaining the distance maps from the selected PDB files, also it separate the images into the respectively classes.
  * CNN.py - In this script the CNN is prepared and trained, also if a model obtain an accuracy grether than 92.0% and a loss smaller than 0.2, the model is saved.
  * PDB_classifier - Here you can find the code to use the saved model and predict the ligand activity classification on the test set. Moreover, you can obtain a confussion matrix to evaluate the model performance.

  Suplementary files
  * Name_files.zip - In this zip you can find the name of all the PDB structural files hat we use in this proyect, they are in txt files separated by their classification. We only use the agonist and antagonist ligand activities.
  * red_V02_0.18.tf.zip - This zip includes a stored model that we obatin from the CNN and thah we used for the classifications.

  Dataset
  * Train_Agonista.zip - The training files for the agonist class.
  * Train_Antagonista.zip - The training files for the antagonist class.
  * validation.zip - The validation files, they are already classified into agonist and antagonist.
  * test.zip - The test files, they are already classified into agonist and antagonist.

You need to unzip the dataset in the following way:

    Data--->train--->Agonist
         |        |->Antagonist
         |
         |->test--->Agonist
         |       |->Antagonist
         |
         |->validation--->Agonist
                       |->Antagonist
                       
Also you can use Data_preparation.py and you are going to obtain the same files. 
