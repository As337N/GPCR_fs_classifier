# GPCR_fs_classifier
## Introduction
GPCR_fs_classifier is a convolutional neural network (CNN) that can classify functional states for G protein-coupled receptors (GPCRs) from the rhodopsin family (class A).

We present a reliable convolutional neural network-based algorithm capable of classifying functional states of G Protein-Coupled Receptors (GPCRs). Classification of GPCR functional states is of great interest in pharmacology due to their relevance as drug targets and abundance in human cells. Our algorithm employs a novel protein representation called RRCS Ballesteros Matrices (RBM) calculated with the RCSB PDB and GPCRdb databases. Employing the RBM representation, our convolutional neural network (Model A) was able to classify a GPCR in its two most frequent functional states: those conducted by the presence of an agonist or an antagonist ligand. We tested our algorithm in an unknown dataset of 48 GPCR PDB structures. As a result, the proposed network successfully classified the structures, achieving accuracy and loss of 95.83% and 0.33, respectively. Also, we obtained an f1-score of 95.86% for this model, despite the limitations due to the scarce data. Moreover, we developed a second model (Model B) with similar accuracy but were able to identify the most relevant contacts from the GPCR that are also related to experimental information. With this, we identified the second extracellular loop (ECL2) and the extracellular sections of transmembrane helix 4 (TM4) and 5 (TM5) helices as key regions employed by our algorithm to characterize the functional states of GPCRs.

## Provided material description

This GitHub repository contains all of the materials needed to replicate this work.You can find the following files in this repository:

### Folders with code

 * CNN - The CNN model to train the network.
 * Grad_CAM - The required functions to obtain the GradCAM maps and the corresponding analysis.
 * Images_preprocessing - The functions required to preprocess the PDBs and prepare the datasets for the model.

### Dataset_files

 * Dataset_files_2.zip - Extra files for the CAM maps visualization and analysis
 * Dataset_files_RBM.zip - All files related to the RBM representation.
 * PDBs_sets - The folder that contains the .zip files for the dataset. 

You need to unzip the dataset in the following way:

Dataset_files
      |
      |              Dataset_files_2.zip
      |
      |----Datasets
      |----Dicts_RRCS_contacts
      |----Grad_CAM_matrix
      |----Grad_CAM_resultados
      |----PDB_Names
      |----Resultados_analisis_CAM
      |----saved_model
      |
      |              Dataset_files_RBM.zip
      |
      |----Ballest_matrix_images
      |----Ballest_matrix
      |----RRCS_matrix_images
      |----RRCS_matrix
      |
      |              PDBs_sets
      |
      |----PDBs--->train--->Agonist
            |            |->Antagonist
            |
            |->test--->Agonist
            |       |->Antagonist
            |
            |->validation--->Agonist
                          |->Antagonist


Also, you can use Images_preprocessing folder scripts and you are going to obtain the same files. 
