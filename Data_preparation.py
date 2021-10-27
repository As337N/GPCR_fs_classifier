from urllib import request
import os
from numpy import random 
import numpy as np
import matplotlib.pyplot as plt

ruta_nombres='' #Folder with txt files that contain the names of the PDBs 
ruta_dataset_PDBs='' #Path where PDB files will be stored
ruta_dataset_images='' #Path where images, distance maps, will be stored
folders=['train\\','validation\\','test\\']
clases=['Agonist\\','Antagonist\\']
PDB_URL='https://files.rcsb.org/download/'

"""We define a function to obtain the distance of each amino acid center
with the others aminoacid centers in the protein. In this project the
amino acid center are the beta carbons of each aminoacid."""
def distancias(x,y):
    d=np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)
    if d >= 30:
        d=30
    return d

#-----------------------------------------------------------
#We create the folders that house the data, they are already
#separated 
#-----------------------------------------------------------
for folder in folders:
    try:
        os.mkdir(ruta_dataset_PDBs+folder)
        print ("Successfully created the PDB directory %s " % folder)
    except OSError:
        print ("Creation of the PDB directory %s failed" % folder)
    try:
        os.mkdir(ruta_dataset_images+folder)
        print ("Successfully created the images directory %s " % folder)
    except OSError:
        print ("Creation of the images directory %s failed" % folder)
    for clase in clases:
        try:
            os.mkdir(ruta_dataset_PDBs+folder+clase)
            print ("Successfully created the PDB directory %s " % clase)
        except OSError:
            print ("Creation of the PDB directory %s failed" % clase)
        try:
            os.mkdir(ruta_dataset_images+folder+clase)
            print ("Successfully created the images directory %s " % clase)
        except OSError:
            print ("Creation of the images directory %s failed" % clase)

#-----------------------------------------------------------
#We obtain the names of the files and classify them in the 
#different categories of folders (train, validtaion and test)
#-----------------------------------------------------------
Names_clases_PDBs=os.listdir(ruta_nombres)
for names_files_PDBs in Names_clases_PDBs:
    print(names_files_PDBs[:-9])
    with open(ruta_nombres+names_files_PDBs,'r')as PDB_ID:
        lineas=PDB_ID.readlines()
        train_ids=random.choice(lineas, size=round(len(lineas)*0.9),replace=False)
        
        l3=[]
        for ID in lineas:
            if ID not in train_ids and ID in lineas:
                l3.append(ID)
        validation_ids=random.choice(l3,size=round(len(lineas)*0.05),replace=False)
        
        test_ids=[]
        for ID2 in l3:
            if ID2 not in validation_ids and ID2 in l3:
                test_ids.append(ID2)
#-----------------------------------------------------------
#From this point, download the PDB files for each category 
#(train, test and validation, as well as agonist and antagonist). 
#
#The folders are as follows: 
#
#Data--->train--->Agonist
#     |        |->Antagonist
#     |
#     |->test--->Agonist
#     |       |->Antagonist
#     |
#     |->validation--->Agonist
#                   |->Antagonist
#-----------------------------------------------------------

        for PDB in train_ids:
            file_name=PDB.strip('\n')
            request.urlretrieve(PDB_URL+file_name.lower()+'.pdb', 
                                ruta_dataset_PDBs+'train\\'+names_files_PDBs[:-9]+'\\'+file_name+'.pdb')      
        
        for PDB in validation_ids:
            file_name=PDB.strip('\n')
            request.urlretrieve(PDB_URL+file_name.lower()+'.pdb', 
                                ruta_dataset_PDBs+'validation\\'+names_files_PDBs[:-9]+'\\'+file_name+'.pdb') 
    
        for PDB in test_ids:
            file_name=PDB.strip('\n')
            request.urlretrieve(PDB_URL+file_name.lower()+'.pdb', 
                                ruta_dataset_PDBs+'test\\'+names_files_PDBs[:-9]+'\\'+file_name+'.pdb') 

#We print the distribution of the files in the different categories. 
            
        print(""""
        train: {}
        validation: {}
        test: {}""".format(train_ids,validation_ids,test_ids))

#We create the distance maps and store them in the corresponding section. 

Names_folders_PDBs=os.listdir(ruta_dataset_PDBs)

for folder in Names_folders_PDBs:
    Names_categorias_PDBs=os.listdir(ruta_dataset_PDBs+folder)
    for categoria in Names_categorias_PDBs:
        PDB=os.listdir(ruta_dataset_PDBs+folder+'\\'+categoria)
        for file in PDB:
            if file[-4:] =='.pdb':
                with open(ruta_dataset_PDBs+folder+'\\'+categoria+'\\'+file,'r') as pdb:
                    cord=[]
                    res=[]
                    for line in pdb:
                        cord_line=[]
                        if line[:4] == 'ATOM' and line[13:15]=='CB'and float(line[23:26]) not in res:
                            res.append(float(line[23:26]))
                            cord_line.append(int(line[23:26]))
                            cord_line.append(float(line[32:38]))
                            cord_line.append(float(line[40:46]))
                            cord_line.append(float(line[47:54]))#Residuo,X,Y,Z
                            cord.append(cord_line)
                    cont_matrix=np.zeros((int(max(res))+1,int(max(res))+1))
                    #c=0
                    for x in cord:
                        for y in cord:
                            cont_matrix[x[0]][y[0]]=distancias(x[1:],y[1:])
                            #if c<15:
                                #c+=1
                    plt.style.use('classic')
                    plt.axis('off')
                    plt.imshow(cont_matrix)
                    plt.savefig(ruta_dataset_images+folder+'\\'+categoria+'\\'+file[:-4]+'.png',transparent=True)
