import data_sample as ds
import os
import files_folders_management as ffm
import files_preparation as fp
import pandas as pd
from tqdm import tqdm

ruta_base = '.../GPCR_classification/Scripts/Dataset_files/' #Path where files are stored

ruta_nombres = ruta_base + 'PDB_Names/'
ruta_dataset_PDBs = ruta_base + 'PDBs/'
folders = ['train/', 'validation/', 'test/']
pdb_classes = ['Agonist/', 'Antagonist/']
PDB_URL = 'https://files.rcsb.org/download/'
meta_url = 'https://gpcrdb.org/services/residues/extended/'

pandas_dataset_ps = ruta_base + 'pandas_dataset/'

states = {'Agonist':0,
          'Antagonist':1}

Names_clases_PDBs = os.listdir(ruta_nombres)
paths = list(map(lambda x: ruta_nombres+x, Names_clases_PDBs))
dataset = list(map(ds.make_sample, paths))
dataset[0].update(dataset[1])
data = dataset[0]

rutas = ffm.get_paths(ruta=ruta_dataset_PDBs, lista1=folders, lista2=pdb_classes)
[ffm.mk_dir_try(x) for x in rutas]

#ffm.download_pdb(rutas)

df_files = []
df_alfa_matrix = []
df_beta_matrix = []
df_aa_vectors = []
df_ps_vectors = []
df_state = []
df_data_split = []

print('---- Creating PD array ----')

files_foldes_paths, file_folder_path, pdb_files_paths = ' ', ' ', []

generar_dataset = True

if generar_dataset:
    for split_folder in os.listdir(ruta_dataset_PDBs):
        files_foldes_paths = ruta_dataset_PDBs + split_folder + '/'
        for files_folder in os.listdir(files_foldes_paths):
            file_folder_path = files_foldes_paths + files_folder + '/'
            pdb_files_paths = list(map(lambda x: file_folder_path + x, os.listdir(file_folder_path)))
            for route in tqdm(pdb_files_paths):
                lineas = fp.get_lines(route)
                cadena = fp.selector_cadena(lineas)

                try:
                    aa_tensor, ps_tensor, pdb_file = fp.get_vectors(route, meta_url)
                    df_aa_vectors.append(aa_tensor)
                    df_ps_vectors.append(ps_tensor)
                except:
                    pdb_file = route.split('/')[-1][:-4].lower()
                    print(f"No se pudo obtener datos para: '{pdb_file}'")

                cord_line_alfa, cord_line_beta = fp.get_matrices(lineas, chain=cadena)

                dist_matrix_alfa = fp.dist_matrix(cord_line_alfa)
                dist_matrix_beta = fp.dist_matrix(cord_line_beta)

                df_files.append(pdb_file)
                df_alfa_matrix.append(dist_matrix_alfa)
                df_beta_matrix.append(dist_matrix_beta)
                df_state.append(split_folder)
                df_data_split.append(states[files_folder])

    pd_data_dict = {'file':df_files,
    'alfa_matrix':df_alfa_matrix,
    'beta_matrix':df_beta_matrix,
    'aa_vectors':df_aa_vectors,
    'ps_vectors':df_ps_vectors,
    'state':df_state,
    'data_split':df_data_split}

    dataset_df = pd.DataFrame(pd_data_dict)

    ffm.save_data(dataset_df, pandas_dataset_ps, 'pd_data_mapas_contactos')

else:
    dataset_df = ffm.load_data(pandas_dataset_ps+'pd_data')
    print("Se cargo el archivo 'pd_data'")

df_train, df_test, df_validation = fp.df_preparation(dataset_df)

ffm.save_data(df_train, pandas_dataset_ps, 'df_train')
ffm.save_data(df_test, pandas_dataset_ps, 'df_test')
ffm.save_data(df_validation, pandas_dataset_ps, 'df_validation')
