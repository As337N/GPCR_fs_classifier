import requests
import pandas as pd
import numpy as np
import tensorflow as tf

AA = {'A':1, 'R':2, 'N':3, 'D':4, 'C':5,
      'Q':6,'E':7,'G':8,'H':9,'I':10,'L':11,
      'K':12,'M':13,'F':14,'P':15,'S':16,
      'T':17,'W':18,'Y':19,'V':20}

protein_segment = {'TM3':1,'TM6':2,'TM1':3,'N-term':4,
                   'TM2':5,'TM5':6,'TM4':7,'ECL2':8,
                   'TM7':9,'C-term':10,'H8':11,'ICL3':12,
                   'ICL2':13,'ECL3':14,'ECL1':15,'ICL1':16}

def get_lines(file_route):
    with open(file_route, 'r') as pdb:
        lines = pdb.readlines()
    return lines

def get_xyz_lines_info(line, res, chain, cord_line_alfa, cord_line_beta):
    if line[:4] == 'ATOM' and line[13:15] == 'CA' and line[20:23] == chain:
        res.append(int(line[22:26]))
        cord_line_alfa[int(line[22:26])] = [float(line[30:38]), float(line[40:46]), float(line[47:54])]
    if line[:4] == 'ATOM' and line[13:15] == 'CB' and line[20:23] == chain:
        cord_line_beta[int(line[22:26])] = [float(line[30:38]), float(line[40:46]), float(line[47:54])]
    return cord_line_alfa, cord_line_beta

def get_meta_info(url, meta_info, max_val, encode_dict):
    res = requests.get(url).json()
    df = pd.DataFrame(res)
    vector_info = [encode_dict.get(x, max_val + 1) for x in df[meta_info]]
    pos_vector = df["sequence_number"]
    vector_zeros = np.zeros([max(pos_vector) + 1])
    for i, j in zip(pos_vector, vector_info):
        vector_zeros[i] = j
    return vector_zeros/(max_val+1)

def get_matrices(lineas, chain):
    res = []
    cord_line_beta = {}
    cord_line_alfa = {}
    for line in lineas:
        cord_line_alfa, cord_line_beta = get_xyz_lines_info(line, res, chain, cord_line_alfa, cord_line_beta)
    return cord_line_alfa, cord_line_beta

def distancias(vect1, vect2):
    dist = np.sqrt((vect1[0]-vect2[0])**2 + (vect1[1]-vect2[1])**2 + (vect1[2]-vect2[2])**2)
    if dist < 15:
        distancia = int(round(dist))
    else:
        distancia = 0
    return distancia

def get_vectors(path, url):
    pdb_file = path.split('/')[-1][:-4].lower()
    url_request = url + pdb_file
    aa_tensor = get_meta_info(url_request, 'amino_acid', 20, AA)
    ps_tensor = get_meta_info(url_request, 'protein_segment', 16, protein_segment)

    return aa_tensor, ps_tensor, pdb_file

def dist_matrix(coord_matrix):
    size = max(coord_matrix.keys())
    matrix = np.zeros((size+1, size+1))
    for i in coord_matrix.keys():
        for j in coord_matrix.keys():
            matrix[i][j] = distancias(coord_matrix[i], coord_matrix[j])
    tensor = tf.convert_to_tensor(matrix, dtype=tf.float32)
    return tensor

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def selector_cadena(lineas, max_len=0):
    cadenas = {}
    for linea in lineas:
        if linea[:4] == 'ATOM' and linea[20:23] not in cadenas:
            cadenas[linea[20:23]] = 1
        elif linea[:4] == 'ATOM' and linea[20:23] in cadenas:
            cadenas[linea[20:23]] += 1
    max_len = max(cadenas.values())
    cadena = get_keys_from_value(cadenas, max_len)[0]
    return cadena

def pad_matrix(tensor, max_value):
    shape = int(max_value - len(tensor))
    padding = tf.constant([[0, shape], [0, shape]])
    pad_tensor = tf.pad(tensor, padding, 'CONSTANT')
    return pad_tensor

def get_max_len(df, column):
    lens_aa = [len(i) for i in df[column]]
    max_len = np.percentile(lens_aa, 95)
    df['lens_aa_chain'] = lens_aa
    df_0 = df.loc[df['lens_aa_chain'] < max_len]
    df_0[column] = [pad_matrix(x, max_value=max_len) for x in df_0[column]]
    return df_0

def df_preparation(df):
    df_0 = get_max_len(df, column='beta_matrix')
    df_0 = get_max_len(df_0, column='alfa_matrix')
    df_train = df_0.loc[df_0['state'] == 'train']
    df_test = df_0.loc[df_0['state'] == 'test']
    df_validation = df_0.loc[df_0['state'] == 'validation']
    return df_train, df_test, df_validation
