import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2

clases = {0:'Agonist', 1:'Antagonist'}

def load_data(saved_route):
    """
    Carga archivos pkl.
    :param saved_route: Ruta de guardado del archivo pkl.
    :return: Los elementos guardados en el archivo pkl.
    """
    with open(saved_route, 'rb') as f:
        data = pickle.load(f)
    return data

def codificador_clase(clave):
    """
    Codifica la actividad para ligandos de los GPCR en etiquetas numéricas.
    :param clave: La string que define la actividad del ligando.
    :return: Una etiqueta int que codifica la actividad del ligando.
    """
    if clave == 'Agonist':
        return 0
    elif clave == 'Antagonist':
        return 1

def get_datasets(path, semilla=4):
    """
    Prepara los datasets para el modelo al generar tuplas con los datos de entrada (x), los datos de
    salida (y) y el nombre de los archivos PDB. Tambien mezcla de forma pseudoaleatoria al dataset.
    :param path: La ruta donde se encuentra el dataset a extraer.
    :param semilla: La semilla para la mezcla pseudoaleatoria del dataset.
    :return: Una lista de tuplas (x, y, nombre_pdb)
    """
    z = []
    x = []
    y = []
    for i in os.listdir(path):
        for j in os.listdir(path+i):
            x.append(load_data(f'{path}{i}/{j}'))
            y.append(codificador_clase(i))
            z.append(j[:-4])
    array = list(zip(x, y, z))
    random.seed(semilla)
    random.shuffle(array)
    return array

def prepare_dataset(path_dataset):#TODO: Revisar esta función, me parece es redundante con get_datasets()
    """
    Prepara las entradas x e y del modelo, tambien regresa el nombre del archivo pdb para cada caso.
    :param path_dataset: La ruta donde se encuentra el dataset a preparar.
    :return:
    """
    set_split = get_datasets(path_dataset)
    x, y, pdb_names = list(zip(*set_split))
    final_x = np.array(x)
    final_y = np.array(y)
    return final_x, final_y, pdb_names

def get_grads(model, last_conv_layer, pred_dataset, last_layer):
    """
    Obtiene los 'grads' para cada predicción del modelo.
    :param model: Modelo CNN a evaluar.
    :param last_conv_layer: Nombre de la última capa de convolución del modelo.
    :param pred_dataset: Dataset del que queremos obtener sus 'grads'.
    :param last_layer: Nombre de la última capa del modelo.
    :return: Matriz con los pesos de los 'grads'.
    """
    grad_model = tf.keras.models.Model(
    [model.inputs],
        [model.get_layer(last_conv_layer).output,
         model.get_layer(last_layer).output,
         model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds, final_preds = grad_model(pred_dataset)
    grads = tape.gradient(preds, last_conv_layer_output)
    return grads, last_conv_layer_output, final_preds

def get_heatmap(grads, last_conv_layer_output, shape):
    """
    Obtiene la matriz 'heatmap', a partir de la matriz de 'grads', que nos indica las secciones relevantes para
    cada matriz de entrada.
    :param grads: Matriz con valores de los grads.
    :param last_conv_layer_output: Nombre de la última capa de convolución del modelo.
    :param shape: Tamaño de las matrices, que corresponde con el número de secciones del GPCR consideradas.
    :return: Matriz 'heatmap'.
    """
    pooled_grads= tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (shape, shape))
    return heatmap

def prepare_tensor_input (input_array):
    """
    Prepara las matrices x del dataset como tensores para ser inputs del modelo CNN.
    :param input_array: Matriz x a ser preparada.
    :return: Matriz x en formato de tensor.
    """
    new_input = tf.expand_dims(input_array, axis=0)
    new_input = tf.expand_dims(new_input, axis=-1)
    new_input = tf.cast(new_input, tf.float32)
    tensor_input = tf.Variable(new_input, trainable=True)
    return tensor_input

def save_data(data, save_route):
    with open(save_route, 'wb') as f:
        pickle.dump(data, f)
    return True

def get_gradCAM_map(x, y, file_name, modelo, shape, last_conv_layer, last_layer_model,ruta_guardado_matrices, ruta_guardado_plt):
    """
    Obtiene los mapas GradCAM para un dataset.
    :param x: Valores x del dataset a procesar.
    :param y: Valores y del dataset a procesar.
    :param file_name: Nombre del PDB.
    :param modelo: Modelo CNN que se esta evaluando.
    :param shape: Tamaño de las matrices, que corresponde con el número de secciones del GPCR consideradas.
    :param last_conv_layer: Nombre de la última capa de convolución del modelo CNN.
    :param last_layer_model: Nombre de la última capa del modelo CNN.
    :param ruta_guardado_matrices: Ruta de guardado para las matrices pkl GradCAM generadas.
    :param ruta_guardado_plt: Ruta de guardado para las imágenes de las matrices GradCAM generadas.
    """
    real_y = np.array(y).reshape(1, )
    real_value = tf.expand_dims(real_y, axis=0)

    input_data = prepare_tensor_input(x)
    grads, last_conv_layer_output, final_preds = get_grads(model=modelo,
                                                           last_conv_layer=last_conv_layer,
                                                           pred_dataset=input_data,
                                                           last_layer=last_layer_model)
    heatmap = get_heatmap(grads, last_conv_layer_output, shape)
    save_data(heatmap, ruta_guardado_matrices + file_name + '.pkl')

    loss = tf.keras.losses.binary_crossentropy(real_value, final_preds[0])

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    axes.imshow(heatmap, cmap='YlOrRd')
    axes.set_title(
        f'Archivo: {file_name}, clase: {clases[y]}, pred: {clases[np.round(final_preds[0][0])]}, loss: {loss}')
    fig.tight_layout()
    fig.savefig(ruta_guardado_plt + file_name + '.png')

def load_cam_maps(ruta):
    """
    Carga los mapas CAM para ser procesados.
    :param ruta: La ruta donde se encuentran los mapas CAM.
    :return: Un diccionario {pdb_name : mapa CAM}.
    """
    data = {}
    for file in os.listdir(ruta):
        cam_matrix = load_data(ruta+file)
        data[file[:-4]] = cam_matrix
    return data


def save_data(data, save_route, name):
    """
    Guarda los datos de Python en la ruta especificada 'save_route' con el
    nombre del archivo 'name'.
    :param data: Datos dse python a guardar.
    :param save_route: Ruta de guardado.
    :param name: Nombre bajo el cual se guardara el archivo.
    :return: True
    """
    with open(save_route+'/'+name,'wb') as f:
        pickle.dump(data,f)
    print(f"Se guardo el archivo: '{name}'")
    return True

def blend_maps(GradCAM_matrix_save_path, save_route, nombre):
    """
    Combina los mapas CAM por clase.
    :param GradCAM_matrix_save_path: Ruta de guardado de los mapas CAM para la clase a combinar.
    :return: El mapa CAM combinado por clase.
    """
    data_dict = load_cam_maps(GradCAM_matrix_save_path)
    total_matrices = len(data_dict)
    first_matrix_key = list(data_dict.keys())[0]
    first_matrix_shape = data_dict[first_matrix_key].shape
    new_matrix = np.zeros(first_matrix_shape)
    for matrix in data_dict.values():
        new_matrix += matrix / total_matrices
    save_data(new_matrix, save_route, name=nombre)
    return new_matrix
