import random
import os
import files_folders_management as ffm

ruta_base = '.../GPCR_classification/Scripts/Dataset_files/' #Path where files are stored

save_route = ruta_base + 'Datasets/'

def make_dict_data(keys, values, file):
    """
    Genera un diccionario de dos iterables (en formato de lista), 'keys' y 'values'. Al mismo
    tiempo guarda el diccionario en la ruta especificada 'save_route' bajo el nombre 'file'.
    :param keys: Lista de llaves para el diccionario.
    :param values: Lista de valores para el diccionario.
    :param file: Nombre bajo el cual se guardará el diccionario.
    :return: El diccionario generado.
    """
    dict_data = {key:value for key,value in zip(keys, values)}
    ffm.save_data(dict_data, save_route, f'datasets_{file}')
    return dict_data

def make_sample(ruta, train_lenght=0.8):
    """
    Selecciona los grupos de entrenamiento, validación y prueba ('train_ids', 'val_ids',
    'test_ids'). De ser necesario guarda la selección en un archivo txt, pero si ya existen los
    txt con previas selecciones carga dichos archivos en formato de diccionario
    {grupo:pdbs_seleccionados}.
    :param ruta: Ruta de donde se encuentran todos los archivos para cada etiqueta (agonista,
                 antagonista, etc.).
    :param train_lenght: Tamaño del grupo de entrenamiento en porcentaje.
    :return file: Nombre de la etiqueta (agonista, antagonista, etc.)
    :return dict_dataset: Diccionario con las selecciones de pdbs para los grupos train,
                          text y validation, {grupo:pdbs_seleccionados}.
    """
    file = ruta.split('/')[-1][:-4]
    if not f'datasets_{file}' in os.listdir(save_route):
        with open(ruta, 'r') as dataset:
            lineas = set(map(lambda x: x.strip('\n'), dataset.readlines()))
            train_ids = set(random.sample(lineas, round(len(lineas) * train_lenght)))
            val_test_sets = lineas - train_ids
            val_ids = set(random.sample(val_test_sets, round(len(val_test_sets) * 0.5)))
            test_ids = val_test_sets - val_ids
            keys, values = ['train', 'validation', 'test'], [train_ids, val_ids, test_ids]
        dict_dataset = make_dict_data(keys, values, file)
    else:
        dict_dataset = ffm.load_data(save_route + f'datasets_{file}')
        print(f"Se cargo el archivo: '{file}'")
    return {file: dict_dataset}
