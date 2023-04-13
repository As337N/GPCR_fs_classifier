import os
from itertools import product
from urllib import request
import pickle

def mk_dir_try(ruta):
    """
    Genera el directorio para la 'ruta' especificada, en caso de que no exista.
    :param ruta: Ruta del directorio a crear.
    :return: No regresa nada.
    """
    if os.path.isdir(ruta) == False:
        try:
            os.mkdir(ruta)
            print(f"Successfully created the PDB directory {ruta} ")
        except:
            print(f"Creation of the PDB directory {ruta} failed")

def get_paths(ruta, lista1, lista2):
    """
    Genera todos los directorios posibles de una combinación de dos listas ('lista1', 'lista2') y una ruta base.
    :param ruta: La ruta base.
    :param lista1: Primera lista de carpetas para la ruta base.
    :param lista2: Segunda lista de carpetas que estaran dentro de las carpetas de la primera lista.
    :return: Una lista con todas las rutas generadas.
    """
    model_sets = list(map(lambda x: ruta + x, lista1))
    paths = list(map(lambda x: x[0] + x[1], product(model_sets, lista2)))
    return model_sets+paths

def download_data(data_set, save_route, url):
    """
    Descarga todos los elementos de un iterable 'data_set' y los guarda en 'save_route'
    :param data_set: Iterable (set) que contiene todos los elementos a descargar.
    :param save_route: Ruta donde se guardaran todos los elementos descargados.
    :param url: url con la cual se descargaran los archivos.
    :return: True.
    """
    for file in data_set:
        if file+'.pdb' not in os.listdir(save_route):
            try:
                request.urlretrieve(url + file.lower() + '.pdb',
                                    save_route + file + '.pdb')
                print(f"Se descargo el archivo pdb de: '{file}'")
            except:
                print(f"No se pudo obtener pdb de: '{file}'")
        else:
            print(f"Ya se habia descargado el pdb: '{file}'")
    return True


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

def load_data(saved_route):
    """
    Carga un archivo de la ruta 'saved_route'.
    :param saved_route: Ruta donde se encuentra el archivo a cargar.
    :return: El archivo cargado en formato de diccionario.
    """
    with open(saved_route,'rb') as f:
        data = pickle.load(f)
    return data

def generate_saving_paths(file_path, saving_paths):
    a, b= file_path.split('/')[-4], file_path.split('/')[-3]
    paths = [f'{x}{a}/{b}' for x in saving_paths]
    return paths
