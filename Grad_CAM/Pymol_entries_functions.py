import math
import pandas as pd
import requests
import re
import numpy as np
import Hauser_matrix_functions as hmf
import itertools

meta_url = 'https://gpcrdb.org/services/residues/extended/'

gpcr_keys_1 = ['ICL1','ICL2','ICL3','ECL1','ECL2','ECL3','C-term','N-term']


def get_meta_df(url, file_path):
    """
    Obtiene los metadatos para cada GPCR, mediante la GPCRdb.
    :param url: URL donde consulta los datos de GPCRdb.
    :param file_path: Ruta donde se encuentra alojado el archivo estructural, para obtener el nombre del GPCR.
    :return: Un dataframe que contiene los metadatos para cada GPCR a procesar.
    """
    pdb_name = file_path.split('/')[-1][:-4].lower()
    url_pdb = url + pdb_name.lower()
    res = requests.get(url_pdb).json()
    df = pd.DataFrame(res)
    reduced_df = df[['sequence_number',
                     'amino_acid',
                     'protein_segment',
                     'display_generic_number']]
    reduced_df['display_generic_number'].fillna('Nan', inplace=True)
    return reduced_df


def df_2_list_of_lists(reduced_df):
    """
    Convierte el dataframe de metadatos en una lista de columnas (Row_list).
    :param reduced_df: El dataframe de metadatos.
    :return: La Row_list generada.
    """
    Row_list = []
    for index, rows in reduced_df.iterrows():
        append_list = [rows.sequence_number,
                       rows.amino_acid,
                       rows.protein_segment,
                       rows.display_generic_number]
        Row_list.append(append_list)
    Row_list = obtain_ballest_repr(Row_list)
    return Row_list


def selector_representacion_ballesteros(key_helix, ballest_index):
    """
    Genera las nuevas representaciones para los fragmentos de las helices de los GPCR.
    :param key_helix: Nombre de la helice a buscar.
    :param ballest_index: Indice de ballesteros a buscar.
    :return: Nueva representación para cada aminoacido en las helices del GPCR.
    """
    if key_helix in hmf.gpcr_keys_0.keys():
        for key, value in hmf.gpcr_keys_0[key_helix].items():
            try:
                if ballest_index in key:
                    return key_helix + value
            except:
                if ballest_index == key:
                    return key_helix + value
    elif key_helix in gpcr_keys_1:
        return key_helix


def make_ballest_index(generic_number):
    """
    Genera los índices de Ballesteros a partir de los números genéricos de GPCRdb
    :param generic_number: Número generico de GPCRdb.
    :return: Indice de Ballesteros
    """
    if generic_number != 'Nan':
        list_generic = re.split('\.|x', generic_number)
        return int(list_generic[-1])


def obtain_ballest_repr(Row_list):
    """
    Para cada elemento en la Row_list genera la representación de los fragmentos del GPCR.
    :param Row_list: Lista de columnas con los metadatos del GPCR.
    :return: Nueva lista de columnas (Row_list), donde el último valor ahora es la representación de fragmentos para
             el GPCR.
    """
    for aa in Row_list:
        repr_ballesteros = selector_representacion_ballesteros(aa[-2], make_ballest_index(aa[-1]))
        aa.append(repr_ballesteros)
    return Row_list


def filtrar_contactos_relevantes(mapa, filtro1, filtro2):
    """
    Filtra solo los contactos del mapa CAM que se encuentren entre las cotas establecidas pro el
    filtro1 y el filtro2.
    :param mapa: Mapa CAM a filtrar.
    :param filtro1: Cota inferior del filtro.
    :param filtro2: Cota superior del filtro.
    :return: Mapa CAM filtrado con los contactos relevantes de acuerdo a las cotas.
    """
    coord_filtro = []
    coordenadas_filtro_tupla = np.where((filtro2 >= mapa) & (mapa > filtro1))
    coord_x, coord_y = coordenadas_filtro_tupla
    for x, y in zip(coord_x, coord_y):
        coord_filtro.append([x, y])
    return coord_filtro


def get_claves_matriz(lista_coordenadas, diccionario_claves):
    """
    Genera una lista que almacena los contactos establecidos, a partir de un diccionario con coordenadas y una
    lista con contactos.
    :param lista_coordenadas: Lista con los residuos que establecen contacto.
    :param diccionario_claves: Diccionario que almacena las coordenadas de los residuos.
    :return: Lista con las coordenadas de los residuos que establecen contacto.
    """
    coords = []
    keys = list(diccionario_claves.keys())
    values = list(diccionario_claves.values())
    for coord in lista_coordenadas:
        index_x = values.index(coord[0])
        index_y = values.index(coord[1])
        key_x = keys[index_x]
        key_y = keys[index_y]
        coords.append((key_x, key_y))
    return coords


def get_secciones_relevantes(mapa, cota_inferior=0.5):
    """
    Encuentra las secciones que establecen contactos, de acuerdo a su relevancia en los mapas CAM.
    :param mapa: Mapa a evaluar.
    :param cota_inferior: Valor mínimo de los mapas CAM desde el que se evaluara la relevancia.
    :return: Diccionario {rango evaluado : secciones del GPCR con contactos relevantes en el rango evaluado}.
    """
    secciones = {}
    val = np.max(mapa) * 10
    frontera = math.ceil(val)
    c = cota_inferior * 10
    while c < frontera:
        cota_inf = c / 10
        cota_sup = (c + 1) / 10
        contactos = filtrar_contactos_relevantes(mapa, cota_inf, cota_sup)
        secciones[(cota_inf, cota_sup)] = get_claves_matriz(contactos, hmf.coords_matrix_ballesteros)
        c += 1
    return secciones


def get_row_list_protein(file_route_pdb):
    """
    Obtiene la información de los residuos del PDB correspondiente a 'file_route_pdb'.
    :param file_route_pdb: Ruta url del PDB a analizar.
    :return: Lista con la información de los residuos del PDB.
    """
    reduced_pdb_df = get_meta_df(url=meta_url, file_path=file_route_pdb)
    row_list = df_2_list_of_lists(reduced_df=reduced_pdb_df)
    return row_list


def get_relevant_residues(row_list, set_sections):
    """
    Obtiene los residuos relevantes por sección.
    :param row_list:
    :param set_sections:
    :return:
    """
    residuos = []
    for i in row_list:
        if i[-1] in set_sections:
            residuos.append(i[0])
    return residuos


def get_contact_residues(cam, pdb_path):
    """
    Obtiene los contactos relevantes que se establecen de acuerdo con un mapa CAM.
    :param cam: Mapa CAM a analizar.
    :param pdb_path: Ruta del PDB correspondiente.
    :return: Diccionario con los contactos relevantes por sección de acuerdo al mapa CAM.
    """
    contacts_by_section_dict = {}
    secc = get_secciones_relevantes(cam)
    row = get_row_list_protein(pdb_path)
    for key, value in secc.items():
        contacts_by_section_dict[key] = set()
        for contact_set in value:
            values_residues = get_relevant_residues(row,
                                                    contact_set)
            contacts_by_section_dict[key].update(set(values_residues))
    return contacts_by_section_dict

def to_ranges(iterable):
    """
    Genera mediante un 'yield' los intervalos numéricos de un iterable.
    :param iterable: iterable con los elementos numéricos.
    """
    iterable = sorted(set(iterable))
    for key, group in itertools.groupby(enumerate(iterable),
                                        lambda t: t[1] - t[0]):
        group = list(group)
        yield group[0][1], group[-1][1]

def prepare_pymol_strings(dict_contacts):
    """
    Genera un diccionario {sección:intervalo_residuos}, donde el intervalo_residuos es un string de los residuos a
    seleccionar en Pymol.
    :param dict_contacts: Diccionario que almacena los residuos involucrados en contactos por sección.
    :return: Diccionario {sección:intervalo_residuos}.
    """
    dict_pymol ={}
    for key, value in dict_contacts.items():
        x_range = to_ranges(list(value))
        dict_pymol[key]=f'select resi {list(x_range)}'
    return dict_pymol
