import numpy as np
import gradCAM_functions as gcf

gpcr_keys_0 = {'TM1':{range(15,36):'1',
                    range(35,46):'2',
                    range(45,56):'3',
                    range(55,76):'4'},
             'TM2':{range(25,46):'1',
                    range(45,56):'2',
                    range(55,66):'3',
                    range(65,86):'4'},
             'TM3':{range(5,26):'1',
                    range(25,36):'2',
                    range(35,46):'3',
                    range(45,66):'4'},
             'TM4':{range(25,46):'1',
                    range(45,56):'2',
                    range(55,76):'3'},
             'TM5':{range(15,36):'1',
                    range(35,46):'2',
                    range(45,50):'3',
                    50:'4',
                    51:'5',
                    52:'6',
                    53:'7',
                    54:'8',
                    55:'9',
                    56:'10',
                    57:'11',
                    58:'12',
                    59:'13',
                    range(60,66):'14',
                    range(65,76):'15',
                    range(75,96):'16'},
             'TM6':{range(5,26):'1',
                    range(25,36):'2', # Aquí van las nuevas secciones
                    36:'3',
                    37:'4',
                    38:'5',
                    39:'6',
                    40:'7',
                    41:'8',
                    42:'9',
                    43:'10',
                    44:'11',
                    45:'12',
                    46:'13',
                    47:'14',
                    range(48,56):'15',
                    range(55,76):'16'},
             'TM7':{range(15,36):'1',
                    range(35,39):'2',
                    39: '3',
                    40: '4',
                    41: '5',
                    42: '6',
                    43: '7',
                    44: '8',
                    range(45,56):'9',
                    range(55,76):'10'},
             'H8':{range(35,56):'1',
                    range(55,76):'2'}}

coords_matrix_ballesteros = {'N-term':0, 'TM11':1, 'TM12':2, 'TM13':3, 'TM14':4, 'ICL1':5,
                             'TM21':6, 'TM22':7, 'TM23':8, 'TM24':9, 'ECL1':10, 'TM31':11,
                             'TM32':12, 'TM33':13, 'TM34':14, 'ICL2':15, 'TM41':16, 'TM42':17,
                             'TM43':18, 'ECL2':19, 'TM51':20, 'TM52':21, 'TM53':22, 'TM54':23,
                             'TM55':24, 'TM56':25, 'TM57':26, 'TM58':27, 'TM59':28, 'TM510':29,
                             'TM511':30, 'TM512':31, 'TM513':32, 'TM514':33, 'TM515':34, 'TM516':35,
                             'ECL3':36, 'TM61':37, 'TM62':38, 'TM63':39, 'TM64':40, 'TM65':41,
                             'TM66':42, 'TM67':43, 'TM68':44, 'TM69':45, 'TM610':46, 'TM611':47,
                             'TM612':48, 'TM613':49, 'TM614':50, 'TM615':51, 'TM616':52, 'ICL3':53,
                             'TM71':54, 'TM72':55, 'TM73':56, 'TM74':57, 'TM75':58, 'TM76':59,
                              'TM77':60, 'TM78':61,'TM79':62,  'TM710':63, 'H81':64, 'H82':65, 'C-term':66}

Antagonista_Hauser = ['3x40.5x50', '3x41.4x49', '6x47.7x40', '3x43.6x44',
                       '2x50.7x49', '1x49.7x50', '1x50.7x46', '1x50.2x47',
                       '1x53.7x53', '2x43.7x53', '2x42.3x46', '5x54.6x41',
                       '2x39.3x50', '3x49.3x50', '3x50.6x34', '7x53.8x50',
                       '8x49.12x50', '2x40.2x37', '2x37.12x51', '7x54.8x51']

Agonista_Hauser = ['3x50.5x58', '3x46.7x53', '5x58.6x40', '5x58.6x37',
                    '7x54.8x50', '7x52.7x53', '5x51.6x44', '3x40.6x48']

segmentos_especiales = ['8', '12']

def get_coords_index(tm, ballest):
    """
    Genera una codificación numerica de los residuos en hélices transmembranales, a partir de la numeración
    de ballesteros y con 'gpcr_keys_0'
    :param tm: Hélice transmembranal.
    :param ballest: Número de Ballesteros.
    :return: La codificación numerica de 'gpcr_keys_0'.
    """
    dict_segmento = gpcr_keys_0[tm]
    for key, value in dict_segmento.items():
        try:
            if int(ballest) in key:
                coord_index = tm+value
        except:
            if int(ballest) == key:
                coord_index = tm+value
    return coord_index

def selector_coord_indx(list_index):
    """
    Genera la codificación numérica para las matrices, de todos los residuos presentes en el GPCR.
    :param list_index: Lista de residuos con la numeración de Ballesteros.
    :return: Codificación numérica de los residuos usando 'coords_matrix_ballesteros'.
    """
    if list_index[0] not in segmentos_especiales:
        tm = 'TM'+list_index[0]
        ballest = list_index[1]
        coord_index = get_coords_index(tm, ballest)
    elif list_index[0] == '8':
        tm = 'H8'
        ballest = list_index[1]
        coord_index = get_coords_index(tm, ballest)
    elif list_index[0] == '12':
        coord_index = 'ICL1'
    return coord_index

def get_index(list_Hausser, index_list):
    """
    A partir de la numeración de Ballesteros obtiene los indices de la hélice y residuo involucrados en
    un contacto de acuerdo a Hauser.
    :param list_Hausser: Lista con los dos residuos que establecen contacto.
    :param index_list: Número de residuo del que queremos obtener su indice.
    :return: Indice codificado mediante 'coords_matrix_ballesteros'.
    """
    a = list_Hausser[index_list].split('x')
    coord_index_a = selector_coord_indx(a)
    index_a = coords_matrix_ballesteros[coord_index_a]
    return index_a

def get_dict_contacts_Hauser(list_contacts_Hausser):
    """
    Genera un diccionario {(res1, res2):#contactos}, que almacena la cantidad de contactos asociados a pares
    de residuos.
    :param list_contacts_Hausser: Lista que contiene las secciones que establecen contactos de acuerdo a Hauser.
    :return: El diccionario {(res1, res2):#contactos}.
    """
    dict_contactos_Hausser = {}
    for contact in list_contacts_Hausser:
        Hausser_contacts = contact.split('.')
        indx_a = get_index(Hausser_contacts, 0)
        indx_b = get_index(Hausser_contacts, 1)
        if (indx_a, indx_b) not in dict_contactos_Hausser:
            dict_contactos_Hausser[(indx_a, indx_b)] = 1
        else:
            dict_contactos_Hausser[(indx_a, indx_b)] += 1
    return dict_contactos_Hausser

def hauser_matrix(dict_contacts_hausser, array_dim):
    """
    Convierte el diccionario de contactos {(res1, res2):#contactos} en una matriz.
    :param dict_contacts_hauser: Diccionario {(res1, res2):#contactos}.
    :param array_dim: Dimensión de la matriz a generar.
    :return: Matriz con los contactos de Hauser codificados.
    """
    new_matrix = np.zeros(array_dim)
    #new_matrix[:] = np.nan
    for cordinates, value in dict_contacts_hausser.items():
        new_matrix[cordinates[0]][cordinates[1]] = value
        new_matrix[cordinates[1]][cordinates[0]] = value
    return new_matrix

def blend_GradCAM_Hauser(shape, hauser, CAM, ruta_guardado, nombre, alpha = 1):
    """
    Combina los mapas CAM por clase con los contactos de Hauser.
    :param shape: Tamaño de la matriz a generar, que corresponde con las secciones consideradas.
    :param hauser: Matriz con los contactos Hauser por clase.
    :param CAM: Matriz del mapa CAM para la clase.
    :param alpha: Factor de escala para la intensidad de los colores en la matriz generada.
    :return: Matriz Hauser-CAM para la clase.
    """
    white_mask = np.ones((shape, shape))
    final_blend_matrix = np.zeros((shape, shape, 3))
    final_blend_matrix[:,:,1] = white_mask-(alpha)*hauser
    final_blend_matrix[:,:,0] = white_mask-(alpha)*CAM
    final_blend_matrix[:,:,2] = white_mask
    gcf.save_data(data=final_blend_matrix, save_route=ruta_guardado, name=nombre)
    return final_blend_matrix
