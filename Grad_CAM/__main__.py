import gradCAM_functions as gcf
import tensorflow as tf
import Hauser_matrix_functions as hmf
import Pymol_entries_functions as pef

path_0 = '.../' #Path where files are stored

ruta_base = path_0 + '.../' #Set this path

ruta_guardado_agonist = ruta_base + 'Grad_CAM_resultados/Agonist/'
ruta_guardado_antagonist = ruta_base + 'Grad_CAM_resultados/Antagonist/'

ruta_guardado_matrices_agonist = ruta_base + 'Grad_CAM_matrix/test/Agonist/'
ruta_guardado_matrices_antagonist = ruta_base + 'Grad_CAM_matrix/test/Antagonist/'

ruta_guardado_resultados = ruta_base + 'Resultados_analisis_CAM'

file_route_agonist = path_0 + '.../PDBs/test/Agonist/6FUF.pdb' #Set this path
file_route_antagonist = path_0 + '.../PDBs/test/Antagonist/5VRA.pdb' #Set this path

ruta_test_RRCS = ruta_base + 'RRCS_matrix/test/'
x_test_RRCS, y_test_RRCS, pdb_names = gcf.prepare_dataset(ruta_test_RRCS)

last_conv_layer_RRCS = 'max_pooling2d_5'
last_layer_RRCS='dense_16'

shape = 67

full_model_00 = tf.keras.models.load_model(ruta_base + 'saved_models/red_V01_loss_0.194_acc_0.94.tf')

for i, j in enumerate(pdb_names):
    if y_test_RRCS[i] == 1:
        save_route_matrix = ruta_guardado_matrices_antagonist
        save_route_plt = ruta_guardado_antagonist
    elif y_test_RRCS[i] == 0:
        save_route_matrix = ruta_guardado_matrices_agonist
        save_route_plt = ruta_guardado_agonist
    gcf.get_gradCAM_map(x=x_test_RRCS[i],
                        y=y_test_RRCS[i],
                        file_name=j,
                        modelo=full_model_00,
                        shape=shape,
                        last_conv_layer=last_conv_layer_RRCS,
                        last_layer_model=last_layer_RRCS,
                        ruta_guardado_matrices=save_route_matrix,
                        ruta_guardado_plt=save_route_plt)

agonist_map = gcf.blend_maps(ruta_guardado_matrices_agonist,
                             ruta_guardado_resultados, nombre='agonista.pkl')
antagonist_map = gcf.blend_maps(ruta_guardado_matrices_antagonist,
                             ruta_guardado_resultados, nombre='antagonista.pkl')

dict_contactos_Hauser_antagonista = hmf.get_dict_contacts_Hauser(hmf.Antagonista_Hauser)
dict_contactos_Hauser_agonista = hmf.get_dict_contacts_Hauser(hmf.Agonista_Hauser)

hauser_agonista = hmf.hauser_matrix(dict_contactos_Hauser_agonista, array_dim = (shape, shape))
hauser_antagonista = hmf.hauser_matrix(dict_contactos_Hauser_antagonista, array_dim = (shape, shape))

Agonist_blend_map = hmf.blend_GradCAM_Hauser(shape=67,
                                        hauser=hauser_agonista,
                                        CAM=agonist_map,
                                        ruta_guardado=ruta_guardado_resultados,
                                        nombre = 'agonista_hauser_cam.pkl')

Antagonist_blend_map = hmf.blend_GradCAM_Hauser(shape=67,
                                        hauser=hauser_antagonista,
                                        CAM=antagonist_map,
                                        ruta_guardado=ruta_guardado_resultados,
                                        nombre = 'antagonista_hauser_cam.pkl')

antagonist_contacts_dict = pef.get_contact_residues(antagonist_map, file_route_antagonist)
agonist_contacts_dict = pef.get_contact_residues(agonist_map, file_route_agonist)

ago_pym_strings = pef.prepare_pymol_strings(agonist_contacts_dict)
ant_pym_strings = pef.prepare_pymol_strings(antagonist_contacts_dict)
secc_ant = pef.get_secciones_relevantes(antagonist_map)
secc_ago = pef.get_secciones_relevantes(agonist_map)

print(ago_pym_strings, ant_pym_strings, secc_ant, secc_ago)
