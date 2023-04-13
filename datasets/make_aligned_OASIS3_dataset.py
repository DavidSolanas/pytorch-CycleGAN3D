import os
import numpy as np
import shutil
from tqdm import tqdm
import nibabel as nib

# Directorio de entrada y salida
input_dir = '/Disco2021-I/david/tfm/dataset/OASIS3_processed'
output_dir = '/Disco2021-I/david/tfm/dataset/cyclegan_dataset'

# Crear carpeta de salida si no existe
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Obtener lista de subcarpetas XXX_YYY
subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

# Iterar sobre las subcarpetas y cargar los archivos T1w y T2w en las listas
for subfolder in tqdm(subfolders, desc='Procesando subcarpetas', unit='subcarpeta'):
    subfolder_path = os.path.join(input_dir, subfolder)
    t1w_path = os.path.join(subfolder_path, 'T1w', 'orig_nu_noskull.nii.gz')
    t2w_path = os.path.join(subfolder_path, 'T2w', 'orig_nu_noskull.nii.gz')

    # Cargar datos de T1w y T2w utilizando nibabel
    t1w_img = nib.load(t1w_path)
    t2w_img = nib.load(t2w_path)

    # Obtener los datos de las imágenes como arrays numpy y apilarlos en un solo array
    data = np.array((t1w_img.get_fdata(), t2w_img.get_fdata()))

    # Crear una carpeta dentro de la carpeta de salida para cada par T1w y T2w
    output_subfolder = os.path.join(output_dir, subfolder)
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Guardar los datos en un archivo numpy en la carpeta de salida correspondiente
    output_path = os.path.join(output_subfolder, 'data.npy')
    np.save(output_path, data)

print('Dataset creado exitosamente en:', output_dir)