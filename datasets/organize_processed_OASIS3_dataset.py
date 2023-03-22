import os
import shutil

# Directorio raíz donde se encuentran los directorios de interés
root_dir = "/ruta/a/directorios"

# Define los directorios de destino para los conjuntos de datos A y B
dest_dir_a = "/ruta/a/dataset/A"
dest_dir_b = "/ruta/a/dataset/B"

# Recorre todos los directorios y archivos en el directorio raíz
for dir_name, subdir_list, file_list in os.walk(root_dir):
    for file_name in file_list:
        # Si el archivo es volumen.nii.gz
        if file_name == "volumen.nii.gz":
            # Separa los componentes del nombre del archivo
            file_parts = dir_name.split("_")
            session_id = file_parts[1]
            t_type = file_parts[2]

            # Define el directorio de destino para el archivo
            if t_type == "T1w":
                dest_dir = dest_dir_a
            elif t_type == "T2w":
                dest_dir = dest_dir_b

            # Crea el directorio de destino si no existe
            os.makedirs(dest_dir, exist_ok=True)
            dest_dir_session = os.path.join(dest_dir, session_id)
            os.makedirs(dest_dir_session, exist_ok=True)

            # Copia el archivo en el directorio de destino
            src_file = os.path.join(dir_name, file_name)
            dest_file = os.path.join(dest_dir_session, file_name)
            shutil.copy2(src_file, dest_file)