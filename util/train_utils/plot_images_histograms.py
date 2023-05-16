import sys
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def generate_histogram(image_paths, output_path):
    image_histograms = []

    for image_path in image_paths:
        image = Image.open(image_path)
        image = image.convert('L')  # Convertir a escala de grises si es necesario
        image_histograms.append(np.array(image).ravel())

    stacked_histogram = np.hstack(image_histograms)

    # Calcular el histograma excluyendo el valor 0
    hist, bins = np.histogram(stacked_histogram, bins=256, range=(1, 256))

    plt.bar(bins[:-1], hist, color='gray')
    plt.xlabel('Intensidad de píxel')
    plt.ylabel('Frecuencia')
    plt.title('Histograma de imágenes tipo: {}'.format(os.path.basename(output_path)))

    plt.savefig(output_path)
    plt.close()

def generate_histograms(folder_path):
    output_folder = os.path.join(folder_path, 'histograms')
    os.makedirs(output_folder, exist_ok=True)

    image_types = ['fake_A', 'fake_B', 'real_A', 'real_B', 'rec_A', 'rec_B']
    for image_type in image_types:
        image_files = [file for file in os.listdir(folder_path) if file.endswith('.png') and image_type in file]
        if len(image_files) > 0:
            print('Generando histograma para imágenes de tipo: {}'.format(image_type))
            image_paths = [os.path.join(folder_path, file) for file in image_files]
            output_path = os.path.join(output_folder, 'hist_{}.png'.format(image_type))
            generate_histogram(image_paths, output_path)
            print('---')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Por favor, especifique la ruta de la carpeta de imágenes como argumento de línea de comandos.')
        sys.exit(1)

    # Obtener la ruta de la carpeta de imágenes del primer argumento
    folder_path = sys.argv[1]

    # Verificar si la ruta de la carpeta es válida
    if not os.path.isdir(folder_path):
        print('La ruta especificada no es una carpeta válida.')
        sys.exit(1)

    # Generar los histogramas
    generate_histograms(folder_path)