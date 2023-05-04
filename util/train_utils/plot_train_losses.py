import re
import matplotlib.pyplot as plt
import argparse

def plot_log_file(log_file_path):
    # Abrir el archivo con los datos
    with open(log_file_path) as f:
        lines = f.readlines()

    # Expresión regular para buscar los valores de interés
    pattern = r"epoch: (\d+),.*D_A: ([\d.]+) G_A: ([\d.]+) cycle_A: ([\d.]+) idt_A: ([\d.]+) D_B: ([\d.]+) G_B: ([\d.]+) cycle_B: ([\d.]+) idt_B: ([\d.]+)"

    # Variables para almacenar los valores
    epochs = []
    D_A = []
    G_A = []
    cycle_A = []
    idt_A = []
    D_B = []
    G_B = []
    cycle_B = []
    idt_B = []

    # Iterar sobre las líneas del archivo
    for line in lines:
        # Buscar los valores de interés en la línea actual
        match = re.search(pattern, line)
        if match:
            # Añadir los valores a las listas correspondientes
            epoch = int(match.group(1))
            epochs.append(epoch)
            D_A.append(float(match.group(2)))
            G_A.append(float(match.group(3)))
            cycle_A.append(float(match.group(4)))
            idt_A.append(float(match.group(5)))
            D_B.append(float(match.group(6)))
            G_B.append(float(match.group(7)))
            cycle_B.append(float(match.group(8)))
            idt_B.append(float(match.group(9)))

    # Crear una figura con 8 subplots
    fig, axs = plt.subplots(8, 1, figsize=(10, 20), sharex=True)

    # Añadir los datos a los subplots correspondientes
    axs[0].plot(epochs, D_A)
    axs[0].set_ylim(0, 0.6)
    axs[0].set_ylabel("D_A")
    axs[1].plot(epochs, G_A)
    axs[1].set_ylim(0, 1.5)
    axs[1].set_ylabel("G_A")
    axs[2].plot(epochs, cycle_A)
    axs[2].set_ylabel("cycle_A")
    axs[3].plot(epochs, idt_A)
    axs[3].set_ylabel("idt_A")
    axs[4].plot(epochs, D_B)
    axs[4].set_ylabel("D_B")
    axs[5].plot(epochs, G_B)
    axs[5].set_ylabel("G_B")
    axs[6].plot(epochs, cycle_B)
    axs[6].set_ylabel("cycle_B")
    axs[7].plot(epochs, idt_B)
    axs[7].set_ylabel("idt_B")

    # Añadir un título general y un eje x compartido
    fig.suptitle("Valores por época", fontsize=16)
    plt.xlabel("Epoca")

    # Mostrar el gráfico
    plt.savefig('train_losses.png')

if __name__ == '__main__':
    # Crear un parser para recibir la ruta del archivo de log como argumento de la línea de comandos
    parser = argparse.ArgumentParser(description='Plotea los valores de un archivo de log')
    parser.add_argument('--log_file_path', required=True, type=str, help='Ruta del archivo de log')

    # Parsear los argumentos de la línea de comandos
    args = parser.parse_args()

    # Llamar a la función plot_log_file con la ruta del archivo de log
    plot_log_file(args.log_file_path)