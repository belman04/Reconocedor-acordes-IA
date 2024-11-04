import os                         # manejo de rutas de archivos
import librosa as lib             # procesamiento de audio
import numpy as np                # manejo de matrices
import pandas as pd               # manejo de datos
import matplotlib.pyplot as plt   # graficar datos


def tomar_datos(ruta_datos):
    a, b = [], []                                                      # listas para guardar MFCCs y etiquetas
    etiquetas = pd.read_csv(os.path.join(ruta_datos, "etiquetas.csv")) # carga las etiquetas 

    for index, row in etiquetas.iterrows():
        ruta_audio = os.path.join(ruta_datos, row["etiquetas"])        # ruta del archivo de audio
        audio, sr = lib.load(ruta_audio, sr=None)                      # carga el archivo de audio
        mfccs = lib.feature.mfcc(b=audio, sr=sr, n_mfcc=13)            # extrae los MFCCs
        
        plot_mfccs(mfccs)                                              # grafica los MFCCs

        a.append(mfccs)                                                # agrega los MFCCs a la lista a
        b.append(row["etiquetas"])                                     # agrega la etiqueta a la lista b
    return np.array(a), np.array(b)                                    # convierte las listas en arreglos de numpy


def plot_mfccs(mfccs):                                                # grafica los MFCCs
    plt.figure(figsize=(10, 4))                                       # tamaño de la figura
    plt.imshow(mfccs, aspect='auto', origin='lower')                
    plt.title('MFCCs') 
    plt.colorbar(format='%+2.0f dB')                                  # barra de color
    plt.tight_layout()                                                # ajusta el tamaño de la figura
    plt.show()                                                        # muestra la figura


if __name__ == "__main__":
    a, b = tomar_datos("../datos")                                     # carga los datos