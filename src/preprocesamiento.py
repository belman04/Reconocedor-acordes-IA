import os                         # manejo de rutas de archivos
import librosa as lib             # procesamiento de audio
import numpy as np                # manejo de matrices
import pandas as pd               # manejo de datos

def tomar_datos(ruta_datos):
    a, b = [], []  # listas para guardar MFCCs y etiquetas
    
    # Verificar que el archivo de etiquetas exista
    try:
        ruta_etiquetas = os.path.join(ruta_datos, "etiquetas.csv")
        etiquetas = pd.read_csv(ruta_etiquetas)  # carga las etiquetas 
        print("Archivo de etiquetas cargado correctamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {ruta_etiquetas}. Verifica la ruta.")
        return None, None

    # Procesar cada archivo de audio
    for index, row in etiquetas.iterrows():
        ruta_audio = os.path.join(ruta_datos, row.get("filename", ""))
        
        # Verificar si la ruta del audio es válida
        if not os.path.exists(ruta_audio):
            print(f"Advertencia: No se encontró el archivo de audio {ruta_audio}. Saltando...")
            continue

        try:
            audio, sr = lib.load(ruta_audio, sr=None)  # cargar el archivo de audio
            print(f"Procesando archivo de audio: {ruta_audio}")
            
            # Extraer los MFCCs
            mfccs = lib.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            
            # Rellenar los MFCCs para que tengan una longitud fija
            if mfccs.shape[1] < 173:  # Suponiendo que 173 es la longitud deseada
                mfccs = np.pad(mfccs, ((0, 0), (0, 173 - mfccs.shape[1])), mode='constant')
            elif mfccs.shape[1] > 173:
                mfccs = mfccs[:, :173]  # Truncar si es más largo
            
            a.append(mfccs)  # agregar MFCCs a la lista
            b.append(row["etiquetas"])  # agregar etiqueta a la lista
            
        except Exception as e:
            print(f"Error al procesar {ruta_audio}: {e}")

    # Verificar si se procesaron datos
    if not a or not b:
        print("No se procesaron datos. Verifica que los archivos de audio y etiquetas existan.")
        return None, None

    print("Procesamiento completo. Se generaron los datos de MFCCs y etiquetas.")
    return np.array(a), np.array(b)

if __name__ == "__main__":
    ruta_datos = "../datos"  # Ruta de los datos
    a, b = tomar_datos(ruta_datos)  # Cargar los datos

    # Mensaje final si se logró o no el procesamiento
    if a is not None and b is not None:
        print("Todo se procesó correctamente.")
        print(f"Total de muestras procesadas: {len(a)}")
    else:
        print("Hubo problemas durante el procesamiento.")
