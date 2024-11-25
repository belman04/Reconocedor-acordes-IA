import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import pygame  # Para reproducir archivos .wav
import os  # Para trabajar con nombres de archivos
from predicciones import tomar_modelo, predecir  # Importamos las funciones desde predicciones.py

# Inicialización de pygame para la reproducción de audio
pygame.mixer.init()

current_file = None
modelo = tomar_modelo("modelos/modelo_13.keras")  # Cargar el modelo

# Variables para controlar la reproducción y el estado
is_playing = False
current_file = None  # Para almacenar el archivo actual

# Función para cargar el archivo .wav
def cargar_archivo():
    global is_playing, is_paused, current_file

    archivo = filedialog.askopenfilename(filetypes=[("Archivos WAV", "*.wav")])
    if archivo:
        # Si ya hay un archivo reproduciéndose, lo detenemos antes de cargar uno nuevo
        if is_playing:
            pygame.mixer.music.stop()

        current_file = archivo
        pygame.mixer.music.load(archivo)
        pygame.mixer.music.play()

        # Mostrar el nombre del archivo en la interfaz
        nombre_archivo = os.path.basename(archivo)  # Obtener solo el nombre del archivo
        nombre_archivo_label.config(text=f"Audio cargado: {nombre_archivo}")

        # Iniciar la actualización del tiempo
        actualizar_tiempo()
        acorde = predecir(archivo, modelo)  # Llamar a la función de detección con el archivo cargado
        mostrar_acorde(acorde)

# Función para mostrar acorde y su imagen
def mostrar_acorde(acorde):
    acorde_label.config(text=f"{acorde}")
    imagen_path = f"img/{acorde}.png"
    
    if os.path.exists(imagen_path):
        img = Image.open(imagen_path)
        img = img.resize((80, 100))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
    else:
        messagebox.showerror("Error", f"No se encontró la imagen del acorde: {acorde}")

# Función para actualizar el tiempo del audio
def actualizar_tiempo():
    if pygame.mixer.music.get_busy():
        tiempo_actual = pygame.mixer.music.get_pos() / 1000
        tiempo_total = pygame.mixer.Sound(current_file).get_length()
        tiempo_actual_str = format_tiempo(tiempo_actual)
        tiempo_total_str = format_tiempo(tiempo_total)
        tiempo_label.config(text=f"{tiempo_actual_str}/{tiempo_total_str}")
        root.after(100, actualizar_tiempo)

# Función para formatear el tiempo en minutos:segundos
def format_tiempo(tiempo):
    minutos = int(tiempo // 60)
    segundos = int(tiempo % 60)
    return f"{minutos:02}:{segundos:02}"

# Configuración de la ventana
root = tk.Tk()
root.title("IACordes")
root.geometry("280x350")

top_frame = tk.Frame(root, bg="black", width=400, height=200)
top_frame.pack_propagate(False)
top_frame.pack(fill=tk.X)

acorde_label = tk.Label(top_frame, text="", font=("Segoe UI", 20), fg="white", bg="black")
acorde_label.pack(pady=10)

img_label = tk.Label(top_frame)
img_label.pack(pady=10)

bottom_frame = tk.Frame(root, height=400)
bottom_frame.pack(fill=tk.X, pady=10)

cargar_button = tk.Button(bottom_frame, text="Selecciona un acorde", font=("Segoe UI", 10), command=cargar_archivo)
cargar_button.pack(pady=10)

nombre_archivo_label = tk.Label(bottom_frame, text="", font=("Segoe UI", 10), fg="black")
nombre_archivo_label.pack(pady=10)

tiempo_label = tk.Label(bottom_frame, text="00:00/00:00", font=("Segoe UI", 10), fg="black")
tiempo_label.pack(pady=10)

root.mainloop()