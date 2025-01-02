import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import librosa
import numpy as np
import joblib
import subprocess
from PIL import Image, ImageTk

import sys
import os

def resource_path(relative_path):
    """Obtiene la ruta absoluta del archivo, compatible con PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        # Cuando se ejecuta desde un ejecutable generado por PyInstaller
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        # Durante el desarrollo o ejecución normal
        return os.path.join(os.path.abspath("."), relative_path)


class AudioClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Voces Artificales - Riego")
        self.root.geometry("750x562")
        self.root.resizable(False, False)
        
        # Configurar el ícono de la ventana
        self.icon_image = Image.open(resource_path('images/logo-v3.png'))
        self.icon_photo = ImageTk.PhotoImage(self.icon_image)
        self.root.iconphoto(False, self.icon_photo)

        # Configurar el estilo
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Personalizar estilos
        self.style.configure('TButton', font=('Helvetica', 12))
        self.style.configure('TLabel', font=('Helvetica', 12))

        # Cargar el modelo de detección de voces
        self.clf_svm_mfcc = joblib.load(resource_path('models/svm_mfcc_model.pkl'))

        # Crear los widgets
        self.create_widgets()

    def create_widgets(self):
        # Cargar imagen de fondo
        self.bg_image = Image.open(resource_path('images/background.png'))
        self.bg_image = self.bg_image.resize((750, 562), Image.Resampling.LANCZOS)
        self.bg_photo = ImageTk.PhotoImage(self.bg_image)
        
        # Crear una etiqueta para el fondo y colocarla
        self.bg_label = tk.Label(self.root, image=self.bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        # Marco principal (usando `place` para colocarlo sobre el fondo)
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Centrar el marco en la ventana
        
        # Cargo el logo
        self.logo_image = Image.open(resource_path('images/logo-v3.png'))
        self.logo_image = self.logo_image.resize((250, 250), Image.Resampling.LANCZOS)
        self.logo_photo = ImageTk.PhotoImage(self.logo_image)
        logo_label = ttk.Label(main_frame, image=self.logo_photo)
        logo_label.pack(pady=10)

        # Etiqueta de título
        # title_label = ttk.Label(main_frame, text="Detector de Voces Artificiales", font=("Helvetica", 18, "bold"))
        # title_label.pack(pady=10)

        # Botón para seleccionar archivo
        select_button = ttk.Button(main_frame, text="Seleccionar archivo de audio (.wav, .mp3)", command=self.select_file)
        select_button.pack(pady=20)

        # Etiqueta de subtítulo
        subtitle_label = ttk.Label(main_frame, text="Solo funciona con voces en inglés", font=("Helvetica", 10, "italic"))
        subtitle_label.pack(pady=10)

        # Etiqueta para mostrar el resultado
        self.result_label = ttk.Label(main_frame, text="", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        # Barra de progreso (más estrecha)
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='indeterminate', length=200)
        self.progress.pack(pady=10, fill=tk.X)

    def select_file(self):
        file_path = filedialog.askopenfilename(title="Seleccionar archivo de audio",
                                               filetypes=(("Archivos de audio", "*.wav;*.mp3;*.flac;*.ogg"), ("Todos los archivos", "*.*")))
        if file_path:
            # Mostrar barra de progreso
            self.progress.start()
            self.root.update_idletasks()
            
            # Convertir el archivo a WAV si es necesario
            wav_file_path = self.convert_to_wav(file_path)
            if wav_file_path is None:
                messagebox.showerror("Error", "No se pudo convertir el archivo a WAV.")
                self.progress.stop()
                return
            
            # Realizar la predicción
            result = self.predict_audio(wav_file_path)
            
            # Mostrar el resultado en la etiqueta
            self.result_label.config(text=f"{result[0]}\n\nProbabilidad de ser Real: {result[1]:.2%}\nProbabilidad de ser Falso: {result[2]:.2%}")
            
            # Detener la barra de progreso
            self.progress.stop()
            
            # Eliminar el archivo WAV convertido si se creó uno nuevo
            if wav_file_path != file_path:
                os.remove(wav_file_path)
    
    def predict_audio(self, file_path):
        # Preprocesar el archivo
        audio_new = self.extract_mfcc(file_path)
        if audio_new is None:
            messagebox.showerror("Error", "No se pudo procesar el archivo de audio.")
            return ["Error al procesar el audio", 0, 0]
        
        # Realizar la predicción
        probabilidades = self.clf_svm_mfcc.predict_proba([audio_new])
        
        # Obtener las probabilidades
        prob_real = probabilidades[0][0]
        prob_fake = probabilidades[0][1]
        
        if prob_real > prob_fake:
            return ["El audio es Real", prob_real, prob_fake]
        else:
            return ["El audio es Falso", prob_real, prob_fake]
    
    def extract_mfcc(self, file_path, n_mfcc=13):
        try:
            y, sr = librosa.load(file_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            return mfcc_mean
        except Exception as e:
            print(f"Error al procesar {file_path}: {e}")
            return None
    
    def convert_to_wav(self, file_path):
        # Lista de extensiones de archivo aceptadas
        accepted_extensions = ['.mp3', '.wav', '.flac', '.ogg']

        # Verificar la extensión del archivo
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in accepted_extensions:
            messagebox.showerror("Error", f"Formato no soportado: {file_extension}. Solo se aceptan {', '.join(accepted_extensions)}.")
            return None

        # Verificar si el archivo ya es WAV
        if file_extension == '.wav':
            return file_path
        else:
            # Crear una ruta para el archivo WAV convertido
            wav_file_path = os.path.splitext(file_path)[0] + '_converted.wav'
            
            # Comando para convertir el archivo a WAV usando ffmpeg
            command = ['ffmpeg', '-y', '-i', file_path, wav_file_path]
            try:
                subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                return wav_file_path
            except subprocess.CalledProcessError as e:
                print(f"Error al convertir el archivo: {e}")
                messagebox.showerror("Error", "No se pudo convertir el archivo a formato WAV.")
                return None

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioClassifierApp(root)
    root.mainloop()