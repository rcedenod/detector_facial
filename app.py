import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, ImageTk
import os

class EmotionDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección de Emociones Faciales")
        self.root.geometry("800x600")
        
        # Cargar modelo y configuración
        self.model = load_model('emotion_detection_model.h5')
        self.emotion_labels = np.load('emotion_labels.npy', allow_pickle=True)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Traducciones al español
        self.translations = {
            'angry': 'Enojo',
            'disgust': 'Disgusto',
            'fear': 'Miedo',
            'happy': 'Felicidad',
            'sad': 'Tristeza',
            'surprise': 'Sorpresa',
            'neutral': 'Neutral'
        }
        
        # Variables
        self.cap = None
        self.is_camera_active = False
        self.current_image = None
        
        # Crear interfaz
        self.create_widgets()
    
    def create_widgets(self):
        # Panel de control
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=10)
        
        self.btn_load = ttk.Button(control_frame, text="Cargar Imagen", command=self.load_image)
        self.btn_load.grid(row=0, column=0, padx=5)
        
        self.btn_camera = ttk.Button(control_frame, text="Activar Cámara", command=self.toggle_camera)
        self.btn_camera.grid(row=0, column=1, padx=5)
        
        self.btn_capture = ttk.Button(control_frame, text="Capturar Imagen", command=self.capture_image, state=tk.DISABLED)
        self.btn_capture.grid(row=0, column=2, padx=5)
        
        # Área de imagen
        self.image_label = ttk.Label(self.root)
        self.image_label.pack(pady=10, expand=True)
        
        # Resultado
        self.result_frame = ttk.Frame(self.root)
        self.result_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(self.result_frame, text="Emoción detectada:", font=("Arial", 12)).pack(side=tk.LEFT)
        self.result_var = tk.StringVar(value="Ninguna")
        self.result_display = ttk.Label(self.result_frame, textvariable=self.result_var, font=("Arial", 12, "bold"))
        self.result_display.pack(side=tk.LEFT, padx=10)
        
        # Barra de precisión
        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(self.progress_frame, text="Precisión:").pack(side=tk.LEFT)
        self.progress_var = tk.StringVar(value="0%")
        self.progress_label = ttk.Label(self.progress_frame, textvariable=self.progress_var)
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(pady=5)
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.process_image(file_path)
    
    def toggle_camera(self):
        if self.is_camera_active:
            self.stop_camera()
            self.btn_camera.config(text="Activar Cámara")
            self.btn_capture.config(state=tk.DISABLED)
        else:
            self.start_camera()
            self.btn_camera.config(text="Desactivar Cámara")
            self.btn_capture.config(state=tk.NORMAL)
    
    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.result_var.set("Error: Cámara no disponible")
            return
        
        self.is_camera_active = True
        self.show_camera_feed()
    
    def stop_camera(self):
        self.is_camera_active = False
        if self.cap:
            self.cap.release()
        self.image_label.config(image='')
    
    def show_camera_feed(self):
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame.copy()
                # Mostrar feed sin procesamiento para mejor rendimiento
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
            self.root.after(10, self.show_camera_feed)
    
    def capture_image(self):
        if self.current_image is not None:
            self.process_captured_image(self.current_image)
    
    def process_captured_image(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.process_image_data(img)
    
    def process_image(self, file_path):
        img = cv2.imread(file_path)
        if img is None:
            self.result_var.set("Error: Imagen no válida")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.process_image_data(img)
    
    def process_image_data(self, img):
        display_img = img.copy()
        # Convertir a escala de grises para detección
        gray = cv2.cvtColor(display_img, cv2.COLOR_RGB2GRAY)
        
        # Parámetros optimizados para imágenes pequeñas
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # Usar imagen completa si no se detectan rostros en imágenes pequeñas
        if len(faces) == 0:
            h, w = gray.shape
            if h <= 100 and w <= 100:
                faces = [(0, 0, w, h)]
            else:
                self.result_var.set("No se detectaron rostros")
                self.progress_var.set("0%")
                self.progress['value'] = 0
                # Mostrar imagen sin anotaciones
                display_img = Image.fromarray(display_img)
                display_img = display_img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=display_img)
                self.image_label.imgtk = imgtk
                self.image_label.configure(image=imgtk)
                return
        
        # Procesar cada rostro detectado
        for (x, y, w, h) in faces:
            # Recortar y preprocesar el rostro (SOLUCIÓN CLAVE: usar escala de grises)
            face_roi = gray[y:y+h, x:x+w]  # Usar la imagen en escala de grises
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Preparar imagen para el modelo (SOLUCIÓN CLAVE)
            face_roi = face_roi.astype("float") / 255.0
            face_roi = np.expand_dims(face_roi, axis=-1)  # Añadir dimensión del canal (48,48,1)
            face_roi = np.expand_dims(face_roi, axis=0)   # Añadir dimensión del batch (1,48,48,1)
            
            # Realizar predicción
            predictions = self.model.predict(face_roi)[0]
            emotion_idx = np.argmax(predictions)
            emotion_key = self.emotion_labels[emotion_idx]
            emotion = self.translations.get(emotion_key, emotion_key)
            
            # Dibujar resultados en la imagen
            cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            emotion_text = f"{emotion}"
            cv2.putText(display_img, emotion_text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Actualizar UI
            self.result_var.set(emotion)
        
        # Mostrar imagen con resultados
        display_img = Image.fromarray(display_img)
        display_img = display_img.resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=display_img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectionApp(root)
    root.mainloop()