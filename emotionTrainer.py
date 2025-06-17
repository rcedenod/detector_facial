import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import time

# Configuración
dataset_path = 'train'  # Ruta base del dataset
img_size = (48, 48)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise' ]

def load_data(data_path):
    images = []
    labels = []
    total_images = 0
    
    # Contar el total de imágenes primero
    for emotion in emotions:
        path = os.path.join(data_path, emotion)
        if not os.path.exists(path):
            print(f"¡ADVERTENCIA! Carpeta no encontrada: {path}")
            continue
        total_images += len(os.listdir(path))
    
    print(f"Total de imágenes a cargar: {total_images}")
    count = 0
    start_time = time.time()
    
    for idx, emotion in enumerate(emotions):
        path = os.path.join(data_path, emotion)
        if not os.path.exists(path):
            print(f"¡ERROR! Carpeta no encontrada: {path}")
            continue
            
        class_num = idx
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Error leyendo: {img_path}")
                    continue
                    
                img = cv2.resize(img, img_size)
                images.append(img)
                labels.append(class_num)
                count += 1
                
                # Mostrar progreso cada 100 imágenes
                if count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"Cargadas: {count}/{total_images} [Tiempo: {elapsed:.2f}s]")
                    
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")
    
    print(f"Imágenes cargadas con éxito: {count}/{total_images}")
    return images, labels

# Cargar datos
print("Cargando imágenes...")
images, labels = load_data(dataset_path)
print(f"{len(images)} imágenes cargadas exitosamente.")

# Si no hay imágenes, terminar el programa
if len(images) == 0:
    print("ERROR: No se encontraron imágenes. Verifica la estructura del dataset.")
    exit()

# Convertir a arrays numpy
images = np.array(images) / 255.0  # Normalizar
images = np.expand_dims(images, axis=-1)  # Añadir canal (48,48,1)
labels = np.array(labels)

# Convertir etiquetas a one-hot encoding
labels_one_hot = to_categorical(labels, num_classes=len(emotions))

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    images, 
    labels_one_hot, 
    test_size=0.2, 
    random_state=42
)

print(f"Datos divididos: Entrenamiento={len(X_train)}, Prueba={len(X_test)}")

# Crear modelo CNN
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    
    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(len(emotions), activation='softmax')
])

# Compilar
model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Resumen del modelo
model.summary()

# Entrenar
print("Iniciando entrenamiento...")
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=15,  # Reducido para prueba, aumentar para mejor precisión
    validation_data=(X_test, y_test),
    verbose=1
)

# Guardar modelo
model.save('emotion_detection_model.h5')
print("Modelo guardado como 'emotion_detection_model.h5'")

# Guardar etiquetas
np.save('emotion_labels.npy', np.array(emotions))
print("Etiquetas guardadas como 'emotion_labels.npy'")

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión final en datos de prueba: {accuracy*100:.2f}%")