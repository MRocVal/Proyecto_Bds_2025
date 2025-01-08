#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:55:11 2025

@author: manuelrocamoravalenti
"""



from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Cargar las imágenes y etiquetas desde el archivo .npz
def load_data(file_path):
    data = np.load(file_path)
    images = data['images']
    labels = data['labels']
    return images, labels

# Cargar los datos
images, labels = load_data('balanced_train_data_manual_downloadable.npz')
print(f"Imágenes cargadas: {images.shape}")
print(f"Etiquetas cargadas: {labels.shape}")

# Normalizar las imágenes
#images = images / 255.0  # Normalizar los píxeles a [0, 1]

# Codificar las etiquetas
labels = np.array(labels).reshape(-1, 1)
encoder = OneHotEncoder()
labels_encoded = encoder.fit_transform(labels).toarray()



# Dividir el conjunto de datos
X_train, X_val, y_train, y_val = train_test_split(images, labels_encoded, test_size=0.3, random_state=42)

# Crear el modelo
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(124, 124, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

# Compilar el modelo
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])



# Entrenamiento con pesos de clase
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=20  # Ajusta este número según lo que desees
)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Pérdida: {loss}, Precisión: {accuracy}")

# Graficar resultados del entrenamiento
def plot_training_results(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # Gráfico de pérdida
    plt.figure(figsize=(8, 6))
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Gráfico de precisión
    plt.figure(figsize=(8, 6))
    plt.plot(accuracy, label="Training Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

plot_training_results(history)

# Predicciones en el conjunto de validación
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_val, axis=1)

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred)
print("Matriz de confusión:")
print(conf_matrix)

# Calcular sensibilidad, especificidad y precisión
tp = conf_matrix[1, 1]  # Verdaderos positivos
tn = conf_matrix[0, 0]  # Verdaderos negativos
fp = conf_matrix[0, 1]  # Falsos positivos
fn = conf_matrix[1, 0]  # Falsos negativos

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"Sensibilidad: {sensitivity:.2f}")
print(f"Especificidad: {specificity:.2f}")
print(f"Precisión: {precision:.2f}")

# Reporte de clasificación
report = classification_report(y_true, y_pred, target_names=['Benigno', 'Maligno'])
print("Reporte de clasificación:")
print(report)

# Curva ROC
fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()