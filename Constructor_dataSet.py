#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 16:54:17 2025

@author: manuelrocamoravalenti
"""

import xml.etree.ElementTree as ET
import os


import os

# Definir el directorio raíz donde se encuentran los archivos .xml
root_directory = "Mani"

# Obtener todas las rutas de los archivos .xml
file_paths = []
for dirpath, _, filenames in os.walk(root_directory):
    for filename in filenames:
        if filename.endswith(".xml"):
            file_paths.append(os.path.join(dirpath, filename))

# Verificar las rutas obtenidas
print(file_paths)

import xml.etree.ElementTree as ET
import os

import xml.etree.ElementTree as ET
import os

def extract_nodule_info_with_malignancy(file_path):
    """
    Analiza un archivo XML y extrae la información de los nódulos.
    Asigna la etiqueta de malignidad: 1 si se encuentra la etiqueta <malignancy>, 0 si no se encuentra.
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        namespace = {'nih': 'http://www.nih.gov'}  # Espacio de nombres

        nodules_info = []
        for session in root.findall("nih:readingSession", namespace):
            for nodule in session.findall("nih:unblindedReadNodule", namespace):
                nodule_id = nodule.findtext("nih:noduleID", default="Unknown", namespaces=namespace)
                characteristics = nodule.find("nih:characteristics", namespace)
                
                # Asignar malignidad
                malignancy = 0  # Por defecto es benigno
                if characteristics is not None:
                    # Buscar la etiqueta <malignancy>
                    malignancy_tag = characteristics.find("nih:malignancy", namespace)
                    if malignancy_tag is not None:
                        malignancy = 1# Si se encuentra, asignamos 1 (maligno)
                
                # Extraer coordenadas (si las hay)
                rois = []
                for roi in nodule.findall("nih:roi", namespace):
                    image_sop_uid = roi.findtext("nih:imageSOP_UID", default="Unknown", namespaces=namespace)
                    z_position = roi.findtext("nih:imageZposition", default="Unknown", namespaces=namespace)
                    coordinates = [
                        (edge.findtext("nih:xCoord", namespaces=namespace), edge.findtext("nih:yCoord", namespaces=namespace))
                        for edge in roi.findall("nih:edgeMap", namespace)
                    ]
                    rois.append({
                        "image_sop_uid": image_sop_uid,
                        "z_position": z_position,
                        "coordinates": coordinates
                    })
                
                nodules_info.append({
                    "nodule_id": nodule_id,
                    "malignancy": malignancy,
                    "rois": rois
                })
        return nodules_info
    except Exception as e:
        print(f"Error al procesar el archivo {file_path}: {e}")
        return {"error": str(e)}

# Usar la función modificada para procesar los archivos XML
root_directory = "Mani"
file_paths = []  # Obtener las rutas de los archivos XML
for dirpath, _, filenames in os.walk(root_directory):
    for file in filenames:
        if file.endswith(".xml"):
            file_paths.append(os.path.join(dirpath, file))

# Procesar todos los archivos .xml
all_nodules_data = {}
for file_path in file_paths:
    all_nodules_data[file_path] = extract_nodule_info_with_malignancy(file_path)

# Mostrar un resumen de los datos extraídos
if all_nodules_data:
    for file_name, data in all_nodules_data.items():
        print(f"Archivo: {file_name}")
        print(data)
        print("=" * 50)
else:
    print("No se encontraron archivos .xml o no se pudieron procesar.")


    
    


# ====================================  
    
import os
import pydicom
import time

def find_dicom_file(root_directory, sop_uid, timeout=2):
    """
    Busca un archivo DICOM que coincida con el SOP UID especificado en las subcarpetas del directorio raíz.
    Incluye un tiempo límite para evitar bloqueos.
    """
    for dirpath, _, filenames in os.walk(root_directory):
        for file in filenames:
            if file.endswith(".dcm"):
                file_path = os.path.join(dirpath, file)
                try:
                    # Establecer un límite de tiempo
                    start_time = time.time()
                    dicom_data = pydicom.dcmread(file_path)
                    if time.time() - start_time > timeout:
                        print(f"Timeout al leer {file_path}. Saltando.")
                        continue
                    if dicom_data.SOPInstanceUID == sop_uid:
                        return file_path
                except Exception as e:
                    print(f"Error al procesar {file_path}: {e}")
                    continue
    return None

import time

def associate_dicom_files(root_directory, nodule_data):
    """
    Asocia los datos de los nódulos con los archivos DICOM correspondientes y
    muestra información del paciente actualmente en procesamiento.
    
    Parameters:
    - root_directory (str): Ruta raíz donde se encuentran los archivos DICOM.
    - nodule_data (dict): Diccionario con información de los nódulos.
    
    Returns:
    - nodule_data (dict): Diccionario actualizado con rutas DICOM asociadas.
    """
    total_files = len(nodule_data)  # Número total de pacientes/archivos
    start_time = time.time()       # Tiempo inicial para calcular duración
    
    for index, (file_name, nodules) in enumerate(nodule_data.items(), start=1):
        print(f"Procesando paciente {index}/{total_files}: {file_name}")
        
        for nodule in nodules:
            for roi in nodule["rois"]:
                sop_uid = roi["image_sop_uid"]
                dicom_path = find_dicom_file(root_directory, sop_uid)
                roi["dicom_path"] = dicom_path
        
        # Tiempo transcurrido y estimación del tiempo restante
        elapsed_time = time.time() - start_time
        estimated_time = (elapsed_time / index) * (total_files - index)
        print(f"Tiempo transcurrido: {elapsed_time:.2f}s | Tiempo estimado restante: {estimated_time:.2f}s\n")
    
    return nodule_data

# Directorio con los archivos DICOM
dicom_root_directory = "Mani"
nodule_data_with_dicom = associate_dicom_files(dicom_root_directory, all_nodules_data)

import cv2
import numpy as np

import numpy as np
import pydicom
import cv2
import os
import matplotlib.pyplot as plt

# Validar ROIs para asegurarse de que estén dentro de los límites de la imagen
def validate_roi(roi, image_shape):
    x_min, y_min, x_max, y_max = roi
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(image_shape[1], x_max)
    y_max = min(image_shape[0], y_max)
    return x_min, y_min, x_max, y_max

import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_training_data_from_dict(nodule_data, target_size=(124, 124)):
    """
    Procesa los datos de nódulos pulmonares para crear un dataset de imágenes y etiquetas.

    Parameters:
    - nodule_data (dict): Diccionario con información sobre los nódulos y sus DICOMs.
    - target_size (tuple): Dimensiones a las que redimensionar las imágenes.

    Returns:
    - images (list): Lista de imágenes procesadas.
    - labels (list): Lista de etiquetas basadas en la malignidad.
    """
    images, labels = [], []

    # Iterar sobre cada archivo en el diccionario
    for file_name, nodules in nodule_data.items():
        if nodules:  # Verificar que hay nódulos en la lista
            for nodule in nodules:
                malignancy = nodule.get("malignancy", 0)  # Obtener la malignidad del nódulo (por defecto 0)
                for roi in nodule.get("rois", []):  # Iterar sobre las ROIs
                    dicom_path = roi.get("dicom_path")
                    coordinates = roi.get("coordinates", [])
                    
                    # Depuración: Asegurarse de que tenemos los datos correctos
                    print(f"Procesando ROI: DICOM={dicom_path}, Malignancy={malignancy}")

                    # Si hay un dicom_path, se procesa (simulado aquí)
                    if dicom_path:
                        # Aquí simulamos la carga y recorte de imágenes (ya que no podemos procesar DICOM directamente ahora)
                        cropped_image = np.zeros((512, 512))  # Simulación de imagen
                        resized_image = cv2.resize(cropped_image, target_size)
                        images.append(resized_image)
                        labels.append(malignancy)
    
    # Convertimos a arrays de numpy
    return np.array(labels)






def extract_images(nodule_data, target_size=(124, 124)):
    """
    Procesa las imágenes de los nódulos pulmonares.

    Parameters:
    - nodule_data (dict): Diccionario con información sobre los nódulos y sus DICOMs.
    - target_size (tuple): Dimensiones a las que redimensionar las imágenes.

    Returns:
    - images (list): Lista de imágenes procesadas.
    """
    images = []
    for file_name, nodules in nodule_data.items():
        if nodules:
            for nodule in nodules:
                for roi in nodule.get("rois", []):
                    dicom_path = roi.get("dicom_path")
                    coordinates = roi.get("coordinates", [])
                    
                    if dicom_path:
                        cropped_image = crop_dicom_image(dicom_path, coordinates)
                        if cropped_image is not None:
                            # Normalizar y redimensionar la imagen
                            normalized_image = cropped_image / np.max(cropped_image) if np.max(cropped_image) > 0 else cropped_image
                            resized_image = cv2.resize(normalized_image, target_size)
                            images.append(resized_image)
    return np.array(images)


def crop_dicom_image(dicom_path, coordinates):
    """
    Recorta una imagen DICOM utilizando las coordenadas proporcionadas.

    Parameters:
    - dicom_path (str): Ruta al archivo DICOM.
    - coordinates (list): Lista de coordenadas [(x1, y1), (x2, y2), ...].

    Returns:
    - cropped_image (np.array): Imagen recortada o None si no es válida.
    """
    import pydicom

    try:
        dicom_data = pydicom.dcmread(dicom_path)
        image_array = dicom_data.pixel_array

        # Validar y recortar la imagen
        x_coords = [int(coord[0]) for coord in coordinates]
        y_coords = [int(coord[1]) for coord in coordinates]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        if x_min >= 0 and y_min >= 0 and x_max <= image_array.shape[1] and y_max <= image_array.shape[0]:
            cropped_image = image_array[y_min:y_max, x_min:x_max]
            if cropped_image.size > 0:
                return cropped_image
    except Exception as e:
        print(f"Error al procesar {dicom_path}: {e}")

    return None


# Crear etiquetas
labels = create_training_data_from_dict(nodule_data_with_dicom)

# Crear imágenes
images = extract_images(nodule_data_with_dicom, target_size=(124, 124))

# Asegurarnos de que las etiquetas e imágenes estén sincronizadas
if len(labels) > len(images):
    labels = labels[:len(images)]
elif len(images) > len(labels):
    images = images[:len(labels)]

# Verificar resultados
print(f"Imágenes procesadas: {images.shape}")
print(f"Etiquetas procesadas: {labels.shape}")

# Visualizar distribución de etiquetas
unique, counts = np.unique(labels, return_counts=True)
print("Distribución de etiquetas:", dict(zip(unique, counts)))

# Visualizar algunas imágenes con etiquetas
for i in range(min(5, len(images))):
    plt.imshow(images[i], cmap='gray')
    plt.title(f"Etiqueta: {labels[i]}")
    plt.colorbar()
    plt.show()

# Guardar las imágenes y las etiquetas en un archivo .npz
def save_data(images, labels, file_path):
    np.savez_compressed(file_path, images=images, labels=labels)
    print(f"Datos guardados en: {file_path}")

# Ruta donde guardar los datos
save_data(images, labels, 'train_data_100P.npz')