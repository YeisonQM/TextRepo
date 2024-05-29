# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:21:50 2024

@author: User
"""

import cv2
import pytesseract
import numpy as np
import pandas as pd
import re
from PIL import Image
# import pixellib
# from pixellib.tune_bg import alter_bg
# from tensorflow.keras.layers import BatchNormalization

## C:\Users\User\Downloads\Fa\Recibos\1.jpeg"

# Cargar la imagen 
image_in = cv2.imread('recibo.jpg')
image_in = Image.fromarray(image_in)
image = image_in.crop((30,200, image_in.size[0],450))
image = np.array(image)
cv2.imshow('Corte',image)
cv2.waitKey(0)

#Convertir imagen a escala de grises
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grises',gray)
# cv2.waitKey(0)

# Apply CLAHE to improve contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
improved_image = clahe.apply(gray)

#Aplicar umbral para convertir a imagen binaria mejorada en contraste
threshold_img = cv2.threshold(improved_image, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) [1]
# cv2.imshow('Threshold',threshold_img)
# cv2.waitKey(0)


# limpieza y recorte de bordes
# Obtener las dimensiones de la imagen
height, width = threshold_img.shape

# Definir el borde superior e inferior
top_border = 10
bottom_border = height - 10

# Definir el borde izquierdo y derecho
left_border = 10
right_border = width - 10

# Recortar la imagen
cropped_image = threshold_img[top_border:bottom_border, left_border:right_border]

#Mostrar la imagen recortada
cv2.imshow('Recorte imagen',cropped_image)
cv2.waitKey(0)

# Cambiar tamaño de la imagen
scale_percent = 150  # percent of original size
width = int(cropped_image.shape[1] * scale_percent / 100)
height = int(cropped_image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_image = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_AREA)

# Mostrar la imagen
# cv2.imshow('Recambio de tamaño imagen',resized_image)
# cv2.waitKey(0)

# Find the contours in the image
contours, _ = cv2.findContours(resized_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Get the bounding rectangle of the largest contour
rect = cv2.boundingRect(largest_contour)

# Get the coordinates of the bounding rectangle
x, y, w, h = rect

# Crop the image to the bounding rectangle
cropped_image = cropped_image[y:y + h, x:x + w]

# Apply perspective transformation to the cropped image
pts1 = np.float32([
    [0, 0],
    [w, 0],
    [0, h],
    [w, h]
])

pts2 = np.float32([
    [0, 0],
    [w, 0],
    [0, h],
    [w, h]
])

matrix = cv2.getPerspectiveTransform(pts1, pts2)
result = cv2.warpPerspective(resized_image, matrix, (w, h))

# Mostrar la imagen
cv2.imshow('Fijar distorsion imagen',result)
cv2.waitKey(0)

# Cambiar el directorio donde se encuentra el tesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


#Pasar la imagen a traves de pytesseract
text = pytesseract.image_to_string(result,lang='spa')
# print('Texto: ',text)

# with open('texto.txt', 'w') as file:
#   file.write(str(text))

 # Remove non-alphanumeric characters
normalized_text = re.sub(r'[^a-zA-Z0-9 ]', '', text)

# Convert all characters to lowercase
normalized_text = normalized_text.lower()

# Remove extra whitespace
normalized_text = ' '.join(normalized_text.split())

# print(normalized_text)
  
  # Extraer datos relevantes utilizando expresiones regulares
datos_relevantes = {}
  
     
     # Find the line that contains "Recibo de caja"
recibo_caja = re.search(r' caja.*', normalized_text, flags=re.IGNORECASE)

# Extract the receipt number from the line
if recibo_caja:
    recibo_numero = re.search(r'\d+', recibo_caja.group(0)).group(0)
    print("Recibo No: ", recibo_numero)

# Busque la línea que contiene "Comprobante de pago"
comprobante_pago = re.search(r' de pago.*', normalized_text, flags=re.IGNORECASE)
# Extraiga el número del comprobante de la línea.
if comprobante_pago:
    comprobante_numero = re.search(r'\d+', comprobante_pago.group(0)).group(0)
    print("Comprobante No: ", comprobante_numero)
    
    
# Busque la línea que contiene "valor de pago"
valor_pago = re.search(r'valor .*', normalized_text, flags=re.IGNORECASE)
#  # Extraiga el valor del comprobante de la línea.
if valor_pago:
      comprobante_valor = re.search(r'\d+', valor_pago.group(0)).group(0)
      print("Valor $: ", comprobante_valor)
      
      
# Busque la línea que contiene "Concepto"
concepto = re.search(r' concepto de .*', normalized_text, flags=re.IGNORECASE )
#  # Extraiga el concepto del comprobante de la línea.
if concepto:
        por_concepto = re.search(r'.*', concepto.group(0)).group(0)
        print("Por : ", por_concepto)  


# # Extraer nombre del donante
# nombre_donante = re.findall(r"(Gara Rosa Buitiayo Cc) (\d+)", text)
# if nombre_donante:
#     datos_relevantes["nombre_donante"] = nombre_donante[0][0]
#     datos_relevantes["numero_cedula"] = nombre_donante[0][1]

# # Extraer valor de la donación
# valor_donacion = re.findall(r"(Lon\. millon\. doscientos Cincuenta\.mil Pesos) (\d+)", text)
# if valor_donacion:
#     datos_relevantes["valor_donacion_letras"] = valor_donacion[0][0]
#     datos_relevantes["valor_donacion_numeros"] = valor_donacion[0][1]

# # Extraer fecha de la donación
# fecha_donacion = re.findall(r"Fecha, (\d{2})\/(\d{2})\/(\d{4})", text)
# if fecha_donacion:
#     datos_relevantes["fecha_donacion"] = f"{fecha_donacion[0][0]}/{fecha_donacion[0][1]}/{fecha_donacion[0][2]}"

# # Extraer concepto de la donación
# concepto_donacion = re.findall(r"Por concepto de: (.*)", text)
# if concepto_donacion:
#     datos_relevantes["concepto_donacion"] = concepto_donacion[0]

# Imprimir los datos relevantes
print(datos_relevantes)

# Guardar el texto en formato txt.
      
