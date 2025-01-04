import cv2
import numpy as np
import os
import glob


def load_images(filenames):
    return [cv2.imread(filename) for filename in filenames]

# Cargar la imagen
imgs_path= []
current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
folder = os.path.join(parent_directory,"data")
folder = folder.replace("\\", "/") + "/"
print(folder)
for filename in glob.glob(folder+ "*.jpg"):
    print(filename)
    imgs_path.append(filename)
imgs= load_images(imgs_path)
imagen = imgs[1]
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises

# Aplicar suavizado para reducir ruido
imagen_gris = cv2.GaussianBlur(imagen_gris, (9, 9), 2)

# Detectar círculos utilizando la transformación de Hough
circulos = cv2.HoughCircles(
    imagen_gris,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=100,
    param1=50,
    param2=100,
    minRadius=10,
    maxRadius=10000
)

# Verificar si se detectaron círculos
if circulos is not None:
    print("Se ha detectado al menos un círculo.")
else:
    print("No se detectaron círculos.")

if circulos is not None:
    circulos = np.uint16(np.around(circulos))  # Redondear y convertir a entero
    for i in circulos[0, :]:
        # Dibujar el círculo
        cv2.circle(imagen, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Dibujar el centro del círculo
        cv2.circle(imagen, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('Circulos detectados', imagen)
cv2.waitKey()
cv2.destroyAllWindows()