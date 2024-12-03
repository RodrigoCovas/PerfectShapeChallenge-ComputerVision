import cv2
import numpy as np

# Cargar la imagen
ruta = "C:/Users/mario/OneDrive/Escritorio/ICAI3.1/Vision/Lab_Project/src/Flag_of_Japan.png"
imagen = cv2.imread(ruta)  # Cambia 'imagen.jpg' por la ruta de tu imagen
print(imagen)
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
    param2=30,
    minRadius=1,
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

cv2.imshow('Círculos detectados', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()