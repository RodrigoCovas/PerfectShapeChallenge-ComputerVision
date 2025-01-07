import cv2
import numpy as np

def detect_bright_object(frame):
    """
    Detecta un objeto luminoso en la imagen y devuelve su posición
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Ajustar el rango para detectar brillo (alta saturación y valor)
    lower_bound = np.array([0, 0, 200], dtype=np.uint8)
    upper_bound = np.array([180, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Encontrar contornos del objeto luminoso
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:  # Filtrar pequeños ruidos
            (x, y), _ = cv2.minEnclosingCircle(largest_contour)
            return int(x), int(y)
    return None

def detectar_circulo(imagen):
    # Ocupa menos memoria que la imagen original y se mantienen mejor los FPS
    scale_factor = 0.5  
    imagen_resized = cv2.resize(imagen, None, fx=scale_factor, fy=scale_factor)

    imagen_gris = cv2.cvtColor(imagen_resized, cv2.COLOR_BGR2GRAY)

    # Ocupa menos memoria que cv2.GaussianBlur y se mantienen mejor los FPS
    imagen_gris = cv2.medianBlur(imagen_gris, 5) 

    circulos = cv2.HoughCircles(
        imagen_gris,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50, 
        param1=50,   
        param2=80,   
        minRadius=30,
        maxRadius=100  
    )

    if circulos is not None:
        return True
    return False


def calcular_angulos(vertices):
    angulos = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        p1 = vertices[i][0]
        p2 = vertices[(i + 1) % num_vertices][0]
        p3 = vertices[(i + 2) % num_vertices][0]
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)

        cos_theta = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angulo = np.degrees(np.arccos(cos_theta))

        angulos.append(angulo)

    return angulos

def aplicar_red_mask(imagen):

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Hay dos rangos de hue para el color rojo
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([190, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_segment = cv2.bitwise_and(imagen, imagen, mask=red_mask)

    return red_segment

def detectar_poligono_regular(imagen, num_lados):
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # print(f"Number of vertices in contour: {len(approx)}")

        if len(approx) == num_lados:
            area = cv2.contourArea(contour)
            if area > 100:
                angulos = calcular_angulos(approx)
                angulo_regular = 180 * (num_lados - 2) / num_lados
                if all(angulo_regular - 5 <= angle <= angulo_regular + 5 for angle in angulos):
                    return True
    return False

def detectar_octogono(imagen):
    red_segment = aplicar_red_mask(imagen)
    return detectar_poligono_regular(red_segment, 8)

def detectar_triangulo(imagen):
    return detectar_poligono_regular(imagen, 3)