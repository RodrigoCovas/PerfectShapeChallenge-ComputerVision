import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from itertools import combinations


def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

def ajustar_circulo(puntos):
    """
    Ajusta un círculo a los puntos dados
    """
    def residuo(params, puntos):
        cx, cy, r = params
        return [np.sqrt((x - cx)**2 + (y - cy)**2) - r for x, y in puntos]

    # Inicialización: Centro aproximado y radio promedio
    x_mean, y_mean = np.mean(puntos, axis=0)
    r_init = np.mean([np.sqrt((x - x_mean)**2 + (y - y_mean)**2) for x, y in puntos])
    params_iniciales = [x_mean, y_mean, r_init]

    # Ajustar el círculo
    resultado = least_squares(residuo, params_iniciales, args=(puntos,))
    cx, cy, r = resultado.x
    return cx, cy, r

def distancia_punto_segmento(p, a, b):
    # Convertir a numpy arrays
    p = np.array(p)
    a = np.array(a)
    b = np.array(b)
    
    # Vector del segmento AB
    ab = b - a
    # Vector del punto a P respecto a A
    ap = p - a
    # Longitud al cuadrado del segmento AB
    ab_len_sq = np.dot(ab, ab)
    
    if ab_len_sq == 0:
        # Caso especial: A y B son el mismo punto
        return np.linalg.norm(p - a)
    
    # Proyección escalar de AP sobre AB, limitado al rango [0, 1]
    t = max(0, min(1, np.dot(ap, ab) / ab_len_sq))
    # Coordenadas del punto proyectado sobre el segmento
    proyeccion = a + t * ab
    
    # Distancia del punto P al punto proyectado
    return np.linalg.norm(p - proyeccion)

def error_distancia(triangle, polygon):
    a, b, c = triangle
    error = 0
    for p in polygon:
        error += min(
            distancia_punto_segmento(p, a, b),
            distancia_punto_segmento(p, b, c),
            distancia_punto_segmento(p, c, a)
        )
    return error

def ajustar_triangulo(puntos):
    hull = ConvexHull(puntos)
    vertices = [puntos[i] for i in hull.vertices]

    best_triangle = None
    min_error = float('inf')
    for triangle in combinations(vertices, 3):
        error = error_distancia(triangle, vertices)
        if error < min_error:
            min_error = error
            best_triangle = triangle
    return np.array(best_triangle)

def generar_puntos_circulo(cx, cy, r, n_puntos):
    """
    Genera n_puntos uniformemente distribuidos en el círculo ajustado
    """
    angulos = np.linspace(0, 2 * np.pi, n_puntos, endpoint=False)
    puntos_circulo = [(cx + r * np.sin(theta), cy - r * np.cos(theta)) for theta in angulos]
    return puntos_circulo

def generar_puntos_triangulo(triangulo, n_puntos):
    """
    Genera n_puntos distribuidos uniformemente en el perímetro del triángulo.
    """
    lados = [np.linalg.norm(triangulo[i] - triangulo[(i + 1) % 3]) for i in range(3)]
    total_largo = sum(lados)
    puntos = []

    for i in range(3):
        p1, p2 = triangulo[i], triangulo[(i + 1) % 3]
        n_lado = int(n_puntos * lados[i] / total_largo)
        for t in np.linspace(0, 1, n_lado + 1, endpoint=False):
            puntos.append(p1 + t * (p2 - p1))

    #Convertir los puntos a tuplas
    puntos = [tuple(punto) for punto in puntos]
    return puntos

def calcular_semejanza_circulo(puntos_originales, puntos_circulo, radio):
    """
    Calcula la semejanza entre los puntos originales y los del círculo ajustado
    """
    #Calcular las distancias entre puntos originales y el más cercano de entre los puntos del círculo ajustado,
    #teniendo en cuenta que cada punto del círculo ajustado solo puede ser emparejado con dos puntos originales
    distancias = []
    puntos_disponibles = puntos_circulo.copy()
    puntos_unidos = {punto: [] for punto in puntos_circulo}
    for punto_orig in puntos_originales:
        distancias_punto = [euclidean_distance(punto_orig, punto_circulo) for punto_circulo in puntos_disponibles]
        distancia_minima = min(distancias_punto)
        distancias.append(distancia_minima)
        punto_cercano = puntos_disponibles[distancias_punto.index(distancia_minima)]
        puntos_unidos[punto_cercano].append(punto_orig)
        if len(puntos_unidos[punto_cercano]) == 2:
            puntos_disponibles.remove(punto_cercano)

    #Elevamos las distancias al cubo para penalizar más las distancias grandes
    distancias = [distancia**3 for distancia in distancias]
    media_distancia = np.mean(distancias)

    #Calculamos la semejanza
    semejanza = 1 / (1 + media_distancia/((radio**3)/200))

    return semejanza, puntos_unidos


def calcular_semejanza_triangulo(puntos_originales, puntos_triangulo):
    """
    Calcula la semejanza usando la media de las distancias.
    """
    distancias = []
    puntos_disponibles = puntos_triangulo.copy()
    puntos_unidos = {punto: [] for punto in puntos_triangulo}
    for punto_orig in puntos_originales:
        distancias_punto = [euclidean_distance(punto_orig, punto_triangulo) for punto_triangulo in puntos_disponibles]
        distancia_minima = min(distancias_punto)
        distancias.append(distancia_minima)
        punto_cercano = puntos_disponibles[distancias_punto.index(distancia_minima)]
        puntos_unidos[punto_cercano].append(punto_orig)
        if len(puntos_unidos[punto_cercano]) == 2:
            puntos_disponibles.remove(punto_cercano)

    #Elevamos las distancias al cuadrado
    distancias = [distancia**3 for distancia in distancias]
    media_distancia = np.mean(distancias)

    #Calculamos la semejanza
    semejanza = 1 / (1 + media_distancia/(10000))

    return semejanza, puntos_unidos

def plot_circulo(puntos_originales, puntos_circulo, puntos_unidos):
    """
    Grafica los puntos originales y los del círculo ajustado
    """
    fig, ax = plt.subplots()

    # Graficar los puntos originales
    x_coords, y_coords = zip(*puntos_originales)
    ax.scatter(x_coords, y_coords, color='blue', label='Puntos originales')

    # Graficar los puntos del círculo ajustado
    x_coords, y_coords = zip(*puntos_circulo)
    ax.scatter(x_coords, y_coords, color='red', label='Puntos modelados')
    #ax.plot(x_coords + x_coords[:1], y_coords + y_coords[:1], color='red', label='Círculo ajustado')

    # Graficar la union entre cada punto original y su punto correspondiente en el círculo ajustado
    for punto_circulo, puntos_originales in puntos_unidos.items():
        for punto_orig in puntos_originales:
            x_coords, y_coords = zip(punto_circulo, punto_orig)
            ax.plot(x_coords, y_coords, color='green')

    # Configuración del gráfico
    ax.set_aspect('equal', adjustable='datalim')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ajuste de círculo a los puntos')
    plt.grid()
    plt.show()

def plot_triangulo(puntos_originales, puntos_triangulo, puntos_unidos):
    """
    Grafica los puntos originales y los del triángulo ajustado
    """
    fig, ax = plt.subplots()

    # Graficar los puntos del triángulo ajustado
    x_coords, y_coords = zip(*puntos_triangulo)
    ax.scatter(x_coords, y_coords, color='red', label='Puntos modelados')
    ax.plot(x_coords + x_coords[:1], y_coords + y_coords[:1], color='red', label='Triángulo ajustado')

    # Graficar los puntos originales
    x_coords, y_coords = zip(*puntos_originales)
    ax.scatter(x_coords, y_coords, color='blue', label='Puntos originales')

    # Graficar la union entre cada punto original y su punto correspondiente en el triángulo ajustado
    for punto_triangulo, puntos_originales in puntos_unidos.items():
        for punto_orig in puntos_originales:
            x_coords, y_coords = zip(punto_triangulo, punto_orig)
            ax.plot(x_coords, y_coords, color='green')

    # Configuración del gráfico
    ax.set_aspect('equal', adjustable='datalim')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Ajuste de triángulo a los puntos')
    plt.grid()
    plt.show()