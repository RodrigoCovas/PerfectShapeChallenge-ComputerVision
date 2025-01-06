import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


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

def ajustar_triangulo(puntos):
    
    max_area = 0
    vertices = None

    num_puntos = len(puntos)
    for i in range(num_puntos):
        for j in range(i + 1, num_puntos):
            for k in range(j + 1, num_puntos):
                p1, p2, p3 = puntos[i], puntos[j], puntos[k]
                area = 0.5 * abs(
                    p1[0] * (p2[1] - p3[1]) +
                    p2[0] * (p3[1] - p1[1]) +
                    p3[0] * (p1[1] - p2[1])
                )
                
                if area > max_area:
                    max_area = area
                    vertices = [p1, p2, p3]
    
    return vertices


def generar_puntos_circulo(cx, cy, r, n_puntos):
    """
    Genera n_puntos uniformemente distribuidos en el círculo ajustado
    """
    angulos = np.linspace(0, 2 * np.pi, n_puntos, endpoint=False)
    puntos_circulo = [(cx + r * np.sin(theta), cy - r * np.cos(theta)) for theta in angulos]
    return puntos_circulo

def interseccionRectas(r,s):
    x1=r[1][1]-r[0][1]
    y1=r[0][0]-r[1][0]
    x2=s[1][1]-s[0][1]
    y2=s[0][0]-s[1][0]
    if x1*y2-x2*y1==0:
        print(f"\n* Rectas paralelas: {r},{s}", end="")
    else:
        z1=r[0][0]*r[1][1]-r[0][1]*r[1][0]
        z2=s[0][0]*s[1][1]-s[0][1]*s[1][0]
        x=(z1*y2-z2*y1)/(x1*y2-x2*y1)
        y=(x1*z2-x2*z1)/(x1*y2-x2*y1)
        return [x,y]

def generar_puntos_triangulo(vertices, puntos_originales):
    # Para cada punto original, se calcula la distancia a cada uno de los vértices del triángulo
    # Se asigna el punto original a los dos vértices más cercanos
    # Se busca el punto de intersección entre la recta que une los dos vértices más cercanos y su recta normal que pasa por el punto original
    # Se añade el punto de intersección a la lista de puntos generados a partir de los vértices del triángulo
    puntos_triangulo = puntos_originales.copy()
    # Quitar los vértices de la lista de puntos originales
    puntos_sin_vertices = [punto for punto in puntos_triangulo if punto not in vertices]
    for punto_orig in puntos_sin_vertices:
        distancias = [euclidean_distance(punto_orig, vertice) for vertice in vertices]
        vertices_cercanos = [vertices[i] for i in np.argsort(distancias)[:2]]
        lado = [vertices_cercanos[0], vertices_cercanos[1]]
        pendiente_lado = (lado[1][1] - lado[0][1]) / (lado[1][0] - lado[0][0])
        if pendiente_lado != 0:
            normal = [[punto_orig[0], punto_orig[1]], [punto_orig[0] + 1, punto_orig[1] - 1/pendiente_lado]]
        elif pendiente_lado == 0:
            normal = [[punto_orig[0], punto_orig[1]], [punto_orig[0], punto_orig[1] + 1]]
        punto_interseccion = interseccionRectas(lado, normal)
        # Sustituir el punto_orig por el punto_interseccion en la lista puntos_originales
        puntos_triangulo[puntos_triangulo.index(punto_orig)] = tuple(punto_interseccion)
    return puntos_triangulo

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


    #Elevamos las distancias al cuadrado
    distancias = [distancia**3 for distancia in distancias]
    media_distancia = np.mean(distancias)

    #Calculamos la semejanza
    semejanza = 1 / (1 + media_distancia/((radio**3)/200))

    return semejanza, puntos_unidos

def calcular_semejanza_triangulo(puntos_originales, puntos_generados, vertices):
    """
    Calcula la semejanza entre los puntos originales y los del triángulo ajustado
    """
    distancias = []
    puntos_unidos = {punto: [] for punto in puntos_generados}
    for punto_orig in puntos_originales:
        distancias_punto = [euclidean_distance(punto_orig, punto_triangulo) for punto_triangulo in puntos_generados]
        distancia_minima = min(distancias_punto)
        distancias.append(distancia_minima)
        punto_cercano = puntos_generados[distancias_punto.index(distancia_minima)]
        puntos_unidos[punto_cercano].append(punto_orig)

    # Elevamos las distancias al cubo
    distancias = [distancia**3 for distancia in distancias]
    media_distancia = np.mean(distancias)

    # Calculamos la semejanza
    semejanza = 1 / (1 + media_distancia/((euclidean_distance(vertices[0], vertices[1])**3)/200))

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