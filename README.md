# Lab-Project-Computer-Vision
## Camera Calibration:
### Intrinsics:
 [[1.47111529e+03 0.00000000e+00 6.74519617e+02]
 [0.00000000e+00 1.46289971e+03 2.52198127e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]
### Distortion coefficients:
 [[ 0.02311053  0.26296089 -0.01937755  0.00569353 -0.26022924]]
### Root mean squared reprojection error:
 0.7224477124263834

## Shape Detection:
### detect_bright_object(imagen): devuelve la posición del objeto brillante
### detectar_circulo(imagen): devuelve true (false) si (no) encuentra un círculo
### detectar_triangulo(imagen): devuelve true (false) si (no) encuentra un triángulo
### detectar_octogono(imagen): devuelve true (false) si (no) encuentra un octógono rojo

## Shape Adjustment:
### euclidean_distance(punto1, punto2): devuelve la distancia euclídea entre 2 puntos
### ajustar_circulo(puntos): devuelve el radio y el centro del círculo que se ajusta mejor a los puntos
### ajustar_triangulo(puntos): devuelve los vértices del triángulo que se ajusta mejor a los puntos
### generar_puntos_circulo(centro, radio, numero_puntos): devuelve el círculo definido por el centro y el radio, pero dividido en un número determinado de puntos
### generar_puntos_triangulo(triangulo, numero_puntos): devuelve el triángulo dividido en un número determinado de puntos
### calcular_semejanza_circulo(puntos_originales, puntos_circulo, radio): devuelve un ratio calculado a partir de la distancia entre puntos originales y modelados
### calcular_semejanza_triangulo(puntos_originales, puntos_circulo): devuelve un ratio calculado a partir de la distancia entre puntos originales y modelados

## Light Follower:
### No se definen funciones, sino que se implementa la lógica necesaria haciendo uso de las funciones anteriores. Con ejecutar este archivo es suficiente.
