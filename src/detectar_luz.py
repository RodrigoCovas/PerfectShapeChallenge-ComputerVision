from picamera2 import Picamera2, Preview
import cv2
import numpy as np

# Inicializar la cámara
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (1280, 720)}, buffer_count=3)
picam2.configure(config)
picam2.start()

# Configurar la grabación de video
video_output = cv2.VideoWriter('grabacion_luz.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))

posiciones = []  # Lista para almacenar posiciones de la luz

try:
    while True:
        # Capturar el frame de la cámara
        frame = picam2.capture_array()

        # Convertir el frame a escala de grises
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Aplicar un umbral para detectar regiones brillantes
        _, umbral = cv2.threshold(gris, 200, 255, cv2.THRESH_BINARY)

        # Encontrar contornos de las regiones brillantes
        contornos, _ = cv2.findContours(umbral, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            # Filtrar contornos pequeños para evitar ruido
            if cv2.contourArea(contorno) > 50:  # Ajusta el área mínima según sea necesario
                # Calcular el centroide del contorno
                M = cv2.moments(contorno)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    posiciones.append((cx, cy))  # Guardar posición

                    # Dibujar el contorno y el centro en el frame
                    cv2.drawContours(frame, [contorno], -1, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Grabar el frame procesado
        video_output.write(frame)

        # Mostrar el frame en tiempo real
        cv2.imshow('Detección de luz en tiempo real', frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Detener la cámara y liberar recursos
    picam2.stop()
    video_output.release()
    cv2.destroyAllWindows()

    # Guardar posiciones en un archivo de texto
    with open('posiciones.txt', 'w') as f:
        for pos in posiciones:
            f.write(f"{pos[0]},{pos[1]}\n")

    print("Grabación finalizada. Posiciones almacenadas en 'posiciones.txt'")

