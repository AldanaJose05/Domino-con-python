# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 08:20:24 2023

@author: Iguan
"""

import cv2

# inicializar la cámara
cap = cv2.VideoCapture(0)

# contador para el número de fotos tomadas
count = 0
# mostrar la vista previa de la cámara en tiempo real
while True:
    # capturar un cuadro de la cámara
    ret, frame = cap.read()

    # mostrar el cuadro capturado en una ventana de OpenCV
    cv2.imshow('Vista previa de la camara', frame)

    # esperar hasta que se presione la tecla 'espacio' para tomar una foto
    if cv2.waitKey(1) == ord(' '):
        # aumentar el contador de fotos tomadas
        count += 1

        # guardar la foto en el directorio actual
        filename = f'foto_{count}.jpg'
        cv2.imwrite(filename, frame)

    # cerrar la ventana de vista previa de la cámara si se presiona la tecla 'q'
    elif cv2.waitKey(1) == ord('q'):
        break

# cerrar la ventana de vista previa de la cámara
cv2.destroyAllWindows()

# liberar la cámara
cap.release()