# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 11:57:26 2023

@author: Ali Sanchez Garcia
"""

import cv2 
import numpy as np 
import numpy as np 
import matplotlib.pyplot as plt
from skimage import data,io,measure,color,morphology
from PIL import Image
################################################################################################################################################################
def recortar_imagen(ruta_imagen, nombrar, nuevo_ancho, nuevo_alto):
    # Leer la imagen
    img = cv2.imread(ruta_imagen)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Aplicar un umbral para obtener una imagen binaria
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Encontrar los contornos de la imagen binaria
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Encontrar el contorno más grande, que debería ser el borde de la imagen
    largest_contour = max(contours, key=cv2.contourArea)

    # Encontrar las coordenadas del rectángulo que rodea el contorno
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Recortar la imagen utilizando las coordenadas del rectángulo
    cropped_img = img[y:y+h, x:x+w]

    # Redimensionar la imagen
    resized_img = cv2.resize(cropped_img, (nuevo_ancho, nuevo_alto))

    # Convertir la imagen de BGR a RGB
    rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

    # Guardar la imagen como jpg
    cv2.imwrite(nombrar, rgb_img)

    return resized_img

##############################################################################################################################################################
plt.close('all')

recortada1=recortar_imagen('foto_1.jpg', 'ficha1.jpg',461,217)
recortada2=recortar_imagen('foto_2.jpg', 'ficha2.jpg',461,217)
recortada3=recortar_imagen('foto_3.jpg', 'ficha3.jpg',461,217)
recortada4=recortar_imagen('foto_4.jpg', 'ficha4.jpg',461,217)
recortada5=recortar_imagen('foto_5.jpg', 'ficha5.jpg',461,217)
recortada6=recortar_imagen('foto_6.jpg', 'ficha6.jpg',461,217)

##############################################################################################################################################################
# Imagen original
img1= cv2.imread('ficha1.jpg', cv2.IMREAD_COLOR) 
img2= cv2.imread('ficha2.jpg', cv2.IMREAD_COLOR) 
img3= cv2.imread('ficha3.jpg', cv2.IMREAD_COLOR) 
img4= cv2.imread('ficha4.jpg', cv2.IMREAD_COLOR) 
img5= cv2.imread('ficha5.jpg', cv2.IMREAD_COLOR) 
img6= cv2.imread('ficha6.jpg', cv2.IMREAD_COLOR) 
##############################################################################################################################################################
# Convertir a escala de grises
gris1= cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) 
gris2= cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
gris3= cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY) 
gris4= cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY) 
gris5= cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY) 
gris6= cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY) 
##############################################################################################################################################################
def deteccion_circulos(gris,original):
    """
    Función que detecta círculos en una imagen utilizando la transformada de Hough.
    
    Argumentos:
    gris -- La imagen en la que se buscan los círculos.
    original -- La imagen original.
    
    Retorna:
    i -- El número de círculos encontrados en la imagen.
    """
    filtros= cv2.blur(gris, (9, 9)) # filtro pasabajas de 3x3
    # Aplicar la tranfromada de Hough para detección de círculos
    detected_circless = cv2.HoughCircles(filtros, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=40)
    # Utilizar la función cv2.HoughCircles para detectar círculos.
    #"filtros" es la imagen de entrada en la que se buscan los círculos.
    #"cv2.HOUGH_GRADIENT" especifica el método de detección de círculos.
    #"1" es el factor de escala de la imagen.Relación inversa de la resolución del acumulador casi seimpre se usa 1 porque se tiene la misma resolución que la imagen.
    #"20" es la distancia mínima entre los círculos detectados.
    # param1,siempre para HOUGH_GRADIENT, es el umbral máximo en la detección de bordes por Canny.
    # param2, para el método HOUGH_GRADIENT, es el umbral mínimo en la detección de bordes por Canny.
    #"minRadius=1" es el radio mínimo del círculo a detectar.
    #"maxRadius=40" es el radio máximo del círculo a detectar.
    
    # Revisar que se hayan detectado círculos
    if detected_circless is not None:
        # Convertir los parámetros a enteros de 16 bits
        detected_circless = np.uint16(np.around(detected_circless))
        i = 0
        
        # Analizar todos los círculos detectados
        for pt in detected_circless[0, :]:
            # Contar la cantidad de círculos
            i = i+1
        return i
    else:
        i=0
        return i
##############################################################################################################################################################
#SEGMENTACION

obje1_1=gris1[:, 0:230]
obje2_1=gris1[:, 240:500]

obje1_2=gris2[:, 0:230]
obje2_2=gris2[:, 240:500]

obje1_3=gris3[:, 0:230]
obje2_3=gris3[:, 240:500]

obje1_4=gris4[:, 0:230]
obje2_4=gris4[:, 240:500]

obje1_5=gris5[:, 0:230]
obje2_5=gris5[:, 240:500]

obje1_6=gris6[:, 0:230]
obje2_6=gris6[:, 240:500]
##############################################################################################################################################################
izq1=morphology.closing(obje1_1)#Limpiamos las imagne para que no este tan ruidosa
der1=morphology.closing(obje2_1)#Limpiamos las imagne para que no este tan ruidosa

izq2=morphology.closing(obje1_2)#Limpiamos las imagne para que no este tan ruidosa
der2=morphology.closing(obje2_2)#Limpiamos las imagne para que no este tan ruidosa

izq3=morphology.closing(obje1_3)#Limpiamos las imagne para que no este tan ruidosa
der3=morphology.closing(obje2_3)#Limpiamos las imagne para que no este tan ruidosa

izq4=morphology.closing(obje1_4)#Limpiamos las imagne para que no este tan ruidosa
der4=morphology.closing(obje2_4)#Limpiamos las imagne para que no este tan ruidosa

izq5=morphology.closing(obje1_5)#Limpiamos las imagne para que no este tan ruidosa
der5=morphology.closing(obje2_5)#Limpiamos las imagne para que no este tan ruidosa

izq6=morphology.closing(obje1_6)#Limpiamos las imagne para que no este tan ruidosa
der6=morphology.closing(obje2_6)#Limpiamos las imagne para que no este tan ruidosa
##############################################################################################################################################################
Izquierdo1=deteccion_circulos(izq1,img1)
Derecho1=deteccion_circulos(der1,img1)

Izquierdo2=deteccion_circulos(izq2,img2)
Derecho2=deteccion_circulos(der2,img2)

Izquierdo3=deteccion_circulos(izq3,img3)
Derecho3=deteccion_circulos(der3,img3)

Izquierdo4=deteccion_circulos(izq4,img4)
Derecho4=deteccion_circulos(der4,img4)

Izquierdo5=deteccion_circulos(izq5,img5)
Derecho5=deteccion_circulos(der5,img5)

Izquierdo6=deteccion_circulos(izq6,img6)
Derecho6=deteccion_circulos(der6,img6)
##############################################################################################################################################################

F1 = [Izquierdo1,Derecho1]
F2 = [Izquierdo2,Derecho2]
F3 = [Izquierdo3,Derecho3]
F4 = [Izquierdo4,Derecho4]
F5 = [Izquierdo5,Derecho5]
F6 = [Izquierdo6,Derecho6]
FF = [7,7]


print('Las fichas de la maquina son')
print(F1)
print(F2)
print(F3)
print(F4)
print(F5)
print(F6)

#Fichas_Maquina
F_M = [F1,F2,F3,F4,F5,F6,FF]

################## TOMA 5 FICHAS EL JUGADOR ####################

print("Cuales son tus fichas?")
# inicializa una lista con elementos nulos
F_J=[None]*7 
b=1
for a in range(0,6):
    print(f"ingresa la ficha {b}")
    b=b+1
    # x, y = input("X,Y = ").split()
    x = input("x = ")
    y = input("y = ")
    x = int(x)
    y = int(y)
    F_J[a] = [x,y]

#Fichas_Jugador
F_J[6] = [7,7] # Representa un espacio de más, si este cambia eljugador tiene
# más de 6 cartas y pierde, se utilizará más adelante
#print(F_J)

#################### ORIENTANDO FICHAS ####################

# Función que toma el valor más alto de una ficha y lo pone en la  segunda posición
# y el valor más bajo en la primera posición
def swap_max_min(F):
    max_num = max(F)
    min_num = min(F)
    max_index = F.index(max_num)
    min_index = F.index(min_num)
    F[max_index] = min_num
    F[min_index] = max_num
    return F

for i in range(0, 7):
    F_M[i] = swap_max_min(F_M[i])
for i in range(0, 7):
    F_J[i] = swap_max_min(F_J[i])

#################### BASE DE DATOS #######################

Ficha0_0 = [0,0]
Ficha0_1 = [0,1]
Ficha0_2 = [0,2]
Ficha0_3 = [0,3]
Ficha0_4 = [0,4]
Ficha0_5 = [0,5]
Ficha0_6 = [0,6]

Ficha1_1 = [1,1]
Ficha1_2 = [1,2]
Ficha1_3 = [1,3]
Ficha1_4 = [1,4]
Ficha1_5 = [1,5]
Ficha1_6 = [1,6]

Ficha2_2 = [2,2]
Ficha2_3 = [2,3]
Ficha2_4 = [2,4]
Ficha2_5 = [2,5]
Ficha2_6 = [2,6]

Ficha3_3 = [3,3]
Ficha3_4 = [3,4]
Ficha3_5 = [3,5]
Ficha3_6 = [3,6]

Ficha4_4 = [4,4]
Ficha4_5 = [4,5]
Ficha4_6 = [4,6]

Ficha5_5 = [5,5]
Ficha5_6 = [5,6]

Ficha6_6 = [6,6]

#Base_de_datos
BD = [Ficha0_0,Ficha0_1,Ficha0_2,Ficha0_3,Ficha0_4,Ficha0_5,Ficha0_6,
                 Ficha1_1,Ficha1_2,Ficha1_3,Ficha1_4,Ficha1_5,Ficha1_6,
                 Ficha2_2,Ficha2_3,Ficha2_4,Ficha2_5,Ficha2_6,
                 Ficha3_3,Ficha3_4,Ficha3_5,Ficha3_6,
                 Ficha4_4,Ficha4_5,Ficha4_6,
                 Ficha5_5,Ficha5_6,
                 Ficha6_6]

Mulas = [Ficha0_0,Ficha1_1,Ficha2_2,Ficha3_3,Ficha4_4,Ficha5_5,Ficha6_6]

################# CONDICION TENER 5 MULAS PARA PERDER ###############

contj = 0
for k in range(6,-1,-1):
    for i in range(0,7):
        if Mulas[k] == F_J[i]:
            contj = contj+1
            if contj >=5:
                print("Jugador tiene más de 5 mulas")
                print("Jugador pierde")
                raise ValueError("fin del juego")

contm = 0        
for k in range(6,-1,-1):
    for i in range(0,7):
        if Mulas[k] == F_M[i]:
            contm = contm+1
            if contm >=5:
                print("Maquina tiene más de 5 mulas")
                print("Maquina pierde")
                raise ValueError("fin del juego")

################# QUIEN EMPIEZA PRIMERO ###############

M = 0
J = 0
for k in range(6,-1,-1):
    while True:
        respuesta = input(f"¿Tienes una mula de {k}? (y/n) ")
        
        if respuesta == "y":
            for i in range(0,7): # Comprobando si dice la verdad
                a = 0
                if Mulas[k] == F_J[i]:
                    print("El jugador empieza primero")
                    J = 1
                    M = 0
                    a = 1
                    break
            if a == 0:
                print("Mentiroso")
                print("No tienes mula de 6")
                print("Jugador Pierde")
                M = 2
                J = 2
                break
            break
        
        
        elif respuesta == "n":
            for i in range(0,7):
                if Mulas[k] == F_M[i]:
                    print("La maquina Empieza primero")
                    M = 1
                    J = 0
                    break              
            break
        
        
        else:
            print("Respuesta inválida. Por favor, responde con 'y' o 'n'.")
    
    
    if J == 1 or M == 1:
        break
    elif J == 2 or M == 2:
        raise ValueError("fin del juego")
        break
    else:
        print(f"Ninguno tienen mula de {k}")
        if Mulas[k] == [0,0]:
            raise ValueError("fin del juego")
        

#################### PONER FICHA INICIAL ################

if M == 1 and J == 0:
    Li = F_M[i][0] # Lado izquierdo
    Ld = F_M[i][1] # Lado derecho
    Tablero = [Li,Ld] 
    F_M[i] = [7,7] # vacio porque la ficha se puso en el tablero 
    print("Tablero:")
    print(Tablero)
    print("Fin del turno de la maquina")
    print(" ")
    M = 0
    J = 1
elif M == 0 and J == 1:
    Li = F_J[i][0] # Lado izquierdo
    Ld = F_J[i][1] # Lado derecho
    Tablero = [Li,Ld] 
    F_J[i] = [7,7] 
    print("Tablero:")
    print(Tablero)
    print("Fin del turno del jugador")
    print(" ")
    J = 0
    M = 1

#################### PONER FICHAS #########################

z = 0
while True:
    if z == 0:
        f = [7,7]
        
        if (F_J[0]==[7,7]) and (F_J[1]==[7,7]) and (F_J[2]==[7,7]) and (F_J[3]==[7,7]) and (F_J[4]==[7,7]) and (F_J[5]==[7,7]) and (F_J[6]==[7,7]):
            print("El jugador no tiene ninguna fichas")
            print("El jugador gana")
            break
        
        if M == 1 and J == 0:
            print("Turno de la maquina")
            print(F_M)
            for i in range(0,7):
                for j in range(0,2):
                    
                    if Li == F_M[i][j]:
                        print("Ficha puesta:")
                        print(F_M[i])
                        f = F_M[i]
                        F_M[i] = [7,7]
                        M = 0
                        J = 1
                        print(F_M)
                        break
                    
                    elif Ld == F_M[i][j]:
                        print("Ficha puesta:")
                        print(F_M[i])
                        f = F_M[i]
                        F_M[i] = [7,7]
                        M = 0
                        J = 1
                        print(F_M)
                        break
                    
                if f != [7,7]:
                    if Li == f[0]:
                        Li = f[1]
                        Tablero = [Li,Ld]
                        print("Tablero:")
                        print(Tablero)
                        print(" ")
                        break

                    elif Ld == f[0]:
                        Ld = f[1]
                        Tablero = [Li,Ld]
                        print("Tablero:")
                        print(Tablero)
                        print(" ")
                        break
                    
                    elif Li == f[1]:
                        Li = f[0]
                        Tablero = [Li,Ld]
                        print("Tablero:")
                        print(Tablero)
                        print(" ")
                        break

                    elif Ld == f[1]:
                        Ld = f[0]
                        Tablero = [Li,Ld]
                        print("Tablero:")
                        print(Tablero)
                        print(" ")
                        break
                    
                    
            if f == [7,7]:
                print("La maquina no tiene fichas con numeros de algún lado")
                print("La maquina debe comer una ficha")
                print("ingresa la ficha")
                x = input("x = ")
                y = input("y = ")
                print(" ")
                x = int(x)
                y = int(y)
                for d in range(0,7):
                    if F_M[d] == [7,7]:
                        F_M[d] = [x,y]
                        if F_M[6] != [7,7]:
                            print("Maquina tiene más de 7 fichas")
                            print("La maquina pierde")
                            z = 1
                            break
                        else:
                            break

        if (F_M[0]==[7,7]) and (F_M[1]==[7,7]) and (F_M[2]==[7,7]) and (F_M[3]==[7,7]) and (F_M[4]==[7,7]) and (F_M[5]==[7,7]) and (F_M[6]==[7,7]):
            print("La maquina no tiene ninguna fichas")
            print("La maquina gana")
            break        

        elif J == 1 and M == 0:
            f = [7,7]
            print("Turno del jugador")
            print(F_J)
            while True:
                respuesta = input("¿Quieres poner alguna ficha? (y/n) ")
                if respuesta == "y":
                    print("ingresa la ficha")
                    x = input("x = ")
                    y = input("y = ")
                    print(" ")
                    x = int(x)
                    y = int(y)
                    a = 0
                    for i in range(0,7): # Comprobando 
                        if F_J[i] == [x,y] or F_J[i] == [y,x]:
                            a = 1
                            #print("Dectectada")
                            for j in range(0,2):
                                if Li == F_J[i][j]:
                                    print("Ficha puesta:")
                                    print(F_J[i])
                                    f = F_J[i]
                                    F_J[i] = [7,7]
                                    M = 1
                                    J = 0
                                    break
                                
                                elif Ld == F_J[i][j]:
                                    print("Ficha puesta:")
                                    print(F_J[i])
                                    f = F_J[i]
                                    F_J[i] = [7,7]
                                    M = 1
                                    J = 0
                                    break
                    
                    if f != [7,7]:
                        if Li == f[0]:
                            Li = f[1]
                            Tablero = [Li,Ld]
                            print("Tablero:")
                            print(Tablero)
                            print(" ")
                            a = 2
                            break
                        

                        elif Ld == f[0]:
                            Ld = f[1]
                            Tablero = [Li,Ld]
                            print("Tablero:")
                            print(Tablero)
                            print(" ")
                            a = 2
                            break
                        
                        elif Li == f[1]:
                            Li = f[0]
                            Tablero = [Li,Ld]
                            print("Tablero:")
                            print(Tablero)
                            print(" ")
                            a = 2
                            break
                        
                        

                        elif Ld == f[1]:
                            Ld = f[0]
                            Tablero = [Li,Ld]
                            print("Tablero:")
                            print(Tablero)
                            print(" ")
                            a = 2
                            break
                                
                        
            
                    if a == 0:
                        print("No tienes esa ficha")
                        respuesta = "n"
                        
                    elif a == 2:
                       print("Fin del turno del jugador")
                       print(" ")
                       break
                   
                
                
                elif respuesta == "n":
                    print("El jugador no tiene fichas con numeros de algún lado")  
                    print("El jugador debe comer una ficha")
                    print("ingresa la ficha")
                    x = input("x = ")
                    y = input("y = ")
                    print(" ")
                    x = int(x)
                    y = int(y)
                    for d in range(0,8):
                        if F_J[d] == [7,7]:
                            F_J[d] = [x,y]
                            #print(F_J)
                            if F_J[6] != [7,7]:
                                print("Jugador tiene más de 7 fichas")
                                print("El jugador pierde")
                                z = 1
                                a = 3
                                break
                            else:
                                break 
                    
                    if a == 3:
                      print(" ")
                      break 

                else:
                    print("Respuesta inválida. Por favor, responde con 'y' o 'n'.")        


    elif z == 1:
        print("Fin del juego")
        break        




    