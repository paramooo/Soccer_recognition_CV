import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def marcar_campo(imagen):
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen , cv2.COLOR_BGR2HSV)

    # Definir rango de color verde en HSV
    verde_bajo = np.array([35, 20, 0])
    verde_alto = np.array([70, 255, 255])

    # Crear máscara con valores dentro del rango verde
    mascara = cv2.inRange(hsv, verde_bajo, verde_alto)

    #Aplicar open 
    kernel = np.ones((20,20),np.uint8) 
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    #Filtro de medianas grande para hacer mas suave el borde del campo y de paso quitar ruido de la grada
    mascara = cv2.medianBlur(mascara, 95)

    #Open para eliminar algun ruido si quedó por fuera
    kernel = np.ones((20,20),np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos de la máscara
    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Crear una máscara vacía para dibujar los contornos rectos
    mascara_campo = np.zeros_like(mascara)

    # Obtener el poligono del contorno
    epsilon = 0.01*cv2.arcLength(contours[0],True)
    poligono = cv2.approxPolyDP(contours[0],epsilon,True)

    # Dibujar la aproximación poligonal en la máscara
    cv2.drawContours(mascara_campo, [poligono], -1, (255), -1)

    #Aplicar la máscara a la imagen original
    segmented_image = cv2.bitwise_and(imagen, imagen, mask=mascara_campo)

    return segmented_image, poligono



def marcar_jugadores(imagen):
    # Convertir a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Extraer componente V
    v = hsv[:,:,2]
    
    # Realizar la transformación top-hat y black-hat y combinarlas
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, se)
    blackhat = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, se)
    combined = cv2.addWeighted(tophat, 0.6, blackhat, 0.4, 0)

    # Aplicar k-means para separar los jugadores del fondo
    z = np.float32(combined.reshape((-1,1)))
    c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(z, 2, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
    jugadores = np.uint8(centers[labels.flatten()]).reshape((combined.shape))

    # Copia de las etiquetas
    n_labels = np.copy(labels)

    # Pasar a binario la imagen del kmeans
    n_labels [labels == np.argmin(centers)] = 0
    n_labels [labels == np.argmax(centers)] = 255
    
    # Pasar a binario la imagen del kmeans
    jugadores = np.uint8(n_labels).reshape((combined.shape))

    # Recortar lineas y ruido con dilate y luego erode
    kernel = np.ones((10, 10),np.uint8)
    jugadores = cv2.dilate(jugadores,kernel,iterations = 1)

    kernel = np.ones((27, 3),np.uint8)
    jugadores = cv2.erode(jugadores,kernel,iterations = 1)

    # Encontrar los contornos en la imagen binaria
    contornos, _ = cv2.findContours(jugadores, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Crear una copia de la imagen original para dibujar los rectángulos
    imagen_con_rectangulos = np.copy(imagen)

    # Recorrer cada contorno
    for contorno in contornos:
        # Si el área del contorno está dentro del rango deseado
        if 140 < cv2.contourArea(contorno) < 2000:
            # Calcular el rectángulo del contorno
            x, y, w, h = cv2.boundingRect(contorno)

            # Si el ancho del contorno es menor que 80 px y el alto es menor que 150 px
            if w < 80 and h < 150:
                # Calcular el rectángulo del contorno
                x, y, w, h = cv2.boundingRect(contorno)

                # Dibujar el rectángulo en la imagen
                cv2.rectangle(imagen_con_rectangulos, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return imagen_con_rectangulos 

def marcar_segas(campo, poligono):
    # Convertir a HSV
    hsv = cv2.cvtColor(campo, cv2.COLOR_BGR2HSV)

    # Filtro gaussiano y de mediana para suavizar la imagen
    hsv = cv2.GaussianBlur(hsv, (19, 19), 0)
    hsv = cv2.medianBlur(hsv, 23)

    # Extraer componente S
    s = hsv[:,:,1]

    # Equalizar histograma
    s = cv2.equalizeHist(s)

    # Canny para detectar bordes
    canny = cv2.Canny(s, 80, 100)

    # Hough para detectar lineas rectas
    lines = cv2.HoughLinesP(canny, 1, np.pi/150, 70, minLineLength=200, maxLineGap=300)

    # Hacer media de orientacion de las lienas
    orientacion_media = 0
    num_lines = 0
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calcular la orientación de la línea
            orientacion = np.arctan2(y2-y1, x2-x1)

            # Ignorar las líneas que son casi horizontales (entre -30 y 30 grados)
            if abs(np.degrees(orientacion)) > 30:
                orientacion_media += orientacion
                num_lines += 1
    if num_lines > 0:
        orientacion_media /= num_lines

    # Crear una imagen vacía para dibujar las líneas
    campo = np.copy(campo)  

    # Dibujar lineas
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calcular la orientación de la línea
            orientacion = np.arctan2(y2-y1, x2-x1)

            #Estirar la linea para que llegue a los bordes de la imagen
            x1 = int(x1 - 1000*np.cos(orientacion))
            y1 = int(y1 - 1000*np.sin(orientacion))
            x2 = int(x2 + 1000*np.cos(orientacion))
            y2 = int(y2 + 1000*np.sin(orientacion))

            # Calcular puntos de corte con la imagen
            _, (_, y1c), (_, y2c) = cv2.clipLine((0, 0, campo.shape[1], campo.shape[0]), (x1, y1), (x2, y2))

            # Las que las y2-y1 se mayor de 200 y mantengan una orientacion similar a las demas pintar 
            if abs(y2c-y1c) > 200 and np.degrees(abs(orientacion - orientacion_media)) < 34:
                    # Dibujar la línea en la imagen
                    cv2.line(campo, (x1, y1), (x2, y2), (255, 0, 0), 2)

    mascara_campo = np.zeros_like(campo[:,:,0])

    # Dibujar la la mascara del terreno de juego
    cv2.drawContours(mascara_campo, [poligono], -1, (255), -1)

    #Aplicar la máscara a la imagen original
    campo_lineas = cv2.bitwise_and(campo, campo, mask=mascara_campo)

    return campo_lineas


def evaluar_marcar_campo():
    # Listas para almacenar los resultados
    resultados_imagen = []
    resultados_imagen_rec = []
    
    for num in range(16):
        if num in (0,1,2,3,10,11,12,13,14,15):
            # Cargar imagen
            imagen = cv2.imread("Material/" + str(num) + ".jpg")
            imagen_rec = cv2.imread("Material/MaterialRecortado/" + str(num) + ".jpg")

            # Eliminar fondo y dejar solo el campo
            imagen_campo, _ = marcar_campo(imagen)

            # Convertir las imágenes a escala de grises
            imagen_gray = cv2.cvtColor(imagen_campo, cv2.COLOR_BGR2GRAY)
            imagen_rec_gray = cv2.cvtColor(imagen_rec, cv2.COLOR_BGR2GRAY)

            # Pasar los umbrales para obtener imágenes binarias
            _, imagen_bin = cv2.threshold(imagen_gray, 0, 1, cv2.THRESH_BINARY)
            _, imagen_rec_bin = cv2.threshold(imagen_rec_gray, 0, 1, cv2.THRESH_BINARY)

            # Agregar a los resultados las matrices aplanadas
            resultados_imagen.extend(imagen_bin.flatten())
            resultados_imagen_rec.extend(imagen_rec_bin.flatten())

    # Calcular la matriz de confusión general
    cm_general = confusion_matrix(resultados_imagen, resultados_imagen_rec)

    # Imprimir la matriz de confusión general
    print("Evaluacion recortar terreno de juego:")
    print(f"Matriz de confusión:\n{cm_general}\n")
    
    # Extraer los valores de la matriz de confusión
    _, FP, FN, VP = cm_general.ravel()

    # Calcular el total de la muestra en la región ST
    ST = VP + FP

    # Calcular la fraccion falsa
    fraccion_falsa = 1 - ((FP + FN) / ST)
    print(f"Fracción Falsa: {format(fraccion_falsa, '.4f')}")

    # Calcular la precisión
    precision = VP / (VP + FP)  
    print(f"Precisión: {format(precision, '.4f')}")

    # Calcular el sensibilidad
    sensibilidad = VP / (VP + FN)
    print(f"Sensibilidad: {format(sensibilidad, '.4f')}")

    # Calcular el coeficiente de Similaridad de Dice
    dsc = 2 * VP / (2 * VP + FP + FN)
    print(f"Dice: {format(dsc, '.4f')}")



if __name__ == "__main__":
    evaluar_marcar_campo()
    # Cargar imagenes
    for num,filename in enumerate(os.listdir("Material/")):
        #Si no es una carpeta
        if not os.path.isdir("Material/"+filename):
            # Cargar imagen
            imagen = cv2.imread("Material/"+filename)

            # Eliminar fondo y dejar solo el campo
            campo, poligono = marcar_campo(imagen)
        
            # Segmentar jugadores
            jugadores = marcar_jugadores(campo)

            # Marcar segas campo
            campo_s = marcar_segas(campo, poligono)


            #Mostrar en una tabla la imagen original y las diferentes imagenes con los filtros
            fig, axs = plt.subplots(2, 2, figsize=(15, 5))
            axs[0,0].imshow(imagen)
            axs[0,0].set_title('Campo Original')
            axs[0,1].imshow(campo)
            axs[0,1].set_title('Campo con fondo eliminado')
            axs[1,0].imshow(jugadores)
            axs[1,0].set_title('Campo con jugadores marcados')
            axs[1,1].imshow(campo_s)
            axs[1,1].set_title('Campo con líneas marcadas')
            plt.show()
