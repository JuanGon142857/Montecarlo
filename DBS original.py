#Algoritmo DBS tradicional que permite generar un holograma binario donde se controla la amplitud de su reconstrucción (transformada de Fourier).
import numpy as np
from PIL import Image as im
import torch 
import matplotlib.pyplot as plt  
import torch.nn
import time

path_load = "Imagenes originales/"  #Carpeta donde están los imágenes originales.

threshold = 2e-2    #Threshold que determina cuando terminar el algoritmo.

m = 1080            #Tamaño del holograma binario.
o_size = 256        #Tamaño del objeto.
bs = 1080 // 4      #Desplazamiento horizontal de la región de interés donde se forma el objeto.

device = "cpu"     #"cuda" es la arquitectura de cáculo paralelo de NVIDIA, cambiar esto a "cpu" en caso de ocasionar errores.

Modo = "Lexicographic" #"Random" #Esto define el modo en que se van a analizar los píxeles del holograma.

def tfourier(x):    #Función para sacar la transformada de Fourier.
    x=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
    return x

H = (torch.rand((m, m)) * 2) // 1 #Holograba binario aleatorio.
H = H.to(device)

loss = torch.nn.MSELoss() #Definimos la función para calcular el error cuadrático medio.

#Cargamos la imagenes en escala de grises que se codificará en la amplitud y lo redimensionamos al tamaño deseado.
o = im.open("".join([path_load, "/1.bmp"]))
o = o.convert('L')
o = o.resize((o_size, o_size))
o = np.asarray(o)
o = o / 255.
o = torch.from_numpy(o)
o = o.to(device)

t0 = time.time() #Tiempo inicial

R_prev = tfourier(H) #Propagamos el campo
R_prev = R_prev[(m - o_size) // 2: (m + o_size) // 2, (m - o_size) // 2 + bs: (m + o_size) // 2 + bs] #Recortamos la región de interés de la reconstrucción.
R_prev = abs(R_prev) #Toma la amplitud de la reconstruccón
R_prev = R_prev / torch.max(R_prev) #Normalizamos la amplitud en la región de interes

MSE_prev = loss(R_prev, o) #Evalua el error cuadrático medio

Indexes = np.arange(m ** 2) #Definimos una lista de índices que determinarán el orden en el que evaluamos los píxeles del holograma.


iter = 0
while MSE_prev > threshold: #Mientras la métrica tenga un valor mayor al límite predefinido
    iter += 1
    print("Iteración: " + str(iter)) 
    if Modo == "Random": #Analiza el modo en que se analizaran los pixeles
        np.random.shuffle(Indexes) #Randomiza el orden de los píxeles para evaluar el holograma. 
    for i in np.arange(len(Indexes)):
        Iy = Indexes[i] // m
        Ix = Indexes[i] % m #Calculamos las coordenadas del píxel a evaluar de acuerdo al índice

        H[Iy][Ix] = not(H[Iy][Ix])  #Cambia el valor de los píxeles en cuestión

        R_new = tfourier(H) #Propagamos el campo
        R_new = R_new[(m - o_size) // 2: (m + o_size) // 2 , (m - o_size) // 2  + bs: (m + o_size) // 2 + bs] #Recortamos la región de interés de la reconstrucción.
        R_new = abs(R_new) #Toma la amplitud de la reconstruccón
        R_new = R_new / torch.max(R_new) #Normalizamos la amplitud en la región de interes

        MSE_new = loss(R_new, o) #Evalua el error cuadrático medio

        if MSE_new <= MSE_prev: #Evalua si la métrica indica que el cambio mejoró la reconstrucción.
            MSE_prev = MSE_new #Cambiamos el valor con el que comparara la métrica de ahora en adelante.
        else:
            H[Iy][Ix] = not(H[Iy][Ix]) #Si el cambio del valor en el píxel no mejora la reconstrucción lo revertimos

    t1 = time.time()
    print("MSE: ", MSE_prev) #Indica el valor de la métrica luego de cada iteración
    print("Time : " + str(t1 - t0)) #Indica el tiempo de cómputo total

    #Toma el holograma, lo transforma en una imagen binario y lo guarda 
    Hsave = H.cpu()
    Hsave = np.asarray(Hsave)
    Hsave = im.fromarray(Hsave).convert('1')
    Hsave.save("".join(["Hologramas original/", Modo, str(iter), "Iter.png"]))
    print("MSE: ", MSE_prev)


R = tfourier(H)
R = R[(m - o_size) // 2: (m + o_size) // 2, (m - o_size) // 2 + bs: (m + o_size) // 2 + bs]
R = abs(R)
R = R / torch.max(R)

#Muestra la amplitud final
plt.imshow(R.cpu(), cmap = 'gray')
plt.title("Reconstruccion")
plt.show()