#Algoritmo DBS modificado que permite generar un holograma binario donde se controla la amplitud y fase de su reconstrucción (transformada de Fourier).
import numpy as np
from PIL import Image as im
import torch 
import matplotlib.pyplot as plt  
import torch.nn
import time

path_load = "Imagenes originales/"  #Carpeta donde están los objetos originales.

threshold = 4e-2    #Threshold que determina cuando terminar el algoritmo.

m = 1080            #Tamaño del holograma binario.
o_size = 128        #Tamaño del objeto.
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
o[o < 0.1] = 0.1    #Eliminamos los valores nulos ya que estos ocasionan problemas para luego codificar en la fase.
o = torch.from_numpy(o)
o = o.to(device)

#Cargamos la imagenes en escala de grises que se codificará en la fase y lo redimensionamos al tamaño deseado.
of = im.open("".join([path_load, "/2.bmp"]))
of = of.convert('L')
of = of.resize((o_size, o_size))
of = np.asarray(of)
of = of / 255.
of[of < 0.1] = 0.1
of[of > 0.9] = 0.9 #Quitamos los valores cerca de los extremos superior e inferior para evitar posibles problemas de convergencia por la discontinuidad de la fase.
of = torch.from_numpy(of)
of = of.to(device)

c = o * torch.exp(1j * 2 * torch.pi * (of - 0.5))   #Definimos el campo complejo.

c_IM = torch.imag(c)
c_RE = torch.real(c)    #Separamos la parte real e imaginaria del campo complejo.

c_IM = c_IM + torch.abs(torch.min(c_IM))
c_RE = c_RE + torch.abs(torch.min(c_RE))    #Eliminamos los valores negativos de cada conjunto, esto ayuda con la convergencia del algoritmo.

t0 = time.time() #Tiempo inicial

loss = torch.nn.MSELoss() #Definimos la función para calcular el error cuadrático medio.

R_prev = tfourier(H) #Propagamos el campo
R_prev = R_prev[(m - o_size) // 2: (m + o_size) // 2, (m - o_size) // 2 + bs: (m + o_size) // 2 + bs] #Recortamos la región de interés de la reconstrucción.
R_prev = R_prev / torch.max(abs(R_prev)) #Normalizamos la amplitud en la región de interes

R_prev_re = torch.real(R_prev)
R_prev_im = torch.imag(R_prev)  #Separamos la parte real e imaginaria de la reconstrucción

R_prev_re = R_prev_re + torch.abs(torch.min(R_prev_re))
R_prev_im = R_prev_im + torch.abs(torch.min(R_prev_re)) #Eliminamos los valores negativos de cada conjunto, esto ayuda con la convergencia del algoritmo.

MSE_prev = loss(R_prev_re, c_RE) + loss(R_prev_im, c_IM) #Calculamos el error cuadrático medio entre las partes reales, entre las partes imaginarias y lo sumamos

Indexes = np.arange(m ** 2) #Definimos una lista de índices que determinarán el orden en el que evaluamos los píxeles del holograma


iter = 0
while MSE_prev > threshold: #Mientras la métrica tenga un valor mayor al límite predefinido
    iter +=1
    print("Iteración: " + str(iter))
    if Modo == "Random": #Analiza el modo en que se analizaran los pixeles
        np.random.shuffle(Indexes) #Randomiza el orden de los píxeles para evaluar el holograma. 
    for i in np.arange(len(Indexes)):
        Iy = Indexes[i] // m
        Ix = Indexes[i] % m #Calculamos las coordenadas del píxel a evaluar de acuerdo al índice

        H[Iy][Ix] = not(H[Iy][Ix])  #Cambia el valor de los píxeles en cuestión

        R_new = tfourier(H) #Propagamos el campo
        R_new = R_new[(m - o_size) // 2: (m + o_size) // 2 , (m - o_size) // 2  + bs: (m + o_size) // 2 + bs] #Recortamos la región de interés de la reconstrucción.
        R_new = R_new / torch.max(torch.abs(R_new)) #Normalizamos la amplitud en la región de interes

        R_new_re = torch.real(R_new)
        R_new_im = torch.imag(R_new) #Separamos la parte real e imaginaria de la reconstrucción

        R_new_re = R_new_re + torch.abs(torch.min(R_new_re))
        R_new_im = R_new_im + torch.abs(torch.min(R_new_im)) #Eliminamos los valores negativos de cada conjunto, esto ayuda con la convergencia del algoritmo.

        MSE_new = loss(R_new_re, c_RE) + loss(R_new_im, c_IM) #Evalua la métrica

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
    Hsave.save("".join(["Hologramas modificado/", Modo, str(iter), "Iter.png"]))

R = tfourier(H)
R = R[(m - o_size) // 2: (m + o_size) // 2, (m - o_size) // 2 + bs: (m + o_size) // 2 + bs]

Rf = torch.angle(R)
Rf = (Rf + torch.pi) / (2 * torch.pi)
Rf = Rf * 255

R = torch.abs(R)
R = R / torch.max(R) * 255

#Muestra la amplitud y fase final
plt.imshow(R, cmap = 'gray')
plt.title("Amplitud")
plt.show()

plt.imshow(Rf, cmap = 'gray')
plt.title("Fase")
plt.show()                                          

R = np.asarray(R)
R = im.fromarray(R).convert('L')
R.save("".join(["Resultados modificado/", Modo, str(iter), "Iter_Amplitud.png"]))

Rf = np.asarray(Rf)
Rf = im.fromarray(Rf).convert('L')
Rf.save("".join(["Resultados modificado/", Modo, str(iter), "Iter_Fase.png"]))
