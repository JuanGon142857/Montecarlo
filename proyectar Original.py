#Este script toma los hologramas generados con el DBS original, los reconstruye y muestra la amplitud
import numpy as np
from PIL import Image as im
import torch 
import matplotlib.pyplot as plt  

m = 1080            #Tamaño del holograma binario.
o_size = 256        #Tamaño del objeto.
bs = 1080 // 4      #Desplazamiento horizontal de la región de interés donde se forma el objeto.

def tfourier(x):    #Función para sacar la transformada de Fourier.
    x=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
    return x

Modo = "Random" #"Random" #Cambiar esto entre "Lexicographic" y "Random" para analizar ambos casos

Iteraciones = np.arange(1, 7)

for iter in Iteraciones:
    #Carga los hologramas
    H = im.open("".join(["Hologramas original/", Modo, str(iter), "Iter.png"])) 
    H = H.convert('1')
    H = np.asarray(H)
    H = torch.from_numpy(H)

    O = tfourier(H) #Reconstruye el holograma
    O = O[(m - o_size) // 2: (m + o_size) // 2, (m - o_size) // 2 + bs: (m + o_size) // 2 + bs] #Recorta la región de interés de la reconstrucción

    plt.imshow(torch.abs(O), cmap = 'gray')
    plt.title("Amplitud")
    plt.show()

