#Este script toma los hologramas generados con el DBS modificado, los reconstruye y muestra la amplitud y fase
import numpy as np
from PIL import Image as im
import torch 
import matplotlib.pyplot as plt  

m = 1080            #Tamaño del holograma binario.
o_size = 128        #Tamaño del objeto.
bs = 1080 // 4      #Desplazamiento horizontal de la región de interés donde se forma el objeto.

def tfourier(x):    #Función para sacar la transformada de Fourier.
    x=torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x)))
    return x

Modo = "Random" #"Random" #Cambiar esto entre "Lexicographic" y "Random" para analizar ambos casos

Iteraciones = np.arange(1, 7)

for iter in Iteraciones:
    #Carga los hologramas
    H = im.open("".join(["Hologramas modificado/", Modo, str(iter), "Iter.png"]))
    H = H.convert('1')
    H = np.asarray(H)
    H = torch.from_numpy(H)

    O = tfourier(H) #Reconstruye el holograma
    O = O[(m - o_size) // 2: (m + o_size) // 2, (m - o_size) // 2 + bs: (m + o_size) // 2 + bs] #Recorta la región de interés de la reconstrucción

    OABS = torch.abs(O) #Amplitud de la reconstrucción
    OABS = OABS / torch.max(OABS) * 255

    Ofase = torch.angle(O)
    Ofase = (Ofase + torch.pi) / (2 * torch.pi)
    Ofase = Ofase * 255

    plt.imshow(OABS, cmap = 'gray')
    plt.title("Amplitud")
    plt.show()

    plt.imshow(Ofase, cmap = 'gray')
    plt.title("Fase")
    plt.show()

    OABS = np.asarray(OABS)
    OABS = im.fromarray(OABS).convert('L')
    OABS.save("".join(["Resultados modificado/", Modo, str(iter), "Iter_Amplitud.png"]))

    Ofase = np.asarray(Ofase)
    Ofase = im.fromarray(Ofase).convert('L')
    Ofase.save("".join(["Resultados modificado/", Modo, str(iter), "Iter_Fase.png"]))


