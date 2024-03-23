# Montecarlo

DBS original.py es el código con el que se generan los hologramas binarios donde se controla su amplitud y los guarda en la carpeta Hologramas original.
DBS modifaco.py es el código con el que se generan los hologramas binarios donde se controla su amplitud y su fase y los guarda en la carpeta Hologramas modificado.

En ambos códigos se tiene una variable llamada Modo que puede cambiarse entre "Lexicographic" y "Random" para decidir el orden de análisis de los píxeles del holograma.
En ambos códigos se tiene una variable llamada device que es "cpu" por defecto pero, si se tiene una GPU compatible, esta variable puede cambiarse por "cuda" para acelerar el proceso.

proyectar Original.py es el código que toma los hologramas generados por DBS original.py y muestra su fase 
