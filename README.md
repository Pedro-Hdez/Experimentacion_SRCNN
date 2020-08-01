# Experimentacion_SRCNN

En este repositorio se encuentran el código fuente de mi proyecto de **Redes Neuronales Convolucionales** donde realizo algunos experimentos utilizando la red neuronal [*SRCNN*](https://arxiv.org/pdf/1501.00092.pdf) para aumentar la resolución de imágenes degradadas. Este trabajo llamado *Superresolución de imágenes* se encuentra en [mi blog](https://pedro-hdez.github.io/), te invito a echarle un vistazo.

El código fuente original (Matlab / Caffe) lo puedes encontrar [aquí](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html), y una versión del mismo (en el que yo me basé) pero traducido en Python / Tensorflow, se encuentra en [este repositorio](https://github.com/MarkPrecursor/SRCNN-keras).

## PREREQUISITOS

* Este proyecto fue desarrollado completamente bajo Linux (Ubuntu 20.04); por lo tanto, explicaré cómo usar los scripts con línea de comandos desde la  temrinal.

* Para facilitar todo el manejo de las librerías necesarias se hará uso de un Entorno de Anaconda. Para instalarlo en tu máquina sigue los pasos de la [guía oficial](https://docs.anaconda.com/anaconda/install/)

* Trabajaremos con imágenes, entonces es necesario contar con algunos ejemplos para entrenar y otros para predecir. Puedes usar tus propias fotos o visitar el Post donde desarrollo el proyecto donde encontrarás algunos links a DataSets que yo utilicé.

## CÓMO USARLO

A continuación, voy a presentarte una guía para mi código. Una vez instalado Anaconda, los pasos son los siguientes:

1. [Importar el entorno de Anaconda](#1-importar-el-entorno-de-anaconda)
2. [Degradar las imágenes](#Degradar-las-imágenes)
3. [Preparar los datos de entrenamiento](#Preparar-los-datos-de-entrenamiento)
4. [Entrenar la red neuronal](#Entrenar-la-red-neuronal)
5. [Reconstruir imágenes](#Reconstruir-imágenes)

Adicionalmente, algunos Scripts tienen la finalidad de analizar el desempeño de los modelos.

5. [Analizar los indicadores de desempeño](#Analizar-los-indicadores-de-desempeño)
6. [Comparar dos modelos](#Comparar-dos-modelos)
7. [Encontrar los mejores resultados](#Encontrar-los-mejores-resultados)


### 1 Importar el entorno de Anaconda

Para usar los scripts es necesario tener instaladas algunas librerías como *Pandas*, *NumPy* y *OpenCV*; además de Tensorflow y Python 3.X. Todas estas especificaciones se encuentran en el archivo **experimentacion_srcnn.yml**. Anaconda automáticamente creará un entorno llamado <ins>*experimentacion_srcnn*</ins> al ejecutar la siguiente instrucción:

```console
usr@dev:~$ conda env create -f experimentacion_srcnn.yml
```
Una vez creado el entorno debemos activarlo y es necesario que se mantenga así mientras trabajemos con los Scripts.

### 2. Degradar las imágenes

Reescalaremos las imágenes del conjunto de prueba para después intentar reconstruirlas con la red neuronal usando el Script <ins>*degrade_images.py*</ins> de la siguiente manera:

```console
usr@dev:~$ degrade_images.py -i <ruta_imgs_originales> -o <ruta_guardar_resultados> -f <factor_reescalado> 
```
A continuación una comparativa entre una imagen degradada y la original

| ![original](/imgs/original.bmp) | ![factor2](/imgs/f2.bmp) | ![factor4](/imgs/f4.bmp) |
| :--: | :--: | :--: |
| *Original* | *Factor 2* | *Factor 4* |

### 3. 






