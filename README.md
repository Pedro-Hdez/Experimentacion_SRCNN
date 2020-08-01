# Experimentacion_SRCNN

En este repositorio se encuentran el código fuente de mi proyecto de **Redes Neuronales Convolucionales** donde realizo algunos experimentos utilizando la red neuronal [*SRCNN*](https://arxiv.org/pdf/1501.00092.pdf) para aumentar la resolución de imágenes degradadas. Este trabajo llamado *Superresolución de imágenes* se encuentra en [mi blog](https://pedro-hdez.github.io/), te invito a echarle un vistazo.

El código fuente original (Matlab / Caffe) lo puedes encontrar [aquí](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html), y una versión del mismo (en el que yo me basé) pero traducido en Python / Tensorflow, se encuentra en [este repositorio](https://github.com/MarkPrecursor/SRCNN-keras).

## PREREQUISITOS

* Este proyecto fue desarrollado completamente bajo Linux (Ubuntu 20.04); por lo tanto, explicaré cómo usar los scripts con línea de comandos desde la terminal.

* Para facilitar todo el manejo de las librerías necesarias se hará uso de un **Entorno de Anaconda**. Para instalarlo en tu máquina sigue los pasos de la [guía oficial](https://docs.anaconda.com/anaconda/install/).

* Trabajaremos con imágenes, por lo que es necesario contar con algunos ejemplos para entrenar y otros para predecir. Puedes usar tus propias fotos o visitar el Post donde desarrollo el proyecto y ahí encontrarás algunos links de DataSets que yo utilicé.

## CÓMO USARLO

A continuación, voy a presentarte una guía para mi código. Una vez instalado Anaconda, los pasos son los siguientes:

1. [Importar el entorno de Anaconda](#1-importar-el-entorno-de-anaconda)
2. [Degradar las imágenes](#2-degradar-las-imágenes)
3. [Preparar los datos de entrenamiento](#3-preparar-los-datos-de-entrenamiento)
4. [Entrenar la red neuronal](#4-entrenar-la-red-neuronal)
5. [Reconstruir imágenes](#5-reconstruir-imágenes)

Adicionalmente, algunos Scripts tienen la finalidad de analizar el desempeño de los modelos.

6. [Analizar los indicadores de desempeño](#6-analizar-los-indicadores-de-desempeño)
7. [Comparar dos modelos](#7-comparar-dos-modelos)
8. [Encontrar los mejores resultados](#8-encontrar-los-mejores-resultados)


### 1 Importar el entorno de Anaconda

Para usar los scripts es necesario tener instaladas algunas librerías como *Pandas*, *NumPy* y *OpenCV*; además de Tensorflow y Python 3.X. Todas estas especificaciones se encuentran en el archivo **experimentacion_srcnn.yml**. Anaconda automáticamente creará un entorno llamado *experimentacion_srcnn* al ejecutar la siguiente instrucción:

```console
usr@dev:~$ conda env create -f experimentacion_srcnn.yml
```
Una vez creado el entorno debemos activarlo y es necesario que se mantenga así mientras trabajemos con los Scripts.

### 2 Degradar las imágenes

Reescalaremos las imágenes del conjunto de prueba para después intentar reconstruirlas con la red neuronal usando el Script *degrade_images.py* de la siguiente manera:

```console
usr@dev:~$ degrade_images.py -i <ruta_imgs_originales> -o <ruta_guardar_resultados> -f <factor_reescalado> 
```
Aquí un ejemplo de imágenes degradadas

| ![original](/imgs/original.bmp) | ![f2](/imgs/f2.bmp) | ![f4](/imgs/f4.bmp) |
| :--: | :--: | :--: |
|Original|Degradada (factor 2)|Degradada (factor 4)|

### 3 Preparar los datos de entrenamiento

Usando el script *prepare_data.py* vamos a procesar cada uno de los ejemplos de entrenamiento y guardaremos su información en archivos .h5 desde los cuales vamos a alimentar a los modelos en la siguiente etapa. Para ésto, debemos ejecutar la siguiente instrucción:

```console
usr@dev:~$ python prepare_data.py -i <ruta_imgs_entrenamiento> -f <factor_reescalado> -c <(OPCIONAL) 1>
```
El argumento opcional **-c** es para convertir las imágenes de su formato original a archivos .bmp, ésto con la finalidad de tener información más precisa de cada pixel.

Los resultados serán dos archivos: **crop_train.h5** y **test.h5** que se guardarán en el directorio donde estemos trabajando.

### 4 Entrenar la red neuronal

Una vez obtenida la información del conjunto de entrenamiento, vamos a entrenar la red neuronal con la siguiente instrucción:

```console
usr@dev:~$ python train.py -i <guardar_pesos.h5> -m <tipo_modelo> -e <# epochs> -l <(OPCIONAL)pesos_preentrenados.h5>
```

Como necesitamos los archivos que se generaron en el [paso anterior](#3-preparar-los-datos-de-entrenamiento), debemos ejecutar este Script en ese mismo directorio.

En mis experimentos, utilicé tres diferentes modelos, entonces el argumento **-m** puede ser:

* 1 = Modelo original (SRCNN)
* 2 = Mi modelo (6 capas convolucionales, 249 filtros totales)
* 3 = Mi modelo 2 (6 capas convolucionales, 448 filtros totales)

Si usamos el argumento **-l** cargaremos, desde un archivo .h5, algún modelo preentrenado y continuará con su entrenamiento durante **-e** epochs más.

**Nota** Hay que tener cuidado en esta etapa porque puede demorar muchas horas. Esto depende de la arquitectura del modelo que estamos entrenando, el tamaño del conjunto de entrenamiento y de nuestro poder de cómputo.

### 5 Reconstruir imágenes

Después de entrenar algún modelo, ahora seremos capaces de reconstruir las imágenes que reescalamos en el [paso 2](#2-degradar-las-imágenes). Lo haremos utilizando el script *predict.py*

```console
usr@dev:~$ python predict.py -m <tipo_modelo> -w <pesos_entrenados.h5> -r <ruta_imgs_originales> -t <ruta_imgs_reescaladas> -o <ruta_guardar_resultados>
```

Este script generará varios resultados y los guardará en la ruta **<ruta_guardar_resultados>**:

* Carpeta **analysis**: Almacena una comparativa entre las imágenes originales, degradadas y reconstruidas.
* Carpeta **individual_images**: Contiene únicamente las imágenes reconstruidas.
* Archivo **degraded_img_scores.csv**: Métricas de similitud de las imágenes degradadas.
* Archivo **scores_csv**: Métricas de similitud de las imágenes reconstruidas.

|![analysis](/imgs/cuatro.png)|
|:--:|
|Ejemplo carpeta *analysis*|

### 6 Analizar los indicadores de desempeño

El script *analysis.py* ejecuta un análisis cuantitativo de las métricas de similitud (psnr, mse, ssim) obtenidas por las reconstrucciones de los modelos, para ésto, ejecutamos la instrucción:

```console
usr@dev:~$ python analysis.py -a <ruta_reconstruccion_modelo> -b <(OPCIONAL) <ruta_reconstruccion_modelo2>
```

Los resultados se van a imprimir en consola. Si utilizamos el argumento **-b**, se imprimirá el análisis para cada modelo y se decidirá cuál de ellos tuvo mejores resultados de acuerdo a cada métrica.

|![dos_modelos](/imgs/a2.png)
| :--: |
|Ejemplo|

### 7 Comparar dos modelos

Para comparar las reconstrucciones de dos modelos diferentes podemos usar el script *comparator.py* de la siguiente manera:

```console
usr@dev:~$ python comparator.py -r <ruta_imgs_originales> -d <ruta_imgs_reescaladas> -a <ruta_resultados_modelo1> -b <ruta_resultados_modelo2> -o <ruta_guardar_resultados>
```

Como resultado, se van a guardar las comparaciones en la carpeta **<ruta_guardar_resultado>**.

|![comparacion](/imgs/comparacion.png)|
|:--:|
|Ejemplo de una comparación|

### 8 Encontrar los mejores resultados

Si deseamos saber en qué ejemplos se lograron mejores resultados, de acuerdo a una métrica de similitud específica, deberemos usar el script *find_best.py*:

```console
usr@dev:~$ python find_best.py -i <ruta_resultados_modelo> -m <metrica>
```

Los resultados se mostrarán en consola

|![mejores](/imgs/mejor.png)|
|:--:|
|Ejemplo|




















