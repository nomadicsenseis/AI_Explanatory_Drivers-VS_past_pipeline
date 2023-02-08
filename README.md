# Machine Learning Project - Sagemaker



## Introduction

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Setup

### Framework

### Commands

### Terminal

### Gitlab

### Pipeline

### Filesystem

## Sagemaker Notebooks [Jupyter]

Amazon SageMaker Notebook es un servicio de Jupyter notebooks integrado en la nube, diseñado para simplificar el desarrollo y la ejecución de código en Amazon SageMaker. Permite a los desarrolladores y científicos de datos crear y ejecutar notebooks de Jupyter en un entorno seguro y administrado, con acceso a los recursos y la capacidad de procesamiento de AWS.

Los notebooks de SageMaker están basados en Amazon Elastic Container Service (ECS) y se ejecutan en instancias dedicadas. SageMaker administra la infraestructura subyacente, lo que significa que los usuarios pueden centrarse en el código y los datos, en lugar de preocuparse por la configuración de la infraestructura.

Los notebooks de SageMaker también ofrecen una integración fluida con otras características de SageMaker, como el almacenamiento de datos y los servicios de entrenamiento y despliegue. Los usuarios pueden acceder a los recursos de AWS, como S3, EC2 y ECR, desde dentro del notebook, lo que les permite utilizar herramientas y servicios de AWS para construir y entrenar modelos de aprendizaje automático y crear flujos de trabajo de análisis de datos completos.

En resumen, el entorno de notebooks de AWS SageMaker es una plataforma de desarrollo en la nube segura y administrada para crear, ejecutar y distribuir aplicaciones de aprendizaje automático y análisis de datos.

//TODO: FOTOS, COMENTARIOS DEL LAYOUT, ETC

### Framework

### Instances

#### Memory/CPU based

##### Installing libraries

Instalación de bibliotecas en una máquina SageMaker Notebook:

Abra un terminal en el notebook de Jupyter.
Utilice el comando !pip install para instalar las bibliotecas necesarias. Por ejemplo, !pip install pandas matplotlib.
Tenga en cuenta que algunas bibliotecas, como las bibliotecas de análisis de datos, pueden requerir más memoria y capacidad de procesamiento que la que se proporciona por defecto en una máquina SageMaker Notebook. En estos casos, es posible que sea necesario aumentar la memoria y la capacidad de procesamiento de la máquina antes de instalar las bibliotecas.

Se pueden encontrar los siguientes problemas al intentar instalar librerias en SageMaker:

Permisos insuficientes: Si el usuario no tiene permisos suficientes para instalar librerias en el entorno de SageMaker, puede recibir un mensaje de error que indica que no tiene permisos para instalar librerias.

Conflictos de dependencias: Las librerias a menudo dependen de otras librerias para funcionar correctamente. Si una libreria depende de una versión específica de otra libreria, pero la versión actualmente instalada es diferente, puede ocurrir un conflicto de dependencias que resulte en un error al instalar la libreria.

Problemas de conexión a Internet: Si el entorno de SageMaker no tiene acceso a Internet, es posible que no se puedan descargar las librerias necesarias durante la instalación.

Problemas de compatibilidad: Las librerias a menudo son desarrolladas para funcionar con versiones específicas de Python y otras tecnologías. Si una libreria no es compatible con la versión de Python o tecnología que se ejecuta en el entorno de SageMaker, puede ocurrir un error al intentar instalar la libreria.

Problemas de recursos: Algunas librerias pueden requerir una gran cantidad de recursos para funcionar correctamente. Si el entorno de SageMaker no tiene suficientes recursos disponibles, puede ocurrir un error al intentar instalar la libreria.

Estos son algunos de los problemas más comunes que un usuario puede encontrar al intentar instalar librerias en SageMaker. Para resolver estos problemas, es importante investigar el mensaje de error, revisar la documentación y consultar a la comunidad de usuarios.

##### Tips

#### Distributed based [EMR]

##### Installing libraries

La instalación de bibliotecas en un cluster Apache Spark y en una máquina SageMaker Notebook es un proceso diferente, dependiendo de la forma en que se esté utilizando Spark.

Instalación de bibliotecas en un cluster de Spark:

Cree un archivo de dependencias de Python, como un archivo requirements.txt, que contenga las bibliotecas que desea instalar.
Suba el archivo de dependencias a S3 o a otro almacenamiento de archivos accesible desde Spark.
Inicie un cluster Spark en Amazon EMR, que incluirá una opción para especificar la ubicación del archivo de dependencias.
Una vez que el cluster esté activo, puede instalar las bibliotecas en el cluster utilizando !pip install -r y proporcionando la ubicación del archivo de dependencias en S3.

Otra opcion de la que dispone el usuario es el uso del comando //TODO

##### SparkMagic

SparkMagic es un conjunto de comandos que permiten a los usuarios ejecutar código en un cluster Apache Spark desde dentro de un notebook de Jupyter. Estos comandos están diseñados para facilitar la integración de Spark con Jupyter y proporcionar una interfaz de línea de comandos para interactuar con Spark desde el notebook.

Con SparkMagic, los usuarios pueden ejecutar código en Spark desde el notebook sin tener que abandonar la interfaz de Jupyter. Además, SparkMagic también permite a los usuarios visualizar y explorar los resultados de los comandos de Spark en el mismo notebook.

SparkMagic también proporciona una integración fluida con otras herramientas y servicios de AWS, como S3, EC2 y EMR. Esto significa que los usuarios pueden acceder a datos almacenados en S3 y procesarlos con Spark, utilizar EC2 para ejecutar el cluster Spark y utilizar EMR para gestionar la infraestructura subyacente.

En resumen, los comandos SparkMagic proporcionan una forma sencilla y eficiente de interactuar con Apache Spark desde un notebook de Jupyter, y también permiten a los usuarios utilizar las características y recursos de AWS para procesar y analizar datos en grandes volúmenes.

Los comandos de SparkMagic incluyen los siguientes:

%spark configure: Este comando permite a los usuarios configurar la conexión a un cluster Spark. Los usuarios pueden especificar la dirección URL del cluster, el puerto, el usuario y la contraseña, entre otros detalles.

%spark: Este comando permite a los usuarios crear una sesión Spark en el cluster especificado mediante %spark configure. Después de crear la sesión, los usuarios pueden escribir y ejecutar código en Spark directamente desde el notebook.

%spark info: Este comando muestra información sobre la sesión Spark actual, incluyendo el ID de la sesión, el cluster al que está conectado y el estado de la sesión.

%spark log: Este comando muestra los registros de la sesión Spark actual. Estos registros pueden ser útiles para depurar errores y comprender mejor lo que sucede en el cluster.

%spark cleanup: Este comando cierra la sesión Spark actual y libera los recursos asociados con la sesión.

%spark add: Este comando permite a los usuarios agregar archivos a la sesión Spark actual. Los usuarios pueden agregar archivos locales o remotos a la sesión para que puedan ser accedidos y utilizados por Spark.

%spark load: Este comando permite a los usuarios cargar datos en Spark desde un archivo local o remoto.

%spark display: Este comando permite a los usuarios visualizar los resultados de los comandos Spark en el notebook. Los resultados se muestran en un formato amigable para el usuario, como una tabla o un gráfico.

Estos son los comandos principales de SparkMagic, y juntos proporcionan una forma eficiente y fácil de interactuar con Spark desde el notebook de Jupyter.

##### Tips

### Plotting

### boto3 [s3]

## Production

### Project structure

#### Exploratory data analysis (EDA)

#### Develop

#### Production

### Continuous integration with GitLab

### Sagemaker Pipeline (framework and tips)

### Foto grafo + explicación de cada step y su código

# Utils for mkdocs

```
introduce code
```