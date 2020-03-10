# Proyecto Final: Bootcamp BigData&MachineLearning

Nombre: María Araceli Paredes Delgado

Fecha: marzo - 2020

Bootcamp: BIG DATA & MACHINE LEARNING - KEEPCODING

## Índice 
0. [Estructura del proyecto](#id1)
1. [Objetivo](#id2)
2. [Conjunto de datos](#id3)
3. [Sentiment Analysis - Reviews](#id4)
4. [Topic Modeling - Description](#id5)
5. [Cálculo de distancias a puntos de interés](#id6)
   - [Distancia a la estación de metro más cercana](#id11)
   - [Distancia al parking público más cercano](#id12)
   - [Distancia a los 5 museos más importantes de Madrid](#id13)
   - [Distancia a los 5 lugares/monumentos más importantes de Madrid](#id14)
6. [Limpieza, Análisis Exploratorio y Procesamiento de Datos](#id7)
7. [Modelado con algoritmos de Machine Learning](#id8)
8. [Modelado con algoritmos de Deep Learning](#id9)
9. [Conclusión del proyecto](#id10)


## 0.- Estructura del proyecto<a name="id1"></a>

Este proyecto tiene la siguiente estructura

- code: Contiene los distintos módulos en python correspondientes al desarrollo del proyecto. La numeración que tienen los módulos python coincide con el orden de desarrollo del proyecto.
- data: Contiene los ficheros csv con los diferentes datos utilizados como datasets del proyecto. Aquí se van almacenando todos los ficheros intermedios que se van generando y que se usan en los distintos módulos python.
- model: Contiene los dos modelos resultantes tras la evaluación de modelos de Machine-Learning y DeepLearning

Este proyecto se ha desarrollado utilizando **Google Colab** y las carpetas para almacenamiento están en **Google Drive**.
Una forma de montar la estructura de este proyecto será subir todo a Google Drive. Al principio de cada modulo python se define la ruta Google Drive donde está el proyecto. Esta variable se tendrá que cambiar en cada módulo para poner la ruta Google Drive donde se quiera alojar.

## 1.- Objetivo<a name="id2"></a>

Airbnb es un mercado comunitario para alquileres a corto plazo que sirve para publicar, dar publicidad y reservar alojamiento de forma económica en más de 190 países a través de internet. Es uno de los sistemas mas éxitos de la economía colaborativa – sistema económico en el que se comparten e intercambian bienes y servicios entre particulares a través de plataformas digitales -.
Éste sistema permite al usuario encontrar alojamiento, con la diferencia de que no será en un hotel sino en el hogar de una persona que puede incluso estar viviendo en él. 

Uno de los principales problemas a los que se enfrentan los anfitriones de Airbnb es determinar el precio óptimo de alquiler por noche. Este precio está vinculado a la dinámica del propio mercado. Si el anfitrión cobra por encima del precio de mercado, los inquilinos seleccionarán otras alternativas más asequibles. Si el precio del alquiler nocturno es demasiado bajo, los anfitriones estarán perdiendo ingresos potenciales.

El objetivo de este proyecto será conseguir el mejor modelo de machine learning o deep learning para predecir los precios óptimos que los anfitriones pueden establecer para sus propiedades. Esto se hace comparando la propiedad con otras del listado en base a parámetros como ubicación, tamaño de la propiedad, distancia a puntos de interés y otros datos demográficos. 

El fin último sería tener una herramienta a disposición de los anfitriones que les permitiera introducir los datos de la vivienda y les proporcionara el precio óptimo de alquilé. Además, en base al análisis que se haga de los datos, podríamos también ofrecer la posibilidad de que el anfitrión simulara añadir servicios o elementos a la vivienda y poder ir viendo como podría repercutir esos elementos en el precio.

El modelado para este proyecto se ha realizado para la ciudad de Madrid

## 2.- Conjunto de datos<a name="id3"></a>

El conjunto principal de datos lo obtengo de Insideairbnb, en concreto en [esta](http://insideairbnb.com/get-the-data.html) dirección podemos encontrar dataset de diferentes ciudades. De aquí nos descargamos para la ciudad de Madrid:
- "listings.csv.gz"
- "reviews.csv.gz"

Actualmente ya hay versiones posteriores pero en el momento del proyecto las fechas de los ficheros eran del 19/11/2019

### listing.csv

Este fichero contiene un listado de todas las viviendas con sus correspondientes características. Durante el primer paso del procesamiento iré eliminando algunas de las características. Muestro a continuación las que iré tratando tras esa limpieza inicial:

Característica | Descripción
-------------- | -------------
id | Identificación de la vivienda
experiences_offered | Actividades ofrecidas
host_since | Fecha en que el anfitrión se unió por primera vez a Airbnb
host_response_time | Cantidad promedio de tiempo que tarda el anfitrión en responder mensajes
host_response_rate | Proporción de mensajes a los que el anfitrión responde
host_is_superhost | Si el host es un superhost. Es una marca de calidad
host_listings_count | Cuántas viviendas tiene el host en total
host_identity_verified | Si el host ha sido verificado con id
neighbourhood_cleansed | El barrio de Madrid en el que está la propiedad
property_type | Tipo de propiedad, por ejemplo: casa, apartamento
room_type | Tipo de vivienda, por ejemplo: casa completa, habitación privada, compartida
accommodates | Cuantas personas pueden alojarse en la propiedad
bathrooms | Número de cuartos de baño
bedrooms | Número de habitaciones
beds | Número de camas
bed_type | Tipo de cama, por ejemplo: cama real, sofá cama
amenities | Lista de servicios/comodidades
price | Precio por noche (variable objetivo)
security_deposit  | Cantidad requerida como deposito
cleaning_fee | Cantidad requerida para limpieza
guests_included | Número de invitados incluidos en el precio de la reserva
extra_people | Precio añadido por cada huesped adicional a los indicados en guests_included
minimum_nights | Mínumo número de noches para la estancia
maximum_nights | Máximo número de noches para la estancia
calendar_updated | Cuando el anfitrión actualizó por última vez el calendario
availability_30 | Número de noches disponibles para la reserva en los próximos 30 dias
availability_60 | Número de noches disponibles para la reserva en los próximos 60 dias
availability_90 | Número de noches disponibles para la reserva en los próximos 90 dias
availability_365 | Número de noches disponibles para la reserva en los próximos 365 dias
number_of_reviews | Cantidad de comentarios que tiene la propiedad
number_of_reviews_ltm | Cantidad de comentarios que tiene la propiedad en los últimos 12 meses
first_review | Fecha del primer comentario
last_review | Fecha del último comentario
review_scores_rating  | Puntación general de la vivienda
review_scores_accuracy | Puntuación de la precisión de la descripción
review_scores_cleanliness | Puntación para la limpieza
review_scores_checkin | Puntación del proceso de checkin
review_scores_communication | Puntación para la comunicación con el anfitrión
review_scores_location | Puntuación para la localización
review_scores_value | Puntuación de la relación calidad/precio
instant_bookable | Si la propiedad puede ser reservada al instante sin enviar mensaje al anfitrión
cancellation_policy | Tipo de política de cancelación
reviews_per_month | Promedio de comentarios al mes que recibe la vivienda

### reviews.csv

Este fichero contiene un listado con todos los comentarios asociados a las viviendas

Característica | Descripción
-------------- | -------------
listing_id | Identificador de la vivienda
id | Identificador del comentario
date | Fecha del comentario
reviewer_id | Identificador del usuario que deja el comentario
reviewer_name | Nombre del usuario que deja el comentario
comments | Comentarios

Ambos ficheros se encuentra en la carpeta data de este proyecto


## 3.- Sentiment Analysis - Reviews<a name="id4"></a>

**code: 1 Sentimental analysis Reviews.ipynb**

Utilizando técnicas de NLP voy a realizar un Sentiment Analysis de los comentarios contenidos en el fichero "reviews.csv". El objetivo es conseguir un valor para cada uno de esos comentarios aplicando análisis de sentimientos. Posteriormente se hará la media de valor obtenido para todos los comentarios para una misma vivienda y ese resultado lo añadiré como una característica más al dataset "listing.csv" de Airbnb.

El código correspondiente se encuentra dentro de la carpeta code **"1 Sentimental analysis Reviews.ipynb"**
Los pasos que he realizado son los siguientes:

- Transformo el fichero reviews.csv en un dataframe y recorro la columna "comments" aplicando **TexBlob** Sentiment Analysis sobre esa columna. TexBlob es una librearía que hace NLP y está entrenada para comentarios en inglés en redes sociales

https://github.com/sloria/textblob

https://textblob.readthedocs.io/en/latest/quickstart.html#quickstart

El resultado que da TextBlob es: polarity y subjectivity. A mi me va a intersar polarity que puede ir del -1 al 1, siendo 1 el valor más positivo y -1 en valor más negativo

- Como los comentarios están en varios idiomas, utilizo la libreria **spacy** para detectar el idioma y quedarme solo con los que están en inglés, ya que es sobre este idioma sobre el que está entrenado TexBlob

- Lo siguiente que hago es agrupar por listing_id (hay varias reviews por id) y hacer la media de la columna sentiment obtenida en el paso anterior

- Me creo un nuevo dataset con id y la media del sentiment

- Hago un join con el dataset "listing" de tipo inner para buscar la intersección y con la clave id

Con esto ya consigo un nuevo dataset al que le añado la columna sentiment correspondiente a los reviews.

El resultado es el fichero **"listing_sentiment.csv"** que está en data. Durante este análisis se han ido creando ficheros csv intermedios que también están en data

## 4.- Topic Modeling - Description<a name="id5"></a>

**code: 2 Topic Modeling Descripciones.ipynb**

Utilizando técnicas de NLP voy a realizar un Topic Modeling sobre la columna "description". El objetivo es encontrar los topics principales sobre los que se habla en esas descripciones de las viviendas. Una vez encontrados se crearán nuevas características correspondientes a estos topics en el dataset "listings". Para cada vivienda se calculará el porcentaje de aparición de cada topic en la descripción y ese valor será añadido como característica en el topic correspondiente.

El código se encuentra dentro de la carpeta code **"2 Topic Modeling Descripciones.ipynb"**
Los pasos que he realizado son los siguientes:

- Como tengo descripciones tanto en inglés como en español realizo topic modeling para cada idioma y luego uno los resultados de ambos idiomas
- Recorro toda la columna "description" y haciendo uso de la librería **spacy** detecto el idioma añadiendo una nueva columna con ese idioma
- Me quedo con todos los que están en inglés
- Hago un preprocesamiento de la característica "description" convirtiendo a minúsculas, quitando stop word...etc
- Utilizo el algoritmo LDA para hacer Topic Modeling sobre la columna "description"
- Determino cual es el número óptimo de Topic pero me quedo con 5 por simplicidad
- Los 5 Topics que resultan son:

        - 1.Topic General: Este topic es el más confuso. Es un topic genérico que no cuadra dentro de las otras 4 categorías
        - 2.Topic Descripcion: Este topic está asociado a la descripción de la propia vivienda	        
        - 3.Topic Atracciones: Este topic está asociado a los monumentos y lugares de interés cercanos a la vivienda
        - 4.Topic Servicios: Este topic está asociado a los servicios de los que dispone la vivienda	
        - 5.Topic Transporte: Este topic está asociado al trasporte cercano a al necesario para llegar a la vivienda

- Añado al dataset 5 columnas con cada uno de estos 5 Topics y su correspondiente distribución para cada vivienda
- Hago lo mismo para las descriciones que están español usando un dataset independiente
- Cuando tengo mis dos datasets en inglés y en español con las nuevas columnas topics los uno 

En este momento le hemos añadido al dataet 5 columnas adicionales de topics de la descripción

El resultado es el fichero **"listings_sentiment_topic.csv"** que está en data. Durante este análisis se han ido creando ficheros csv intermedios que también están en data

## 5.- Cálculo de distancias a puntos de interés<a name="id6"></a>

**code: 3 Distancias.ipynb**

Una vez que he visto que en las descripciones de las viviendas son importantes los temas de transportes y atracciones cercanas voy a añadir al dataset las distancias de las viviendas a ciertos puntos de interés.
El código se encuentra dentro de la carpeta code **"3 Distancias.ipynb"**

### Distancia a la estación de metro más cercana<a name="id11"></a>

Voy a añadir al dataset una columna con la distancia a la estación de metro más cercana de cada vivienda. Obtengo la información desde [esta](https://datos.madrid.es/sites/v/index.jsp?vgnextoid=08055cde99be2410VgnVCM1000000b205a0aRCRD&vgnextchannel=374512b9ace9f310VgnVCM100000171f5a0aRCRD) dirección de Datos Abiertos de Madrid. 
En el propio código **"3 Distancias.ipynb"** están las instrucciones para bajarnos ese fichero que está en formato .kml y que habrá que transformar a csv.

El fichero "Metro.csv" que obtenemos contiene información de coordenadas geográficas de las diferentes estaciones de metro de Madrid.
Los pasos seguidos son:

- Usando los datos de longitud y latitud de la vivienda y de las estaciones de metro se cálcula la distancia de cada vivienda a cada una de las estaciones. Lo almaceno en un nuevo dataset
- Se ordena por el campo calculado de distancia agrupando por el id de la vivienda.
- Al ordenar de menor a mayor nos quedamos con el primero que nos aparece de cada vivienda eliminando el resto, así me quedo con la distancia más corta
- Uno el dataset obtenido de distancias con el dataset de viviendas usando como clave el id

Tras esto he añadido al datset una columna que contiene la distancia a la estación de metro más cercana.
El resultado es el fichero **"listings_sentiment_topic_discstation.csv"** que está en data. 


### Distancia al parking público más cercano<a name="id12"></a>

Voy a añadir al dataset una columna con la distancia al parking púbico más cercano de cada vivienda. Obtengo la información de los parkings consultando la API de EMT de Madrid 

La documentación de esta API está en esta dirección: https://apidocs.emtmadrid.es/

El primer paso antes de usar la API es registrarse [aquí](https://mobilitylabs.emtmadrid.es/) ya que para usar la API se necesita un token que se genera con el e-mail y la contraseña que hayamos puesto en el registro.

En el propio código **"3 Distancias.ipynb"**  se realiza la comunicación con la API para obtener el fichero
El fichero "parking.csv" que obtenemos contiene información de coordenadas geográficas de los diferentes parking públicos  de Madrid.

Los pasos seguidos son:

- Usando los datos de longitud y latitud de la vivienda y de los parkings se cálcula la distancia de cada vivienda a cada parking. Lo almaceno en un nuevo dataset
- Se ordena por el campo calculado de distancia agrupando por el id de la vivienda.
- Al ordenar de menor a mayor nos quedamos con el primero que nos aparece de cada vivienda eliminando el resto, así me quedo con la distancia más corta
- Uno el dataset obtenido de distancias con el dataset de viviendas usando como clave el id

Tras esto he añadido al datset una columna que contiene la distancia al parking más cercano.
El resultado es el fichero **"listings_sentiment_topic_discstation_discparking.csv"** que está en data. 


### Distancia a los 5 museos más importantes de Madrid<a name="id13"></a>

Voy a añadir al dataset cinco columnas con la distancia a los 5 museos más importantes de la ciudad de Madrid. Podría hacer manualmente un fichero que contuviese las coordenadas geográficas de estos 5 museos pero voy a utilizar la API de Datos Abiertos de Madrid para obtener el listado de todos los museos y de ahí escoger los que me interese por si en algún momento quiero cambiar y poner otros museos.

La documentación de la API está [aquí](https://datos.madrid.es/portal/site/egob/menuitem.214413fe61bdd68a53318ba0a8a409a0/?vgnextoid=b07e0f7c5ff9e510VgnVCM1000008a4a900aRCRD&vgnextchannel=b07e0f7c5ff9e510VgnVCM1000008a4a900aRCRD&vgnextfmt=default)

En el propio código **"3 Distancias.ipynb"**  se realiza la comunicación con la API para obtener el fichero.
El fichero "museos.csv" que obtenemos contiene información de coordenadas geográficas de los diferentes museos de Madrid.

A partir de ese fichero me creo un dataset con los 5 museos principales:

- Museo del Prado,
- Museo Nacional Centro de Arte Reina Sofía,
- Museo Nacional Thyssen-Bornemisza,
- Museo Sorolla,
- Museo Arqueológico Nacional

Los pasos seguidos son:

- Usando los datos de longitud y latitud de la vivienda y de los 5 museos cálculo la distancia de una vivienda a cada uno de ellos.
- Añado 5 columnas al dataset original con la distancia de cada vivienda a cada uno de los 5 museos

Tras esto he añadido al datset 5 columnas que contiene la distancia a los 5 museos más importantes
El resultado es el fichero **"listings_sentiment_topic_discstation_discparking_discmuseos.csv"** que está en data.

### Distancia a los 5 lugares/monumentos más importantes de Madrid<a name="id14"></a>

Voy a añadir al dataset cinco columnas con la distancia a los 5 lugares/monumentos más importantes de la ciudad de Madrid. En este caso el fichero con las coordenadas geográficas de esos 5 lugares/monumentos de la ciudad de Madrid lo he creado de forma manual.
El fichero "Atracciones.csv" que está en data contiene las coordenadas geográficade de estos 5 lugares que he escogido que sean:

- Puerta del Sol
- Plaza Mayor
- El Retiro
- Puerta de Alcalá
- Palacio Real

Los pasos seguidos son:

- Usando los datos de longitud y latitud de la vivienda y de los 5 atracciones cálculo la distancia de una vivienda a cada una de ellos.
- Añado 5 columnas al dataset original con la distancia de cada vivienda a cada uno de las 5 atracciones

Tras esto he añadido al datset 5 columnas que contiene la distancia a los 5 atracciones más importantes
El resultado es el fichero **"listings_sentiment_topic_discstation_discparking_discmuseos_discatracciones.csv"** que está en data.

## 6.- Limpieza, Análisis Exploratorio y Procesamiento de Datos<a name="id7"></a>

**code: 4 EDA.ipynb**

El preprocesamiento y la limpieza de datos son tareas importantes que se deben llevar a cabo para que un conjunto de datos se pueda usar para el entrenamiento de modelo. Con el análisis exploratorio de datos realizamos un tratamiento estadístico de las muestras que nos permite ir analizando las posibles relaciones o visualizando conexiones entre ciertas variables. Se utilizan herramientas estadísticas junto con algunas visualizaviones para entender un poco más los datos de los que disponemos.

El código **"4 EDA.ipynb"** se encarga de esta tarea. Pasos realizados:

- Comprobar los tipos de datos de las características
- División en train (70%) y test (30%) antes de comenzar cualquier tipo de limpieza y procesamiento
- Eliminación de caracterísitcas con URLs y textos 
- Eliminación de caracteríticas con la mayoría de valores nulos
- Eliminación de características que contienen información relacionada (se ve a simple vista sin necesidad de análisis de correlación)
- Eliminación de algunos campos calculados
- Transformación de las características con valores boolenaos "f", "t" a valores numéricos "0", "1"
- Análisis de las varibles de localización. Nos quedamos con una de ellas para el análisis y se elimina el resto
- Limpieza de columnas individuales. Se quitan caracteres como "$" 
- Eliminamos valores nulos bien eliminando esas observaciones o dándoles algún valor que puede ser la mediana, el valor más frecuente o un valor 0 dependiendo de la característica
- Eliminamos características que solo tengan un valor (o un porcentaje muy elevado de un mismo valor)  y que por tanto no nos aportan información al modelo
- Generación de nuevas características en base a las existentes
- Transformación de algunas características numéricas en categóricas agrupándolas por valores
- Análisis exploratorio de variables de tipo numérico
- Tratamiento de outliers
- Análisis exploratorio de variables de tipo categórico
- Análisis exploratiro de variables de tipo booleano
- Codificación de variables categóricas
- Análisis de correlación. Eliminación de variables correladas
- Transformación logarítmica de variables numéricas

Como último punto se deben aplicar todas las transformaciones anteriones a las muestras de test

## 7.- Modelado con algoritmos de Machine Learning<a name="id8"></a>

**code: 5 Modelado con algoritmos de Machine Learning - ML.ipynb**

Una vez que ya tenemos los datos limpios y procesados vamos a aplicar diferentes modelos "Supervised Machine Learning" y los compararemos entre si. La métrica que usaré al ser un problema de regresión es el RMSE (Error cuadrático medio).

Antes de comenzar con el modelado normalizaré las variables de entrada. Normalizar significa, en este caso, comprimir o extender los valores de la variable para que estén en un rango definido. Utilizo el método Standard Scaler.

A continuación muestro una tabla con los algoritmos utilizados y sus resultados:

Modelo | RMSE (test) | R2 (test)
-------|-------------|------------- 
Regresión Lineal | 5.7105e+20 | -1.6159e+21
Ridge Regression | 0.3192 | 0.7115
Lasso | 0.323 | 0.7056
Árbol de Regresión | 0.358 | 0.6365
Boosted Tree | 0.281 | 0.7773
SVR | 0.301 | 0.7429

Según los resultados anteriores el modelo que escojo es el **Gradiente Boosted Tree**. Los parámetros concretos de este modelo son:

Niteraciones | learning_rate | profundidad | RMSE Modelo (test) | R2 test
-------------|---------------|-------------|--------------------|---------- 
2000 | 0.05 | 6 | 0.281 | 0.7773

A continuación he realizado una serie de pruebas para intentar mejorar aún más este modelo:

- Eliminación de características reviews particulares quedándome solo con las reviews genéricas. Vi en su momento que mostraban correlación.

Niteraciones | learning_rate | profundidad | RMSE Modelo (test) | R2 test
-------------|---------------|-------------|--------------------|---------- 
1000 | 0.05 | 6 | 0.278 | 0.7819

Se consigue una mejora de resultados y encima hemos quitado complejidad al modelo

  
- Modificación de Niteracions y learning Rate. Mantengo profundidad 6. 

Niteraciones | learning_rate | profundidad | RMSE Modelo (test) | R2 test
-------------|---------------|-------------|--------------------|---------- 
1200 | 0.05 | 6 | 0.278 | 0.7819

Resultado: Ha aumentado el número de iteracciones óptimas pero el resultado sigue siendo el mismo.    
   
- Modificación de profundidad a 4   

Niteraciones | learning_rate | profundidad | RMSE Modelo (test) | R2 test
-------------|---------------|-------------|--------------------|---------- 
1500 | 0.05 | 4 | 0.281 | 0.7766

Resultado: Ha aumentado un poco el error al disminuir la profundidad pero no hay demasiada diferencia y le he quitado complejidad al modelo por lo que me quedo de momento con esta solución

- Modificación de Niteracions y learning Rate. Mantengo profundidad 4. 

Niteraciones | learning_rate | profundidad | RMSE Modelo (test) | R2 test
-------------|---------------|-------------|--------------------|---------- 
1500 | 0.05 | 4 | 0.281 | 0.7766

Los parámetros óptimos vuelven a ser los mismos que en la anterior prueba

### Conclusión del modelado con algoritmos de Machine Learning

El modelo que mejor resultados me da es el **"Gradient Boosted Tree"** con la eliminación de las características reviews particulares. Los parámetros son los siguientes:
   
  Niteraciones | learning_rate | profundidad 
-------------|---------------|--------------
1500 | 0.05 | 4 

Con esto consigo para mi conjunto de test y, tras aplicar todas las transformaciones necesarias, estos resultados:

RMSE Modelo (test) | R2 test 
-------------|---------------
0.7766 | 0.281  

Las tres característica más importantes para el cálculo del precio en orden son:

- Room Type: Entire home/apt
- Accomodates
- Bathrooms

El modelo está almacenado en la carpeta model del proyecto con el nombre **GradientBoostedTree.sav**

## 8.- Modelado con algoritmos de Deep Learning<a name="id9"></a>

**code: 6 Modelado con algoritmos de Deep Learning.ipynb**

Vamos ahora a aplicar diferentes modelos de Deep Learning y los compararemos entre si. La métrica que usaré al ser un problema de regresión es el RMSE (Error cuadrático medio).

Antes de comenzar con el modelado normalizaré las variables de entrada utilizando el método Standard Scaler.

Muestro a continuación los diferentes modelos de Redes Neuronales que he utilizado

 Modelo | RMSE (train) | RMSE (test) | R2 (train) | R2 (test)
--------|--------------|-------------|------------|-----------
1 Red Neuronal con 3 capas | 0.0141 | 0.1448 | 0.9602 | 0.5903
2 Red neuronal con 4 capas, regularización L1 y más épocas | 0.1034 | 0.1006 | 0.708 | 0.7152
3 Red neuronal con 4 capas, regularización Droput 0.5 y más épocas | 0.1151 | 0.1305 | 0.6748 | 0.6307
4 Red neuronal con 4 capas, regularización Dropout 0.2 y mayor batch size | 0.0404 | 0.1245 | 0.8859 | 0.6477

Examinando los resultados se ve que el mejor modelo es el **modelo 2: Red neuronal con 4 capas, regularización L1**

Estas son sus características:

N. Capas | F. activación | F. salida | F. pérdidas | Regularización | Optimizador | N. epocas | Batch_size
---------|---------------|-----------|-------------|----------------|-------------|-----------|------------
4 | Relu | linear | mean squered_error | L1 | Adam | 150 | 256

A continuación he realizado una serie de pruebas para intentar mejorar aún más este modelo:

- Eliminación de características reviews particulares quedándome solo con las reviews genéricas. Vi en su momento que mostraban correlación.

RMSE (train) | RMSE (test) | R2 (train) | R2 (test)
-------------|-------------|------------|-----------
0.1071 | 0.1039 | 0.6974 | 0.7061

Conseguimos resultados similares al del modelo 2 pero he eliminado 30 características con lo que la complejidad es menor por lo que nos quedamos de momento con esta opción

- Red neuronal con 5 capas, regularización L1 y eliminación de características correladas

RMSE (train) | RMSE (test) | R2 (train) | R2 (test)
-------------|-------------|------------|-----------
0.1064 | 0.1031 | 0.6995 | 0.7083

Se consigue muy poca mejora para el aumento de complejidad. Con lo cual me quedo con el modelo anterior

### Conclusión del modelado con algoritmos de Deep Larning

El modelo que mejor resultados me da es una **Red Neuronal con 4 capas, regularización L1, optimización de Adam y eliminación de características review correladas** Los parámetros son los siguientes:
   
 N. Capas | F. activación | F. salida | F. pérdidas | Regularización | Optimizador | N. epocas | Batch_size
---------|---------------|-----------|-------------|----------------|-------------|-----------|------------
4 | Relu | linear | mean squered_error | L1 | Adam | 150 | 256  
  
Con esto consigo para mi conjunto de test y, tras aplicar todas las transformaciones necesarias, estos resultados:

RMSE Modelo (test) | R2 test 
-------------|---------------
0.1031 | 0.7083  

El modelo está almacenado en la carpeta model del proyecto con el nombre **RedNeuronal.sav**

## 9.- Conclusión del proyecto.<a name="id10"></a> 

Esta es una de esas situaciones en las que deep learning simplemente no es necesario para la predicción, y un modelo de machine learning funciona mejor. Sin embargo, incluso en el modelo con mejor rendimiento, éste solo pudo explicar el 77% de la variación en el precio. El 23% restante probablemente esté compuesto por características que no estaban presentes en los datos. Es probable que una proporción significativa de esta varianza inexplicada se deba a las fotos de las vivendas. Las fotos de propiedades en Airbnb son muy importantes para alentar a los huéspedes a reservar, por lo que se puede esperar que tengan un impacto significativo en el precio: mejores fotos (principalmente propiedades y muebles de mejor calidad, pero también fotografías de mejor calidad) equivalen a precios más altos.

El problema que tenemos con el dataset de Airbnb es que para las fotos únicamente nos proporciona una URL que nos lleva a una única foto de la vivienda. Basta con echar un vistazo a estas fotos para comprobar que son de diversas estancias, de exteriores, hasta de mascotas. 
Esto se pudo ver en una de las prácticas del Bootcamp en las que se utilizó redes convolucionales para intentar predecir el precio en base a las fotos y los resultados no fueron nada buenos, de ahí que haya decidido no incluirlo en este proyecto final.

Para poder hacer un entranamiento correcto se debería tener más imagenes de la vivienda y además se debería clasificar por distintos tipos de estancia. Una vez separadas se podría montar una imagen compuesta a base de varias estancias para cada anuncio o bien entrenar solo con las fotos de determinada tipo de estancia: salón, cocina.....

Esto queda para una mejora futura de este proyecto






















