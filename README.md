# Proyecto Final: Bootcamp BigData&MachineLearning

## 0.- Estructura del proyecto

Descripción de las carpeta y comentario sobre google colab

## 1.- Objetivo

Airbnb es un mercado comunitario para alquileres a corto plazo que sirve para publicar, dar publicidad y reservar alojamiento de forma económica en más de 190 países a través de internet. Es uno de los sistemas mas éxitos de la economía colaborativa – sistema económico en el que se comparten e intercambian bienes y servicios entre particulares a través de plataformas digitales -.
Éste sistema permite al usuario encontrar alojamiento, con la diferencia de que no será en un hotel sino en el hogar de una persona que puede incluso estar viviendo en él. 

Uno de los principales problemas a los que se enfrentan los anfitriones de Airbnb es determinar el precio óptimo de alquiler por noche. Este precio está vinculado a la dinámica del propio mercado. Si el anfitrión cobra por encima del precio de mercado, los inquilinos seleccionarán otras alternativas más asequibles. Si el precio del alquiler nocturno es demasiado bajo, los anfitriones estarán perdiendo ingresos potenciales.

El objetivo de este proyecto será conseguir el mejor modelo de machine learning o deep learning para predecir los precios óptimos que los anfitriones pueden establecer para sus propiedades. Esto se hace comparando la propiedad con otras del listado en base a parámetros como ubicación, tamaño de la propiedad, distancia a puntos de interés y otros datos demográficos. 

El fin último sería tener una herramienta a disposición de los anfitriones que les permitira introducir los datos de la vivienda y les proporcionara el precio óptimo de alquilé. Además, en base al análisis que se haga de los datos, podríamos también ofrecer la posiblidad de que el anfitrión simulara añadir servicios o elementos a la vivienda y poder ir viendo como podría repercutir esos elementos en el precio.

El modelado para este proyecto se ha realizado para la ciudad de Madrid

## 2.- Conjunto de datos

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


## 3.- Sentiment Analysis - Reviews

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

## 4.- Topic Modeling - Description

**code: 2 Topic Modeling Descripciones.ipynb**

Utilizando técnicas de NLP voy a realizar un Topic Modeling sobre la columna "description". El objetivo es encontrar los topics principiales sobre los que se habla en esas descripciones de las viviendas. Una vez encontrados se crearán nuevas características correspondientes a estos topics en el dataset "listings". Para cada vivienda se calculará el porcentaje de aparación de cada topic en la descripción y ese valor será añadido como característica en el topic correspondiente.

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

## 5.- Cálculo de distancias a puntos de interés

**code: 3 Distancias.ipynb**

Una vez que he visto que en las descripciones de las viviendas son importantes los temas de transportes y atracciones cercanas voy a añadir al dataset las distancias de las viviendas a ciertos puntos de interés.
El código se encuentra dentro de la carpeta code **"3 Distancias.ipynb"**

### Distancia a la estación de metro más cercana

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


### Distancia al parking público más cercano

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
El resultado es el fichero **"listings_sentiment_topic_discstation_discparking.csv.csv"** que está en data. 




Paso 3: Partiendo de la API de datos abiertos de Madrid ver que datos tenía ahí y añadir en el dataset unas columnas de proximidad a museo, proximidad a parking……. Proximidad a monumento…..

                        Añadido al dataset columnas con:
Distancia al parking más cercano
Distancia a la estación de metro más cercana, tengo un csv
Distancias a los 5 museos principales: Museo del Prado, Reina Sofia, Thyssen,  Museo Sorolla, Museo Arqueológico Nacional
Distancias a los 5 monumentos principales:  Puerta del Sol, Plaza mayor, El Retiro, Puerta de Alcalá, Palacio Real
















