# Proyecto Final: Bootcamp BigData&MachineLearning

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

Este fichero contiene un listado de todas las viviendas con sus correspondientes características. Durante el el primer paso del procesamiento iré eliminando algunas de las características. Muestras a continuación las que iré tratando tras esa limpieza inicial:

Característica | Descripción
-------------- | -------------
experiences_offered | Actividades ofrecidas
host_since | Fecha en que el anfitrión se unió por primera vez a Airbnb
host_response_time | Cantidad promedio de tiempo que tarda el anfitrión en responder mensajes
host_response_rate | Proporción de mensajes a los que el anfitrión responde
host_is_superhost | Si el host es un superhost. Es una marca de calidad
host_listings_count | Cuántas viviendas tiene el host en total
host_identity_verified | Si el host ha sido verificado con id
neighbourhood_cleansed | El barrio de Madrid en el que está la propiedad
property_type | Tipo de propiedad, por ejemplo casa, apartamento...
room_type | Tipo de vivienda, por ejemplo: casa completa, habitación privada, compartida
accommodates | Cuantas personas pueden alojarse en la propiedad
bathrooms | Número de cuartos de baño
bedrooms | Número de habitaciones
beds | Número de camas
bed_type | Tipo de cama, por ejemplo: cama real, sofá cama
amenities | Lista de servicios/comodidades
price | Precio por noche (variable objetivo)
security_deposit  | Cantidad requerida como deposito















Una vez descomprimidos ambos ficheros se ve que en "linsting.csv" tenemos el listado de todas las viviendas con sus correspondientes características: 

atos escogido es éste, extraído de Airbnb mediante técnicas de scraping. Dentro de las opciones recomiendo utilizar el extract (“Only the 14780 selected records”), ya que minimiza el tiempo de ejecución y evita problemas de memoria en equipos con menos prestaciones.




