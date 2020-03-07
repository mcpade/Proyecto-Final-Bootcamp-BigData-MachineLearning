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




































Una vez descomprimidos ambos ficheros se ve que en "linsting.csv" tenemos el listado de todas las viviendas con sus correspondientes características: 

atos escogido es éste, extraído de Airbnb mediante técnicas de scraping. Dentro de las opciones recomiendo utilizar el extract (“Only the 14780 selected records”), ya que minimiza el tiempo de ejecución y evita problemas de memoria en equipos con menos prestaciones.




