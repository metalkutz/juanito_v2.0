# juanito_v2.0
Un excelente asistente (renovado) que tiene la capacidad de leer, entender y clasificar licitaciones en base a la descripción (texto libre) la cual describe el objeto licitado.


## Problema
Las licitaciones públicas son el método utilizado por el Estado de Chile para la compra de productos y obtención de servicios. Es decir, cuando algún órgano del estado tiene una necesidad de comprar o contratar algún tipo de servicio, hace un llamado público en la cual detalla su necesidad entregando características específicas para que proveedores puedan aplicar y ofrecer sus productos y servicios. El Gobierno de Chile ha habilitado una plataforma de licitaciones públicas llamada Mercado Público, la cual se puede acceder mediante el navegador escribiendo la dirección www.mercadopublico.cl. La plataforma tiene filtros de búsqueda que permiten a empresas o personas encontrar oportunidades en las cuales pueda ofrecer sus productos o servicios. Lamentablemente tales filtros no son suficientes para acotar la búsqueda de oportunidades, lo que termina dificultando la identificación de productos y resultando en un proceso de búsqueda costoso (tiempo, recursos). El buscador de la plataforma termina siendo limitado en su facilidad de búsqueda al no entregar como criterio el producto o servicio que se licita. Lo cual convierte el trabajo de una persona en una actividad titánica al tener que leer diariamente la descripción de licitaciones nuevas tratando de identificar si alguno de sus productos o servicios es requerido por el Estado. Y si fuera poco, a este problema de búsqueda se debe sumar el alto volumen de licitaciones que se publican diariamente, inconsistencia que existen en la data detallada en la licitación (diferencias entre el título y la descripción), el grado de conocimiento sobre el catálogo de productos y servicios que debe tener la persona, concentración y comprensión lectora del individuo que busca.


## Enfoque Solución
Crear un modelo de machine learning que identifique, dentro de un listado de licitaciones públicas (de Chile), aquellas donde una empresa o persona pueda participar con su catálogo de productos y/o servicios.
-	Identificar los procedimientos actuales de participación en las licitaciones públicas de Chile.
-	Establecer variables claves que faciliten el proceso de búsqueda de licitaciones. 
-	Construir un algoritmo que permita clasificar si una licitación aplica o no para participar con los productos o servicios del cliente.
-	Lograr que el modelo tenga un grado de confianza tal que nos permita llevarlo a producción.

## Herramientas y Tecnologías empleadas
- Metodología CRISP-DM
- Pandas, Numpy, Matplotlib
- NLP (Nltk, TF-IF, WordCloud)
- Scikit-Learn 

## Fuente de datos
https://www.mercadopublico.cl/Home

https://contrataciondelestado.es/wps/portal/plataforma

