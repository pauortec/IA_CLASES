Sales Bot — RAG + Mistral
Análisis de datos de ventas con un bot inteligente usando RAG y la API de Mistral.
¿Qué hace el notebook?

Instalación — instala las dependencias necesarias
Imports — carga las librerías y define las funciones para llamar a la API de Mistral sin SDK
Carga del CSV — sube y lee sales_data_sample.csv
Limpieza y normalización — detecta y rellena valores nulos, elimina duplicados, trata outliers con IQR y aplica MinMaxScaler, StandardScaler y RobustScaler
Construcción de chunks — divide el dataset en fragmentos temáticos: KPIs, ventas por producto, por año, por trimestre, top clientes, países, estado de órdenes, ciudades y tamaño de deals
Embeddings — convierte cada chunk en un vector numérico usando mistral-embed
Retrieval — busca los chunks más relevantes para cada pregunta usando similitud coseno
Bot RAG — combina los chunks recuperados con mistral-large-latest para responder preguntas sobre los datos
Chat libre — interfaz interactiva donde puedes escribir cualquier pregunta. Prefija con debug para ver qué chunks usó el bot
Preguntas de ejemplo — tres consultas de prueba para verificar el bot
Guardar archivos — exporta sales_clean.csv, sales_normalized.csv y sales_chunks.txt

Requisitos

Google Colab
API key de Mistral → console.mistral.ai
Agregar la key en Colab Secrets con el nombre MISTRAL_KEY

Dataset
sales_data_sample.csv — dataset público de ventas de Kaggle (kyanyoga/sample-sales-data)
