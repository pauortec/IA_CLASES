# IA_CLASES — Proyectos de Inteligencia Artificial

Repositorio de proyectos prácticos de IA desarrollados en Google Colab
durante el curso. Cada notebook implementa un concepto diferente usando
Mistral AI como modelo de lenguaje principal.

---

## Archivos del repositorio

### mistral_agente.ipynb
Agente de análisis de datos construido con LangChain y Mistral AI.
El agente puede ejecutar código pandas de forma autónoma para responder
cualquier pregunta sobre el dataset en lenguaje natural.

Contenido:
- Instalación de langchain, langchain-mistralai y langchain-experimental
- Imports y configuración del entorno
- Carga del dataset sales_data_sample.csv
- Limpieza de datos con pandas: valores nulos, duplicados y estandarización
- Inicialización del modelo ChatMistralAI mistral-large-latest
- Creación del agente con create_pandas_dataframe_agent y descripción de las 25 columnas
- Ejemplos de consultas con agent.invoke(): ventas totales, promedio, top clientes
- Celda de prompt libre donde el usuario escribe su propia pregunta
- Chat interactivo en bucle con manejo de errores y rate limit

### rag_langchain_mistral.ipynb
Sistema RAG (Retrieval-Augmented Generation) completo usando LangChain,
FAISS y Mistral AI. En lugar de pasar todo el dataset al modelo, el sistema
busca solo los fragmentos más relevantes para cada pregunta antes de responder.

Contenido:
- Instalación de langchain, langchain-mistralai, langchain-community y faiss-cpu
- Imports desde langchain_core: Document, ChatPromptTemplate, StrOutputParser, RunnablePassthrough
- Carga del dataset CSV con files.upload()
- Función crear_documentos(): convierte el DataFrame en 10 Documents con metadata temática
- MistralAIEmbeddings + FAISS.from_documents() para construir el vector store
- Retriever con search_type=similarity y k=3 documentos por búsqueda
- ChatMistralAI temperature=0 con prompt en español
- Función formatear_docs() para estructurar el contexto recuperado
- Cadena RAG moderna con LCEL usando el operador pipe: retriever | prompt | llm | StrOutputParser
- Pruebas con rag_chain.invoke() pasando la pregunta como string
- Chat interactivo con manejo automático del error 429 rate limit

### sales_clean.csv
Dataset de ventas procesado con valores originales y columnas estandarizadas.
Generado tras el pipeline de limpieza del notebook.

### sales_normalized.csv
Dataset de ventas con columnas numéricas normalizadas en rango [0,1]
usando MinMaxScaler de scikit-learn. Listo para usar en modelos de ML.

---

## Conceptos aplicados

| Concepto | Descripción |
|---|---|
| Agente LangChain | Sistema autónomo que decide qué herramienta usar para responder |
| create_pandas_dataframe_agent | Agente especializado en ejecutar código pandas sobre un DataFrame |
| invoke | Método de LangChain para ejecutar el agente con una pregunta |
| RAG | Retrieval-Augmented Generation: busca contexto relevante antes de generar la respuesta |
| Embeddings | Representación vectorial del texto para búsqueda semántica |
| Vector Store FAISS | Base de datos vectorial en memoria para búsqueda por similitud coseno |
| Retriever | Interfaz que recupera los documentos más relevantes para cada pregunta |
| LCEL | LangChain Expression Language: conecta componentes con el operador pipe |

---

## Tecnologías

- Python 3.12
- Google Colab
- LangChain y LangChain Core
- Mistral AI (mistral-large-latest y mistral-embed)
- FAISS para el vector store
- pandas y numpy para procesamiento de datos
- scikit-learn para normalización

---

## Requisitos para ejecutar

1. Abrir el notebook en Google Colab
2. Ir al panel izquierdo icono de llave Secrets
3. Agregar un nuevo secret con el nombre MISTRAL_KEY
4. Pegar tu API key de Mistral como valor
5. Ejecutar las celdas en orden de arriba a abajo

Obtén tu API key gratis en https://console.mistral.ai

---

## Dataset

sales_data_sample.csv — dataset público de ventas disponible en Kaggle
bajo el nombre kyanyoga/sample-sales-data. Contiene 2823 registros de
órdenes de venta con 25 columnas incluyendo productos, clientes, países,
fechas y montos de venta.
