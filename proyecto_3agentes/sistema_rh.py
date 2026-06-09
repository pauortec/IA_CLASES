import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage

# =======================================================
# FASE 1: LIMPIEZA Y PREPARACIÓN DE DATOS
# =======================================================
def procesar_datos_empleados(ruta_archivo):
    """
    Lee los datos de RRHH, maneja valores vacíos y normaliza usando MinMaxScaler.
    """
    print("⏳ [Fase 1] Cargando y preprocesando datos de empleados...")

    df_rh = pd.read_csv(ruta_archivo)

    df_rh.fillna(df_rh.median(numeric_only=True), inplace=True)

    scaler = MinMaxScaler()
    columnas_numericas = df_rh.select_dtypes(include='number').columns.tolist()
    if 'abandono' in columnas_numericas:
        columnas_numericas.remove('abandono')
    df_rh[columnas_numericas] = scaler.fit_transform(df_rh[columnas_numericas])

    print(f"✅ [Fase 1] Datos procesados: {df_rh.shape[0]} filas, {df_rh.shape[1]} columnas.")
    return df_rh

# =======================================================
# FASE 2: MODELADO PREDICTIVO (RANDOM FOREST)
# =======================================================
def construir_modelo_fuga(dataframe):
    """
    Entrena un clasificador Random Forest para predecir si un empleado renunciará.
    """
    print("⏳ [Fase 2] Entrenando modelo Random Forest...")

    X = dataframe.drop(columns=['abandono'])
    y = dataframe['abandono']

    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(
        X, y, test_size=0.25, random_state=77
    )

    clasificador_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=77)
    clasificador_rf.fit(X_entrenamiento, y_entrenamiento)

    y_prediccion = clasificador_rf.predict(X_prueba)
    precision = precision_score(y_prueba, y_prediccion, zero_division=0)
    recall = recall_score(y_prueba, y_prediccion, zero_division=0)
    f1 = f1_score(y_prueba, y_prediccion, zero_division=0)

    texto_metricas = (
        f"Precisión: {precision:.2%} | Recall: {recall:.2%} | F1-Score: {f1:.2%}\n"
        f"El modelo evaluó {len(y_prueba)} empleados del conjunto de prueba."
    )

    print(f"✅ [Fase 2] Modelo entrenado.\n{texto_metricas}")
    return clasificador_rf, texto_metricas