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