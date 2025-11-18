import pandas as pd
import numpy as np
import joblib
import os
import sys
import warnings
import json # Import the json module
from economic_formulas import *

# Ignorar advertencias
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- 1. Carga del Modelo y Datos de Entrada ---
MODEL_FILE = 'bankruptcy_model.joblib'

# Aceptar un nombre de archivo desde la línea de comandos, con un valor por defecto
if len(sys.argv) > 1:
    INPUT_CSV = sys.argv[1]
else:
    INPUT_CSV = 'datos_para_predecir.csv'

if not os.path.exists(MODEL_FILE):
    print(json.dumps({"error": f"Archivo del modelo no encontrado: '{MODEL_FILE}'"}))
    exit()
if not os.path.exists(INPUT_CSV):
    print(json.dumps({"error": f"Archivo de datos de entrada no encontrado: '{INPUT_CSV}'"}))
    exit()

try:
    model_payload = joblib.load(MODEL_FILE)
    model = model_payload['model']
    feature_columns = model_payload['feature_columns']
except Exception as e:
    print(json.dumps({"error": f"Error al cargar el modelo: {e}"}))
    exit()

try:
    df = pd.read_csv(INPUT_CSV)
except Exception as e:
    print(json.dumps({"error": f"Error al leer el CSV de entrada: {e}"}))
    exit()

# print(f"Modelo y datos de entrada ('{INPUT_CSV}') cargados. Iniciando predicción con Scikit-learn.")

# --- 2. Ingeniería de Características (Idéntica al entrenamiento) ---
cols_to_process = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'fyear']
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=cols_to_process, inplace=True)
df.sort_values(['company_name', 'fyear'], inplace=True)

WINDOW_SIZE = 3
FALLBACK_INTEREST_RATE = 0.05

results_features = [] # To store features for prediction
grouped = df.groupby('company_name')

for name, group in grouped:
    if len(group) < WINDOW_SIZE:
        continue

    for window in group.rolling(window=WINDOW_SIZE):
        if len(window) < WINDOW_SIZE:
            continue

        flujo_neto_operativo = (window['X16'] - window['X18']).tolist()
        i_deduced = calculate_irr_from_series(flujo_neto_operativo)
        if i_deduced is None:
            i_deduced = FALLBACK_INTEREST_RATE

        vp_neto = calculate_npv_from_series(flujo_neto_operativo, i_deduced)
        roa = window['X6'].iloc[-1] / window['X10'].iloc[-1]
        debt_ratio = window['X17'].iloc[-1] / window['X10'].iloc[-1]

        results_features.append({
            'company_name': name,
            'fyear': window['fyear'].iloc[-1],
            'ROA': roa,
            'Debt_Ratio': debt_ratio,
            'X2': window['X2'].iloc[-1],
            'X3': window['X3'].iloc[-1],
            'X4': window['X4'].iloc[-1],
            'X5': window['X5'].iloc[-1],
            'X7': window['X7'].iloc[-1],
            'X8': window['X8'].iloc[-1],
            'X9': window['X9'].iloc[-1],
            'X11': window['X11'].iloc[-1],
            'X12': window['X12'].iloc[-1],
            'X13': window['X13'].iloc[-1],
            'X14': window['X14'].iloc[-1],
            'X15': window['X15'].iloc[-1],
            'X16': window['X16'].iloc[-1],
            'X18': window['X18'].iloc[-1]
        })

predict_df = pd.DataFrame(results_features)
# print("Ingeniería de características para predicción completada.")

# --- 3. Preparación de Datos para Predicción ---
predict_df.replace([np.inf, -np.inf], np.nan, inplace=True)
predict_df.dropna(inplace=True)

if predict_df.empty:
    print(json.dumps({"error": "No se pudieron calcular las características para la predicción."}))
    exit()

X_pred = predict_df[feature_columns]

# --- 4. Realizar y Mostrar la Predicción ---
probabilities = model.predict_proba(X_pred)[:, 1]
predict_df['prediction_probability'] = probabilities

final_predictions = []
for index, row in predict_df.iterrows():
    company = row['company_name']
    year = int(row['fyear'])
    prob = row['prediction_probability']

    diagnosis = ""
    if prob > 0.5:
        diagnosis = "ALTO RIESGO"
    elif prob > 0.2:
        diagnosis = "RIESGO MODERADO"
    else:
        diagnosis = "BAJO RIESGO"
    
    # Prepare feature values for output
    feature_values = {col: row[col] for col in feature_columns}

    final_predictions.append({
        "company_name": company,
        "fyear": year,
        "probability": round(prob, 4),
        "probability_percent": round(prob * 100, 2),
        "diagnosis": diagnosis,
        "features": feature_values
    })

print(json.dumps(final_predictions))
