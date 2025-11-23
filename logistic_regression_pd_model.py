import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import warnings
from economic_formulas import calculate_irr_from_series, calculate_npv_from_series, calculate_aw_from_pv

# Ignorar advertencias
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# ---------------------------------------------------------------------
# 1. Carga y preparación de datos
# ---------------------------------------------------------------------
try:
    df = pd.read_csv('american_bankruptcy_dataset.csv')
except FileNotFoundError:
    print("Error: El archivo 'american_bankruptcy_dataset.csv' no se encontró.")
    exit()

# Convertir etiqueta binaria
df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})

# Columnas a convertir a numérico
cols_to_process = [
    'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
    'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18',
    'fyear', 'status_label'
]
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores faltantes en las columnas de interés
df.dropna(subset=cols_to_process, inplace=True)
# Ordenar por empresa y año
df.sort_values(['company_name', 'fyear'], inplace=True)
print("Datos cargados y preparados.")

# ---------------------------------------------------------------------
# 2. Ingeniería de características (ventana de 3 años)
# ---------------------------------------------------------------------
WINDOW_SIZE = 3
FALLBACK_INTEREST_RATE = 0.05

results = []
for name, group in df.groupby('company_name'):
    if len(group) < WINDOW_SIZE:
        continue
    # Usar ventana móvil de tamaño fijo
    for i in range(WINDOW_SIZE, len(group) + 1):
        window = group.iloc[i - WINDOW_SIZE:i]
        # Flujo neto operativo
        flujo_neto_operativo = (window['X16'] - window['X18']).tolist()
        # TIR (Tasa Interna de Retorno)
        i_deduced = calculate_irr_from_series(flujo_neto_operativo)
        if i_deduced is None:
            i_deduced = FALLBACK_INTEREST_RATE
        # VPN y VA
        vp_neto = calculate_npv_from_series(flujo_neto_operativo, i_deduced)
        va = calculate_aw_from_pv(vp_neto, i_deduced, WINDOW_SIZE)
        # Ratios (evitar división por cero)
        last_X10 = window['X10'].iloc[-1]
        if last_X10 == 0:
            roa = np.nan
            debt_ratio = np.nan
        else:
            roa = window['X6'].iloc[-1] / last_X10
            debt_ratio = window['X17'].iloc[-1] / last_X10
        # Guardar fila de características
        results.append({
            'company_name': name,
            'fyear': window['fyear'].iloc[-1],
            'status_label': window['status_label'].iloc[-1],
            'ROA': roa,
            'Debt_Ratio': debt_ratio,
            'VA': va,
            'TIR': i_deduced,
            'X1': window['X1'].iloc[-1],
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

final_df = pd.DataFrame(results)
print("Ingeniería de características completada.")

# Limpiar valores infinitos y NaN
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)
if final_df.empty:
    print("Error: No se pudieron generar características válidas.")
    exit()

# ---------------------------------------------------------------------
# 3. Entrenamiento del modelo
# ---------------------------------------------------------------------
Y = final_df['status_label']
# Lista de columnas usadas por el modelo (incluye TIR)
feature_columns = [
    'X1', 'ROA', 'Debt_Ratio', 'VA', 'TIR',
    'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'X9',
    'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X18'
]
X = final_df[feature_columns]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(
        random_state=42,
        solver='liblinear',
        class_weight='balanced',
        C=0.1,
        max_iter=1000
    ))
])

pipeline.fit(X, Y)
print("Entrenamiento completado. Modelo entrenado con 19 características (incluye TIR).")

# Mostrar coeficientes (debug)
print("=" * 80)
print("Coeficientes del modelo (ordenados por importancia)")
coef_df = pd.DataFrame(pipeline.named_steps['lr'].coef_[0], index=feature_columns, columns=['Coeficiente'])
print(coef_df.sort_values(by='Coeficiente', ascending=False))
print("=" * 80)

# Guardar modelo
model_payload = {'model': pipeline, 'feature_columns': feature_columns}
joblib.dump(model_payload, 'bankruptcy_model_v2.joblib')
print("Modelo guardado en 'bankruptcy_model_v2.joblib'.")