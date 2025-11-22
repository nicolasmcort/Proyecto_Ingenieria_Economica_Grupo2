import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import joblib
import warnings
# Asegúrate de que este módulo esté en el mismo directorio
from economic_formulas import calculate_irr_from_series, calculate_npv_from_series 

# Ignorar advertencias
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# --- 1. Carga y Preparación de Datos ---
try:
    df = pd.read_csv('american_bankruptcy_dataset.csv')
except FileNotFoundError:
    print("Error: El archivo 'american_bankruptcy_dataset.csv' no se encontró.")
    exit()

# Mapear la etiqueta binaria
df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})
cols_to_process = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'fyear', 'status_label']
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=cols_to_process, inplace=True)
df.sort_values(['company_name', 'fyear'], inplace=True)
print("Datos cargados. Usando el motor de Scikit-learn.")

# --- 2. Ingeniería de Características (Estable) ---
WINDOW_SIZE = 3
FALLBACK_INTEREST_RATE = 0.05

results = []
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

        # Nota: vp_neto se calcula pero no se usa como feature final (al igual que el script original)
        vp_neto = calculate_npv_from_series(flujo_neto_operativo, i_deduced) 
        
        # Validación para evitar división por cero en los ratios
        if window['X10'].iloc[-1] == 0:
            roa = np.nan
            debt_ratio = np.nan
        else:
            roa = window['X6'].iloc[-1] / window['X10'].iloc[-1]
            debt_ratio = window['X17'].iloc[-1] / window['X10'].iloc[-1]


        results.append({
            'company_name': name,
            'fyear': window['fyear'].iloc[-1],
            'status_label': window['status_label'].iloc[-1],
            'ROA': roa,
            'Debt_Ratio': debt_ratio,
            'X1': window['X1'].iloc[-1], # <<< AÑADIDA X1 EN EL RESULTADO
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

# --- 3. Limpieza Final ---
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)

if final_df.empty:
    print("Error: No se pudieron generar características válidas.")
    exit()

# --- 4. Definición y Entrenamiento del Modelo con Scikit-learn ---
y = final_df['status_label']

# MODIFICACIÓN CLAVE: INCLUIR 'X1' EN LA LISTA DE FEATURES (17 VARIABLES AHORA)
feature_columns = ['X1', 'ROA', 'Debt_Ratio', 'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'X9', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X18']
X = final_df[feature_columns]

# Crear un pipeline con LogisticRegression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced', C=0.1, max_iter=1000)) 
])

# Entrenar el pipeline
pipeline.fit(X, y)

print("\nEntrenamiento con Scikit-learn completado. El modelo ahora utiliza X1.")

# --- 5. Inspección de Importancia de Características (Debug) ---
print("\n" + "="*80)
print("Inspección de Coeficientes del Modelo de Regresión Logística")
print("="*80)
# Los coeficientes están dentro del paso 'lr' del pipeline
coeficientes = pd.DataFrame(pipeline.named_steps['lr'].coef_[0], index=feature_columns, columns=['Coeficiente'])
print("Un coeficiente positivo indica que la característica aumenta la probabilidad de quiebra.")
print(coeficientes.sort_values(by='Coeficiente', ascending=False))
print("="*80)


# --- 6. Guardar el Modelo ---
model_payload = {'model': pipeline, 'feature_columns': feature_columns}
# Se recomienda guardar con un nombre nuevo para no sobrescribir la versión anterior
joblib.dump(model_payload, 'bankruptcy_model_v2.joblib')

print("\nModelo entrenado y guardado exitosamente en 'bankruptcy_model_v2.joblib'")
print("Características finales en el modelo:", feature_columns)