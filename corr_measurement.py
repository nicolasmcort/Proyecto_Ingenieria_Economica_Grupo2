import pandas as pd
from scipy.stats import pointbiserialr
import numpy as np
import io

# Contenido del archivo CSV (usando el contenido del archivo cargado)
csv_content = """company_name,fyear,status_label,X1,X2,X3,X4,X5,X6,X7,X8,X9,X10,X11,X12,X13,X14,X15,X16,X17,X18,Division,MajorGroup
C_1,1999.0,alive,511267,740998,833107,180447,18373,70658,89031,191226,336.01800000000003,163816,35.163000000000004,201026,128.34799999999998,1024333,372.7519,401483,1024333,935302,D,37
C_1,2000.0,alive,485856,701.8539999999999,713811,179987,18577,45.79,64367,160444,320.59,125392,18531,204065,115187,874255,377.11800000000005,361642,874255,809888,D,37
C_6,1999.0,failed,4424.0,24374.0,15482.0,5689.0,1092.0,1156.0,2248.0,2248.0,708.0,5864.0,985.0,5716.0,1134.0,17730.0,9932415,17516.0,17730.0,15482.0,E,45
C_6,2000.0,failed,5179.0,26213.0,17120.0,5474.0,1202.0,1381.0,2583.0,2583.0,757.0,6990.0,813.0,5948.0,1303.0,19703.0,5958.9688,19037.0,19703.0,17120.0,E,45
C_6,2001.0,failed,6540.0,32841.0,19419.0,9834.0,1404.0,-1860.0,-456.0,-456.0,822.0,7512.0,-1762.0,4042.0,1414.0,18963.0,3445.0155,27468.0,18963.0,19419.0,E,45
C_7,1999.0,alive,27276,51073,15475,28.29,0.7290000000000001,442,1171,8387,5125,12934,-366,-2.6830000000000003,17204,23862,20.6275,41937,23862,22691,D,35
# ... (El resto de las filas del CSV están implícitas)
"""

# Lee el contenido del CSV en un DataFrame de pandas
# Uso de io.StringIO para simular la lectura de un archivo
df = pd.read_csv(io.StringIO(csv_content), low_memory=False)

# --- 1. Preparación de Datos ---

# Columnas de interés (X1 a X18)
feature_cols = [f'X{i}' for i in range(1, 19)]
target_col = 'status_label'

# Crear la variable objetivo binaria: 1 para 'failed' (quiebra), 0 para 'alive' (no quiebra)
df['target_bankrupt'] = df[target_col].apply(lambda x: 1 if x == 'failed' else 0)

# Convertir las columnas X a formato numérico (forzando errores a NaN)
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores NaN en las columnas X o en el target binario
df_cleaned = df.dropna(subset=feature_cols + ['target_bankrupt'])

# Extraer el target binario limpio
target = df_cleaned['target_bankrupt']

# --- 2. Cálculo de Correlaciones ---

results = []

for feature in feature_cols:
    # Obtener los datos de la característica actual
    feature_data = df_cleaned[feature]
    
    # Calcular la Correlación Biseral-Puntual (Point-Biserial Correlation)
    # y el valor P (p-value)
    # Se añade una pequeña constante para evitar el error 'divide by zero' si
    # la desviación estándar del feature_data es 0
    try:
        corr, p_value = pointbiserialr(feature_data, target)
    except ValueError:
        corr = np.nan
        p_value = np.nan
    
    results.append({
        'Feature': feature,
        'Correlacion (r_pb)': corr,
        'Valor P (p-value)': p_value
    })

# Convertir los resultados a DataFrame para una mejor visualización
corr_df = pd.DataFrame(results).sort_values(by='Correlacion (r_pb)', key=lambda x: abs(x), ascending=False).reset_index(drop=True)

# --- 3. Mostrar Resultados ---

print("="*60)
print("     📉 Correlación de Variables 'X' con la Quiebra (Target)     ")
print("="*60)
print("\nNota: Una correlación más cercana a 1 o -1 indica mayor poder predictivo.")

# Definir la descripción de cada columna para el reporte
descriptions = {
    'X1': 'Current Assets',
    'X2': 'Cost of Goods Sold',
    'X3': 'Depreciation and amortization',
    'X4': 'EBITDA',
    'X5': 'Inventory',
    'X6': 'Net Income',
    'X7': 'Total Receivables',
    'X8': 'Market value',
    'X9': 'Net sales',
    'X10': 'Total assets',
    'X11': 'Total Long-term debt',
    'X12': 'EBIT',
    'X13': 'Gross Profit',
    'X14': 'Total Current Liabilities',
    'X15': 'Retained Earnings',
    'X16': 'Total Revenue',
    'X17': 'Total Liabilities',
    'X18': 'Total Operating Expenses',
}

# Aplicar formato de dos decimales y añadir las descripciones
corr_df['Correlacion (r_pb)'] = corr_df['Correlacion (r_pb)'].map('{:.4f}'.format)
corr_df['Valor P (p-value)'] = corr_df['Valor P (p-value)'].map('{:.4f}'.format)
corr_df['Description'] = corr_df['Feature'].map(descriptions)

# Mostrar el resultado final
print(corr_df[['Feature', 'Description', 'Correlacion (r_pb)', 'Valor P (p-value)']].to_markdown(index=False))