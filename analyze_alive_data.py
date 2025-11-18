import pandas as pd
import numpy as np
import warnings
from economic_formulas import calculate_irr_from_series, calculate_npv_from_series

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

try:
    df = pd.read_csv('american_bankruptcy_dataset.csv')
except FileNotFoundError:
    print('Error: El archivo american_bankruptcy_dataset.csv no se encontró.')
    exit()

df['status_label'] = df['status_label'].map({'alive': 0, 'failed': 1})
cols_to_process = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'fyear', 'status_label']
for col in cols_to_process:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(subset=cols_to_process, inplace=True)
df.sort_values(['company_name', 'fyear'], inplace=True)

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

        vp_neto = calculate_npv_from_series(flujo_neto_operativo, i_deduced)
        roa = window['X6'].iloc[-1] / window['X10'].iloc[-1]
        debt_ratio = window['X17'].iloc[-1] / window['X10'].iloc[-1]

        results.append({
            'company_name': name,
            'fyear': window['fyear'].iloc[-1],
            'status_label': window['status_label'].iloc[-1],
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

final_df = pd.DataFrame(results)
final_df.replace([np.inf, -np.inf], np.nan, inplace=True)
final_df.dropna(inplace=True)

# Filter for 'alive' companies
alive_df = final_df[final_df['status_label'] == 0]

feature_columns = ['ROA', 'Debt_Ratio', 'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'X9', 'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X18']
X_alive = alive_df[feature_columns]

print('Descriptive statistics for "alive" companies in training features:')
print(X_alive.describe().to_string())
