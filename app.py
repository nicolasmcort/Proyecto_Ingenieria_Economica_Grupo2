import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from utils.economic_formulas import calculate_npv_from_series, calculate_aw_from_pv, calculate_discounted_payback_period

# --- Configuración de la Aplicación Flask ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Necesario para los mensajes flash

# --- Funciones auxiliares ---

def allowed_file(filename):
    """Verifica si la extensión del archivo es válida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_prediction(file_path):
    """Procesa un archivo CSV y devuelve las predicciones de quiebra."""
    MODEL_FILE = os.path.join('models', 'bankruptcy_model_v2.joblib')
    WINDOW_SIZE = 3
    FALLBACK_INTEREST_RATE = 0.05

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Archivo del modelo no encontrado: '{MODEL_FILE}'. Asegúrate de haber entrenado y guardado 'bankruptcy_model_v2.joblib'."
        )

    model_payload = joblib.load(MODEL_FILE)
    model = model_payload['model']
    feature_columns = model_payload['feature_columns']
    # El modelo entrenado puede esperar una columna 'TIR', la reemplazaremos por 'i'
    # si está presente. Esto puede requerir re-entrenar el modelo.
    if 'TIR' in feature_columns:
        feature_columns = [col if col != 'TIR' else 'i' for col in feature_columns]
    elif 'i' not in feature_columns:
        feature_columns.append('i')

    # Lectura del CSV con manejo de codificaciones y delimitadores
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception:
            raise ValueError("Error de codificación: El archivo CSV no está en formato UTF-8 ni Latin-1.")
    except pd.errors.ParserError:
        try:
            df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, delimiter=';', encoding='latin1')
            except Exception:
                raise ValueError(
                    "Error de formato: El archivo CSV no parece estar delimitado por comas ni por punto y coma, o tiene un formato incorrecto."
                )
        except Exception:
            raise ValueError(
                "Error de formato: El archivo CSV no parece estar delimitado por comas ni por punto y coma, o tiene un formato incorrecto."
            )
    except Exception as e:
        raise ValueError(f"Error al leer el archivo CSV: {e}")

    # Ingeniería de características (idéntica al entrenamiento)
    cols_to_process = [
        'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10',
        'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'fyear'
    ]
    for col in cols_to_process:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.sort_values(['company_name', 'fyear'], inplace=True)

    prediction_features = []
    grouped = df.groupby('company_name')

    for name, group in grouped:
        if len(group) < WINDOW_SIZE:
            continue
        # Tomar los últimos WINDOW_SIZE años
        last_window = group.iloc[-WINDOW_SIZE:]
        # Tasa de Interés de Oportunidad (TIO) estándar
        i_estandar = 0.12 # 12%

        flujo_neto_operativo = (last_window['X16'] - last_window['X18']).tolist()
        vp_neto = calculate_npv_from_series(flujo_neto_operativo, i_estandar)
        va = calculate_aw_from_pv(vp_neto, i_estandar, WINDOW_SIZE)
        
        # Payback Period (Discounted)
        # Usamos X10 (Activos Totales) del primer año como inversión inicial aproximada
        initial_investment = last_window['X10'].iloc[0]
        payback = calculate_discounted_payback_period(initial_investment, flujo_neto_operativo, i_estandar)
        
        # Evitar división por cero
        last_X10 = last_window['X10'].iloc[-1]
        if last_X10 == 0:
            continue
        roa = last_window['X6'].iloc[-1] / last_X10
        debt_ratio = last_window['X17'].iloc[-1] / last_X10
        prediction_features.append({
            'company_name': name,
            'fyear': last_window['fyear'].iloc[-1],
            'ROA': roa,
            'Debt_Ratio': debt_ratio,
            'VA': va,
            'Payback_Period': payback,
            'i': i_estandar,
            'X1': last_window['X1'].iloc[-1],
            'X2': last_window['X2'].iloc[-1],
            'X3': last_window['X3'].iloc[-1],
            'X4': last_window['X4'].iloc[-1],
            'X5': last_window['X5'].iloc[-1],
            'X6': last_window['X6'].iloc[-1], # Added X6
            'X7': last_window['X7'].iloc[-1],
            'X8': last_window['X8'].iloc[-1],
            'X9': last_window['X9'].iloc[-1],
            'X10': last_window['X10'].iloc[-1], # Added X10
            'X11': last_window['X11'].iloc[-1],
            'X12': last_window['X12'].iloc[-1],
            'X13': last_window['X13'].iloc[-1],
            'X14': last_window['X14'].iloc[-1],
            'X15': last_window['X15'].iloc[-1],
            'X16': last_window['X16'].iloc[-1],
            'X17': last_window['X17'].iloc[-1], # Added X17
            'X18': last_window['X18'].iloc[-1]
        })

    if not prediction_features:
        return None

    predict_df = pd.DataFrame(prediction_features)
    
    # Handle infinite Payback Period (replace with a large number for visualization logic)
    if 'Payback_Period' in predict_df.columns:
        predict_df['Payback_Period'].replace([np.inf, -np.inf], 1000, inplace=True)

    predict_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    predict_df.dropna(inplace=True)
    if predict_df.empty:
        return None

    X_pred = predict_df[feature_columns]
    probabilities = model.predict_proba(X_pred)[:, 1]
    predict_df['prediction_probability'] = probabilities

    results = []
    for _, row in predict_df.iterrows():
        prob = row['prediction_probability']
        if prob > 0.5:
            diagnosis = 'ALTO RIESGO'
            color_class = 'danger'
        elif prob > 0.2:
            diagnosis = 'RIESGO MODERADO'
            color_class = 'warning'
        else:
            diagnosis = 'BAJO RIESGO'
            color_class = 'success'
        warnings = []
        if not (-10 < row['ROA'] < 10):
            warnings.append(
                "Advertencia: El ratio ROA (Rentabilidad sobre Activos) tiene un valor extremo, lo que sugiere un posible error en los datos de entrada. La predicción puede no ser fiable."
            )
        if not (0 <= row['Debt_Ratio'] < 10):
            warnings.append(
                "Advertencia: El ratio de Endeudamiento tiene un valor extremo, lo que sugiere un posible error en los datos de entrada. La predicción puede no ser fiable."
            )
        # Formatear características para la plantilla
        processed_features = {}
        
        # Define display order: X1-X18 first, then metrics
        display_order = [f'X{j}' for j in range(1, 19)] + ['ROA', 'Debt_Ratio', 'VA', 'Payback_Period', 'i']
        
        for col in display_order:
            if col not in row: continue
            value = row[col]
            desc_info = feature_descriptions.get(col, {})
            color = desc_info.get('impact_color', lambda v: 'text-muted')(value)
            unit = desc_info.get('unit', '')
            if unit == 'ratio':
                formatted = f"{value:.4f}"
            elif unit == 'currency':
                formatted = f"{value:,.2f}"
            elif unit == 'percent':
                formatted = f"{value * 100:.2f}%"
            elif unit == 'years':
                if value >= 1000: # Check for the placeholder value
                    formatted = "> 50 años"
                else:
                    formatted = f"{value:.1f} años"
            else:
                formatted = f"{value:.4f}"
            processed_features[col] = {
                'value': formatted,
                'desc': desc_info.get('desc', 'Descripción no disponible.'),
                'impact_color': color
            }
        results.append({
            'company': row['company_name'],
            'year': int(row['fyear']),
            'probability': f"{prob * 100:.2f}%",
            'diagnosis': diagnosis,
            'color_class': color_class,
            'features': processed_features,
            'warnings': warnings
        })
    return results

# --- Descripciones de características (incluye TIR) ---
feature_descriptions = {
    'X1': {
        'desc': 'Activos Corrientes Totales. Recursos que se espera convertir en efectivo o consumir en un año. Es una cifra monetaria. Un valor alto en relación con los Pasivos Corrientes (X14) es favorable.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'ROA': {
        'desc': 'Return on Assets (ROA). Mide la eficiencia para generar ganancias con los activos. Se expresa como un ratio, donde un valor de 0.05 equivale a un 5%. Un ROA superior a 0.05 (5%) generalmente se considera bueno.',
        'unit': 'ratio',
        'impact_color': lambda val: 'text-success' if val > 0.05 else ('text-warning' if val > 0 else 'text-danger')
    },
    'Debt_Ratio': {
        'desc': 'Debt Ratio (Ratio de Endeudamiento). Proporción de activos financiados por deuda. Se expresa como un ratio, donde un valor de 0.6 equivale a un 60%. Un ratio inferior a 0.6 es saludable. Superior a 1.0 indica que la deuda supera los activos.',
        'unit': 'ratio',
        'impact_color': lambda val: 'text-success' if val < 0.4 else ('text-warning' if val < 0.6 else 'text-danger')
    },
    'VA': {
        'desc': 'Valor Anual (VA). Valor anual equivalente de los flujos de caja operativos. Normaliza el VPN por período, permitiendo comparar empresas de diferentes tamaños. Un VA positivo indica que las operaciones generan valor sostenible por año. Fórmula del Capítulo 6: VA = VPN × (A/P, i, n).',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'Payback_Period': {
        'desc': 'Periodo de Recuperación Descontado. Tiempo estimado para recuperar la inversión inicial (Activos Totales) mediante flujos descontados. Un valor bajo indica menor riesgo y mayor liquidez.',
        'unit': 'years',
        'impact_color': lambda val: 'text-success' if val < 5 else ('text-warning' if val < 10 else 'text-danger')
    },
    'i': {
        'desc': "Tasa de Interés de Oportunidad (TIO). Tasa de descuento estándar utilizada para los cálculos de Valor Anual. Representa el rendimiento mínimo aceptable de una inversión. En este caso, se usa un valor por defecto del 12%.",
        'unit': 'percent',
        'impact_color': lambda val: 'text-muted'
    },
    'X2': {
        'desc': 'Costo de Bienes Vendidos. Costos directos de producción. Es una cifra monetaria. Su interpretación depende de la relación con los ingresos y la industria.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X3': {
        'desc': 'Depreciación y Amortización. Gasto no monetario que reduce el valor de los activos. Es una cifra monetaria. No es directamente un indicador de riesgo.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X4': {
        'desc': 'EBITDA. Ganancias antes de intereses, impuestos, etc. Es una cifra monetaria que mide la rentabilidad operativa. Un valor positivo y creciente es favorable.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X5': {
        'desc': 'Inventario. Valor de los bienes para la venta. Es una cifra monetaria. Un nivel muy alto en comparación con las ventas puede indicar problemas.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X6': {
        'desc': 'Ganancia Neta (Net Income). Beneficio total después de todos los gastos e impuestos. Es una cifra monetaria. Es fundamental para calcular el ROA.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X7': {
        'desc': 'Cuentas por Cobrar Totales. Dinero que los clientes deben. Es una cifra monetaria. Un aumento drástico puede ser una señal de alerta.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X8': {
        'desc': 'Valor de Mercado. Capitalización de mercado total. Es una cifra monetaria. Un valor alto y estable es una señal de confianza del mercado.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X9': {
        'desc': 'Ventas Netas. Ingresos totales por ventas. Es una cifra monetaria. Un valor alto y creciente es favorable.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X10': {
        'desc': 'Activos Totales. Suma de todos los recursos de la empresa. Es el denominador para calcular ROA y Debt Ratio.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X11': {
        'desc': 'Deuda Total a Largo Plazo. Deudas con vencimiento mayor a un año. Es una cifra monetaria. Debe evaluarse en relación con los activos y ganancias.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X12': {
        'desc': 'EBIT. Ganancias Antes de Intereses e Impuestos. Es una cifra monetaria que mide la rentabilidad operativa. Un valor positivo es crucial.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X13': {
        'desc': 'Ganancia Bruta. Ganancia tras deducir costos de producción. Es una cifra monetaria. Un margen bruto saludable depende de la industria.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X14': {
        'desc': 'Pasivos Corrientes Totales. Deudas a pagar en menos de un año. Es una cifra monetaria. Un valor bajo en relación a los activos corrientes es favorable.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X15': {
        'desc': 'Ganancias Retenidas. Ganancias reinvertidas en la empresa. Es una cifra monetaria. Un historial de crecimiento constante es muy favorable.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X16': {
        'desc': 'Ingresos Totales. Dinero total generado por ventas. Es una cifra monetaria. El crecimiento constante es una señal de salud.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-success' if val > 0 else 'text-danger'
    },
    'X17': {
        'desc': 'Pasivos Totales. Deuda total de la empresa. Es una cifra monetaria. Se usa para calcular el Debt Ratio.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    },
    'X18': {
        'desc': 'Gastos Operativos Totales. Costos del funcionamiento del negocio. Es una cifra monetaria. Deben ser sostenibles en relación con los ingresos.',
        'unit': 'currency',
        'impact_color': lambda val: 'text-muted'
    }
}

GITHUB_REPO_URL = "https://github.com/nicolasmcort/Proyecto_Ingenieria_Economica_Grupo2.git"

# --- Rutas de la Aplicación Web ---

@app.route('/')
def index():
    return render_template('index.html', github_repo_url=GITHUB_REPO_URL)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No se encontró el campo del archivo.')
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash('Ningún archivo seleccionado.')
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        try:
            predictions = get_prediction(filepath)
            if predictions is None:
                flash('No se pudieron generar predicciones. Verifica que el CSV tenga suficientes datos (al menos 3 años por empresa) y que los datos sean válidos.')
                return redirect(url_for('index'))
            return render_template('results.html', predictions=predictions, feature_descriptions=feature_descriptions, uploaded_filename=filename)
        except Exception as e:
            import traceback
            app.logger.error(f"Error al procesar el archivo: {e}\n{traceback.format_exc()}")
            flash(f'Ocurrió un error al procesar el archivo: {e}. Por favor, revisa el formato de tu CSV.')
            return redirect(url_for('index'))
    else:
        flash('Formato de archivo no permitido. Por favor, sube un archivo .csv')
        return redirect(url_for('index'))

@app.route('/visualize/<filename>/<company_name>')
def visualize(filename, company_name):
    """Crea una visualización del diagrama de flujo de caja para una empresa."""
    # Sanitize filename to prevent directory traversal
    safe_filename = secure_filename(filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

    if not os.path.exists(filepath):
        flash(f'No se encontró el archivo de datos: {safe_filename}')
        return redirect(url_for('index'))

    df = pd.read_csv(filepath)
    company_df = df[df['company_name'] == company_name].copy()

    if company_df.empty:
        flash(f'No se encontraron datos para la empresa: {company_name} en el archivo {safe_filename}')
        return redirect(url_for('index'))
		
		# Asegurarse que los datos son numéricos y ordenar
    company_df['fyear'] = pd.to_numeric(company_df['fyear'], errors='coerce')
    company_df['X16'] = pd.to_numeric(company_df['X16'], errors='coerce')
    company_df['X18'] = pd.to_numeric(company_df['X18'], errors='coerce')
    company_df.dropna(subset=['fyear', 'X16', 'X18'], inplace=True)
    company_df.sort_values('fyear', inplace=True)

    # Calcular Flujo de Caja Neto
    company_df['net_cash_flow'] = company_df['X16'] - company_df['X18']
    
    chart_data = {
        'labels': [int(y) for y in company_df['fyear']],
        'values': list(company_df['net_cash_flow']),
        'company_name': company_name
    }

    return render_template('visualize.html', chart_data=chart_data)

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)