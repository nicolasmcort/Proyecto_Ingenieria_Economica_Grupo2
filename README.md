# Análisis de Riesgo de Quiebra Corporativa
<br>
<div align="center">
  <img src="https://www.pngkey.com/png/detail/268-2688228_universidad-nacional-colombia-logo.png" width="230" alt="Logo Universidad Nacional de Colombia">
</div>
<h2 align="center">UNIVERSIDAD NACIONAL DE COLOMBIA</h2> 
<p align="center">
  <strong>Ingeniería Económica - 2015703</strong>
</p>
<br><br>

### Autores: 

- Fabián David Mora Martínez (fmoram@unal.edu.co)
- Ever Nicolás Muñoz Cortés (evmunoz@unal.edu.co)
- Isaias David Gallardo Felizzola (igallardo@unal.edu.co)
- Ángel Manuel Cortavarria Salas (acortavarria@unal.edu.co)
- Nicolás Alejandro Diosa Benavides (ndiosab@unal.edu.co)
- Juan Esteban Ocampo Vidal (jocampov@unal.edu.co)
- Paula Alejandra Murcia Ramírez (pmurciar@unal.edu.co)

### Docente: 
Diego Alejandro Hernández Castañeda 
<br><br>

---

## Contenido

- [1. Objetivos](#1-objetivos)
- [2. Principios de Ingeniería Económica Aplicados](#2-principios-de-ingeniería-económica-aplicados)
- [3. Guía de Usuario](#3-guía-de-usuario)
- [4. Guía para Desarrolladores](#4-guía-para-desarrolladores)
- [5. Estructura del Proyecto](#5-estructura-del-proyecto)
- [6. Modelo de Machine Learning](#6-modelo-de-machine-learning)
- [7. Tecnologías Utilizadas](#7-tecnologías-utilizadas)
- [8. Conclusiones](#8-conclusiones)
- [9. Referencias](#9-referencias)

<br><br>

---

## 1. Objetivos

- **Objetivo General**: Desarrollar una aplicación web interactiva que, aplicando conceptos de ingeniería económica y un modelo de Machine Learning, permita predecir la probabilidad de quiebra de una empresa a partir de sus datos financieros históricos.
- **Objetivos Específicos**:
  - Implementar un modelo de Machine Learning en Python para clasificar a las empresas en categorías de riesgo.
  - Realizar una ingeniería de características para transformar datos crudos en ratios financieros significativos.
  - Diseñar una interfaz de usuario web intuitiva utilizando Flask y Bootstrap para facilitar la carga de datos y la visualización clara de los resultados.
  - Empaquetar la aplicación en contenedores Docker para garantizar un despliegue y ejecución sencillos.
<br><br>

---

## 2. Principios de Ingeniería Económica Aplicados

Para construir el modelo predictivo, se aplicaron ratios y métricas fundamentales del análisis financiero, que son pilares de la ingeniería económica para evaluar la salud y viabilidad de una empresa.

- **Ratios de Rentabilidad**:
  - **ROA (Return on Assets)**: Mide la rentabilidad de la empresa en relación con sus activos totales. Un ROA bajo o negativo indica un uso ineficiente de los activos para generar ganancias.
  - **EBIT y EBITDA**: Miden la ganancia operativa de la empresa antes de deducir intereses e impuestos (EBIT) y, adicionalmente, depreciación y amortización (EBITDA). Son indicadores clave de la capacidad de una empresa para generar beneficios a partir de sus operaciones principales.

- **Ratios de Endeudamiento**:
  - **Debt Ratio (Ratio de Endeudamiento)**: Compara la deuda total con los activos totales. Un ratio elevado puede indicar un apalancamiento excesivo y un mayor riesgo de insolvencia si la empresa no puede cumplir con sus obligaciones.

- **Análisis de Estados Financieros**: El modelo utiliza directamente variables extraídas de los estados financieros, como:
  - **Ingresos Totales (X16)** y **Ventas Netas (X9)**: Indican la capacidad de la empresa para generar ventas.
  - **Costos y Gastos (X2, X18)**: Reflejan la estructura de costos de la empresa.
  - **Activos y Pasivos (X1, X10, X14, X17)**: Permiten evaluar la estructura de capital y la liquidez.
  - **Ganancias Retenidas (X15)**: Muestran la porción de las ganancias que se reinvierte en el negocio, un indicador de crecimiento y solidez a largo plazo.

La combinación de estos indicadores permite al modelo de Regresión Logística aprender los patrones que distinguen a las empresas saludables de aquellas en riesgo de quiebra.
<br><br>

---

## 3. Guía de Usuario

Esta guía explica cómo utilizar la aplicación para obtener y entender el diagnóstico de riesgo financiero de una empresa.

### Paso 1: Preparar el Archivo CSV

- **Formato**: El archivo debe ser de tipo `.csv`.
- **Columnas Requeridas**: Debe contener `company_name`, `fyear` (año fiscal), y las variables financieras (`X1`, `X2`, etc.). Puedes usar los archivos de ejemplo (`C_low_risk.csv`, `C_high_risk.csv`, etc.) como plantilla.
- **Datos Históricos**: Para que una empresa sea analizada, debe tener registros de **al menos 3 años consecutivos**.

### Paso 2: Subir y Analizar

1.  En la página de inicio, haz clic en **"Seleccionar archivo"** y elige tu archivo `.csv`.
2.  Presiona el botón **"Analizar y Predecir"**.
3.  La aplicación te redirigirá a la página de resultados.

### Paso 3: Interpretar los Resultados

Para cada empresa analizada, verás una tarjeta de resultados como la siguiente:

#### a) Diagnóstico General
En la parte superior, encontrarás el resumen del análisis:
- **Diagnóstico**: `BAJO RIESGO`, `RIESGO MODERADO` o `ALTO RIESGO`, con un color distintivo (verde, amarillo o rojo).
- **Nombre de la Empresa y Año**: Identifica la empresa y el último año de datos utilizado.
- **Probabilidad de Quiebra**: El porcentaje de riesgo que el modelo ha calculado.

#### b) Advertencias sobre Datos (Si Aplica)
Si los datos de entrada para una empresa parecen anómalos (por ejemplo, ratios financieros con valores extremos), aparecerá una alerta como esta. Estas advertencias indican que la predicción para esa empresa puede no ser fiable.

<div class="alert alert-warning small p-2" role="alert">
    <i class="bi bi-exclamation-triangle-fill me-2"></i>Advertencia: El ratio ROA tiene un valor extremo, lo que sugiere un posible error en los datos de entrada.
</div>

#### c) Detalles del Análisis
Al hacer clic en el desplegable **"Detalles del Análisis"**, encontrarás una tabla con las métricas clave utilizadas por el modelo:

- **Métrica**: El nombre del indicador financiero (ej. `ROA`, `Debt_Ratio`).
- **Valor**: El valor calculado para esa métrica.
  - **Ratios (ej. ROA, Debt_Ratio)**: Se muestran como un número decimal (ej., `0.08`). La descripción te ayudará a interpretarlo como un porcentaje (ej., `un valor de 0.08 equivale a un 8%`).
  - **Valores Monetarios (ej. EBITDA, Ventas Netas)**: Se muestran como números con separadores de miles para facilitar la lectura.
- **Descripción**: Explica qué es la métrica, cómo interpretarla y cuál es un rango generalmente saludable. El color del valor (`verde`, `amarillo`, `rojo` o `gris`) te da una pista visual rápida sobre si el valor es favorable, riesgoso o neutro.

<br><br>

---

## 4. Guía para Desarrolladores

### Requisitos Previos
- **Docker Desktop** (Recomendado) o **Python 3.8+** y **pip**.

### Instalación y Ejecución con Docker (Recomendado)
1.  **Construye y levanta el contenedor:**
    ```bash
    docker-compose up --build
    ```
2.  **Accede a la aplicación:**
    Abre tu navegador y ve a: [http://localhost:5000](http://localhost:5000)

### Instalación y Ejecución Manual
1.  **Crea y activa un entorno virtual:**
    ```bash
    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```
    ```bash
    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
2.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ejecuta la aplicación:**
    ```bash
    python app.py
    ```
4.  **Accede a la aplicación:**
    Abre tu navegador y ve a: [http://localhost:5000](http://localhost:5000)
<br><br>

---

## 5. Estructura del Proyecto
```
.
├── Dockerfile                  # Define el entorno de la aplicación para Docker.
├── app.py                      # Lógica principal de la aplicación web (Flask).
├── bankruptcy_model.joblib     # Archivo del modelo de Machine Learning entrenado.
├── *.csv                       # Archivos de datos de ejemplo y el dataset completo.
├── docker-compose.yml          # Orquesta la construcción y ejecución del contenedor.
├── logistic_regression_pd_model.py # Script para entrenar y guardar el modelo.
├── requirements.txt            # Lista de dependencias de Python.
├── README.md                   # Este archivo.
├── templates/
│   ├── index.html              # Página de inicio con el formulario de subida.
│   └── results.html            # Página que muestra los resultados de la predicción.
└── uploads/                    # Carpeta para almacenar los CSV subidos (se crea autom.).
```
<br><br>

---

## 6. Modelo de Machine Learning

El núcleo del proyecto es un modelo de **Regresión Logística** entrenado con el dataset `american_bankruptcy_dataset.csv`. Este modelo fue elegido por su interpretabilidad y eficiencia.

- **Características (Features)**: El modelo no utiliza directamente las 18 variables `X`, sino un conjunto de características de ingeniería que incluyen los ratios **ROA** y **Debt Ratio**, junto con las variables financieras más relevantes.
- **Entrenamiento**: El script `logistic_regression_pd_model.py` contiene el código para pre-procesar los datos, realizar la ingeniería de características y entrenar el modelo, que finalmente se guarda en `bankruptcy_model.joblib` para ser utilizado por la aplicación Flask.
<br><br>

---

## 7. Tecnologías Utilizadas

- **Backend**: Python, Flask, Pandas, NumPy, Scikit-learn.
- **Frontend**: HTML5, CSS3, Bootstrap 5, Bootstrap Icons, AOS.
- **Despliegue**: Docker, Docker Compose.
<br><br>

---

## 8. Conclusiones

- Se desarrolló con éxito un modelo de Machine Learning capaz de predecir la quiebra empresarial, integrando de manera efectiva los principios de ingeniería económica en la selección y creación de variables predictivas.
- Se construyó una aplicación web funcional y estéticamente agradable que abstrae la complejidad del modelo. La interfaz ahora ofrece una herramienta de diagnóstico financiero accesible, con explicaciones detalladas, formato de valores claro y advertencias automáticas sobre datos de entrada anómalos.
- Se garantizó la portabilidad y reproducibilidad del proyecto mediante la containerización con Docker, lo que simplifica enormemente su despliegue.
<br><br>

---

## 9. Referencias

- **Texto Guía**: Blank, L., & Tarquin, A. (2012). *Ingeniería Económica* (7.a ed.). McGraw-Hill.
- **Dataset**: `futureinternet-14-00244-v2.pdf` - Documento que describe las variables del dataset original.

---
<div align="center">
    <img src="https://media.tenor.com/mG24i4G4qZkAAAAC/thumbs-up-computer.gif" width="300" alt="Todo listo">
</div>
<br>
