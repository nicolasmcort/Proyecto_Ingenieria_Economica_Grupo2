# 📉 Análisis de Riesgo de Quiebra Corporativa

<div align="center">
  <img src="https://www.pngkey.com/png/detail/268-2688228_universidad-nacional-colombia-logo.png" width="200" alt="Logo Universidad Nacional de Colombia">
  <br><br>
  
  ![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
  ![Flask](https://img.shields.io/badge/Flask-2.0+-green?style=for-the-badge&logo=flask&logoColor=white)
  ![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=for-the-badge&logo=docker&logoColor=white)
  ![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
  ![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?style=for-the-badge&logo=bootstrap&logoColor=white)

  <h3>UNIVERSIDAD NACIONAL DE COLOMBIA</h3>
  <p><strong>Ingeniería Económica - 2015703</strong></p>
  <p><strong>2025-2</strong></p>
</div>

---

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

---

## 📋 Descripción del Proyecto

Este proyecto es una herramienta avanzada de **predicción de quiebra corporativa** que fusiona el poder del **Machine Learning** con los principios fundamentales de la **Ingeniería Económica**.

El objetivo principal es proporcionar a analistas financieros, inversores y gerentes una plataforma web intuitiva para evaluar la salud financiera de una empresa. A partir de datos históricos (estados financieros), el sistema calcula una probabilidad de quiebra y ofrece un diagnóstico detallado, complementado con métricas de valor y tiempo de recuperación.

### 🌟 Características Clave
*   **Predicción Inteligente**: Utiliza un modelo de Regresión Logística entrenado con miles de registros históricos.
*   **Análisis Financiero Profundo**: Calcula y visualiza ratios clave como ROA y Debt Ratio.
*   **Ingeniería Económica**: Integra cálculos de Valor Presente Neto (VPN), Valor Anual (VA) y Periodo de Recuperación Descontado.
*   **Interfaz Moderna**: Dashboard web limpio y responsivo para una fácil interpretación de los datos.
*   **Alertas Automáticas**: Detecta anomalías en los datos de entrada que podrían afectar la fiabilidad del análisis.

---

## 💰 Principios de Ingeniería Económica

Este proyecto no solo predice riesgos, sino que evalúa la viabilidad económica de la empresa utilizando conceptos del texto guía *Ingeniería Económica* de Blank & Tarquin.

### 1. Valor Presente Neto (VPN)
El **VPN** trae todos los flujos de caja futuros (ingresos menos egresos) al presente, descontados a una tasa de interés de oportunidad (TIO).
*   **Fórmula**: $VPN = \sum_{t=1}^{n} \frac{F_t}{(1+i)^t}$
*   **Interpretación**: Un $VPN > 0$ indica que la empresa está generando valor por encima de su costo de oportunidad.

### 2. Valor Anual Equivalente (VA)
El **VA** convierte el VPN en una serie uniforme anual equivalente. Esto es crucial para comparar empresas de diferentes tamaños o vidas útiles.
*   **Fórmula**: $VA = VPN(A/P, i, n)$
*   **Aplicación**: Nos permite decir "esta empresa genera X cantidad de valor *por año*", facilitando la comparación directa.

### 3. Periodo de Recuperación Descontado (Payback Period)
Calcula cuánto tiempo tarda la empresa en recuperar su inversión inicial (considerada aquí como los Activos Totales) utilizando los flujos de caja descontados.
*   **Importancia**: Mide la liquidez y el riesgo. Un periodo más corto significa una recuperación más rápida y menor exposición al riesgo.
*   **Visualización**: Si la inversión no se recupera en el horizonte de tiempo, el sistema proyecta los flujos o indica "> 50 años".

---

## 🚀 Guía de Instalación y Uso

### Opción A: Docker (Recomendada) 🐳
La forma más fácil de ejecutar la aplicación sin preocuparse por dependencias.

1.  **Clonar el repositorio:**
    ```bash
    git clone https://github.com/nicolasmcort/Proyecto_Ingenieria_Economica_Grupo2.git
    cd Proyecto_Ingenieria_Economica_Grupo2
    ```
2.  **Construir y correr:**
    ```bash
    docker-compose up --build
    ```
3.  **Abrir en el navegador:**
    Visita [http://localhost:5000](http://localhost:5000)

### Opción B: Ejecución Manual 🐍

1.  **Crear entorno virtual:**
    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # Mac/Linux: source venv/bin/activate
    ```
2.  **Instalar dependencias:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Ejecutar la aplicación:**
    ```bash
    python app.py
    ```

---

## 📖 Cómo Usar la Herramienta

1.  **Prepara tus Datos**: Necesitas un archivo `.csv` con columnas como `company_name`, `fyear`, y las variables financieras `X1` a `X18`. (Ver carpeta `data/` para ejemplos).
2.  **Carga el Archivo**: En la página de inicio, selecciona tu archivo CSV.
3.  **Analiza**: Haz clic en "Analizar y Predecir".
4.  **Interpreta**:
    *   🟢 **Bajo Riesgo**: La empresa es sólida.
    *   🟡 **Riesgo Moderado**: Precaución, revisar indicadores.
    *   🔴 **Alto Riesgo**: Alta probabilidad de insolvencia.
5.  **Explora**: Despliega los detalles para ver el **ROA**, **Debt Ratio**, **VA** y el **Payback Period**.

---

## 📂 Estructura del Proyecto

```
📦 Proyecto_Ingenieria_Economica
 ┣ 📂 data/                     # Datasets de ejemplo y pruebas
 ┣ 📂 models/                   # Modelos entrenados (.joblib) y scripts de entrenamiento
 ┣ 📂 static/                   # Assets (CSS, JS, imágenes)
 ┣ 📂 templates/                # Plantillas HTML (Flask)
 ┣ 📂 utils/                    # Módulos de utilidades (fórmulas económicas)
 ┣ 📜 app.py                    # Aplicación principal Flask
 ┣ 📜 Dockerfile                # Configuración de Docker
 ┣ 📜 docker-compose.yml        # Orquestación de servicios
 ┣ 📜 requirements.txt          # Dependencias del proyecto
 ┗ 📜 README.md                 # Documentación del proyecto
```

---

<div align="center">
    <p>Hecho con ❤️ y ☕ por estudiantes de la UNAL</p>
</div>
