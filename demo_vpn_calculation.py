"""
Script de demostración: Cálculo del VPN para el modelo de predicción de quiebra

Este script muestra cómo se calcula el VPN paso a paso y cómo se usa en el modelo.
"""

import pandas as pd
import numpy as np
from economic_formulas import calculate_irr_from_series, calculate_npv_from_series

print("=" * 80)
print("DEMOSTRACIÓN: CÁLCULO DEL VPN PARA EL MODELO")
print("=" * 80)

# Simular datos de una empresa para 3 años
print("\n1. DATOS DE ENTRADA (Ventana de 3 años)")
print("-" * 80)

# Ejemplo de datos financieros
ingresos = [100000, 120000, 150000]  # X16
gastos_operativos = [80000, 90000, 100000]  # X18

print(f"Año 1: Ingresos = ${ingresos[0]:,}, Gastos Operativos = ${gastos_operativos[0]:,}")
print(f"Año 2: Ingresos = ${ingresos[1]:,}, Gastos Operativos = ${gastos_operativos[1]:,}")
print(f"Año 3: Ingresos = ${ingresos[2]:,}, Gastos Operativos = ${gastos_operativos[2]:,}")

# Paso 1: Calcular flujo neto operativo
print("\n2. CÁLCULO DEL FLUJO NETO OPERATIVO")
print("-" * 80)
print("Fórmula: Flujo Neto = Ingresos Totales (X16) - Gastos Operativos (X18)")

flujo_neto = [ing - gasto for ing, gasto in zip(ingresos, gastos_operativos)]
print(f"\nAño 1: ${ingresos[0]:,} - ${gastos_operativos[0]:,} = ${flujo_neto[0]:,}")
print(f"Año 2: ${ingresos[1]:,} - ${gastos_operativos[1]:,} = ${flujo_neto[1]:,}")
print(f"Año 3: ${ingresos[2]:,} - ${gastos_operativos[2]:,} = ${flujo_neto[2]:,}")

# Paso 2: Calcular TIR
print("\n3. CÁLCULO DE LA TIR (Tasa Interna de Retorno)")
print("-" * 80)
print("Fórmula del Capítulo 6: TIR es la tasa donde VPN = 0")
print("0 = Σ [CFₜ / (1 + TIR)ᵗ]")

tir = calculate_irr_from_series(flujo_neto)
if tir is None:
    print("\n⚠️  No se pudo calcular TIR, usando tasa de respaldo: 5%")
    tir = 0.05
else:
    print(f"\n✅ TIR calculada: {tir*100:.2f}%")

# Paso 3: Calcular VPN
print("\n4. CÁLCULO DEL VPN (Valor Presente Neto)")
print("-" * 80)
print("Fórmula del Capítulo 5: VPN = Σ [CFₜ / (1 + i)ᵗ]")
print(f"Usando tasa de descuento i = {tir*100:.2f}%")

vpn = calculate_npv_from_series(flujo_neto, tir)

print("\nCálculo detallado:")
for t, cf in enumerate(flujo_neto, 1):
    vp = cf / ((1 + tir) ** t)
    print(f"Año {t}: ${cf:,} / (1 + {tir:.4f})^{t} = ${vp:,.2f}")

print(f"\n✅ VPN Total = ${vpn:,.2f}")

# Interpretación
print("\n5. INTERPRETACIÓN DEL VPN")
print("-" * 80)
if vpn > 0:
    print(f"✅ VPN POSITIVO (${vpn:,.2f})")
    print("   → Los flujos operativos generan valor")
    print("   → Indicador favorable para el modelo")
else:
    print(f"❌ VPN NEGATIVO (${vpn:,.2f})")
    print("   → Los flujos operativos destruyen valor")
    print("   → Señal de alerta para el modelo")

# Mostrar cómo se usa en el modelo
print("\n6. USO EN EL MODELO DE MACHINE LEARNING")
print("-" * 80)
print("El VPN calculado se usa como una de las 18 features del modelo:")
print("\nfeature_columns = [")
print("    'X1',          # Activos Corrientes")
print("    'ROA',         # Return on Assets (calculado)")
print("    'Debt_Ratio',  # Ratio de Endeudamiento (calculado)")
print("    'VPN',         # ⭐ Valor Presente Neto (calculado) ← AQUÍ")
print("    'X2', 'X3', 'X4', 'X5', 'X7', 'X8', 'X9',")
print("    'X11', 'X12', 'X13', 'X14', 'X15', 'X16', 'X18'")
print("]")

print("\n" + "=" * 80)
print("CONCLUSIÓN")
print("=" * 80)
print("✅ El modelo SÍ se alimenta con el VPN")
print("✅ El interés (TIR) SÍ se calcula para obtener el VPN")
print("✅ El VPN es la feature #4 de 18 en el modelo")
print("=" * 80)
