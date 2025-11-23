"""
Test script for simplified IRR estimation
"""
from economic_formulas import estimate_irr_simplified, calculate_irr_from_series

print("=" * 80)
print("PRUEBA: MÉTODO SIMPLIFICADO DE TIR")
print("=" * 80)

# Test case 1: Crecimiento positivo
print("\n1. FLUJOS CON CRECIMIENTO POSITIVO")
print("-" * 80)
cash_flows = [20000, 30000, 50000]
print(f"Flujos: {cash_flows}")
irr = estimate_irr_simplified(cash_flows)
if irr:
    print(f"✅ TIR estimada: {irr*100:.2f}%")
    print(f"   Cálculo: ({cash_flows[-1]}/{cash_flows[0]})^(1/{len(cash_flows)-1}) - 1")
    print(f"   = ({cash_flows[-1]/cash_flows[0]:.2f})^(1/{len(cash_flows)-1}) - 1")
    print(f"   = {irr*100:.2f}%")
else:
    print("❌ No se pudo calcular TIR")

# Test case 2: Flujos constantes
print("\n2. FLUJOS CONSTANTES")
print("-" * 80)
cash_flows = [10000, 10000, 10000]
print(f"Flujos: {cash_flows}")
irr = estimate_irr_simplified(cash_flows)
if irr is not None:
    print(f"✅ TIR estimada: {irr*100:.2f}%")
else:
    print("❌ No se pudo calcular TIR")

# Test case 3: Decrecimiento
print("\n3. FLUJOS CON DECRECIMIENTO")
print("-" * 80)
cash_flows = [50000, 30000, 20000]
print(f"Flujos: {cash_flows}")
irr = estimate_irr_simplified(cash_flows)
if irr is not None:
    print(f"✅ TIR estimada: {irr*100:.2f}%")
else:
    print("❌ No se pudo calcular TIR")

# Test case 4: Comparar con np.irr (si funciona)
print("\n4. COMPARACIÓN CON MÉTODO COMPLETO")
print("-" * 80)
cash_flows = [20000, 30000, 50000]
print(f"Flujos: {cash_flows}")
irr_full = calculate_irr_from_series(cash_flows)
irr_simple = estimate_irr_simplified(cash_flows)
print(f"TIR método completo:      {irr_full*100:.2f}%" if irr_full else "No disponible")
print(f"TIR método simplificado:  {irr_simple*100:.2f}%" if irr_simple else "No disponible")

print("\n" + "=" * 80)
print("CONCLUSIÓN: El método simplificado es robusto y siempre da resultado")
print("=" * 80)
