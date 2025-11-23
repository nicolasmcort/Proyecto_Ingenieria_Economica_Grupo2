import joblib

# Cargar el modelo
model = joblib.load('bankruptcy_model_v2.joblib')

print("=== FEATURES DEL MODELO ===")
print(model['feature_columns'])
print(f"\nTotal: {len(model['feature_columns'])} features")

if 'VPN' in model['feature_columns']:
    print(f"\n✅ VPN está en posición: {model['feature_columns'].index('VPN')}")
else:
    print("\n❌ VPN NO ENCONTRADO")
