import numpy as np

# Este módulo implementa las fórmulas de factores discretos del Capítulo 2
# de Blank & Tarquin, 6ta Edición, y funciones para calcular métricas económicas.

# --- Factores Fundamentales (Capítulo 2) ---

def factor_P_dado_F(f, i, n):
    """ (P/F, i, n) - Calcula el Valor Presente de una cantidad Futura """
    if (1 + i) == 0:
        return 0
    return f / ((1 + i) ** n)

def factor_A_dado_P(p, i, n):
    """ (A/P, i, n) - Factor de Recuperación de Capital """
    if n == 0 or i == 0:
        return p / n if n != 0 else 0
    numerador = i * ((1 + i) ** n)
    denominador = ((1 + i) ** n) - 1
    if denominador == 0:
        return 0
    return p * (numerador / denominador)

# --- Funciones para Métricas Económicas (Capítulos 5 y 6) ---

def calculate_npv_from_series(cash_flows, discount_rate):
    """Calcula el Valor Presente Neto (VPN) de una serie de flujos de caja.
    
    Args:
        cash_flows: Lista de flujos de caja.
        discount_rate: Tasa de descuento (decimal).
    Returns:
        VPN (float).
    """
    npv = 0.0
    for t, cf in enumerate(cash_flows, 1):
        npv += factor_P_dado_F(cf, discount_rate, t)
    return npv

def calculate_aw_from_pv(pv, i, n):
    """Calcula el Valor Anual Equivalente (VA) desde un Valor Presente (VP)."""
    return factor_A_dado_P(pv, i, n)

def calculate_discounted_payback_period(investment, cash_flows, i, max_years=50):
    """Calcula el Periodo de Recuperación Descontado (Discounted Payback Period).
    
    Estima el tiempo necesario para recuperar la inversión inicial considerando el valor del dinero en el tiempo.
    Si no se recupera dentro de los flujos dados, proyecta usando el promedio de los flujos.
    
    Args:
        investment: Inversión inicial (positiva).
        cash_flows: Lista de flujos de caja netos.
        i: Tasa de descuento (decimal).
        max_years: Años máximos para proyectar si no se recupera en el periodo dado.
        
    Returns:
        float: Número de años (interpolado) o float('inf') si no se recupera.
    """
    cumulative_pvs = 0.0
    
    # 1. Verificar recuperación dentro de los flujos proporcionados
    for t, cf in enumerate(cash_flows, 1):
        pv = factor_P_dado_F(cf, i, t)
        cumulative_pvs += pv
        if cumulative_pvs >= investment:
            # Interpolación lineal para mayor precisión
            prev_cum = cumulative_pvs - pv
            fraction = (investment - prev_cum) / pv
            return t - 1 + fraction

    # 2. Si no se recuperó, proyectar con el flujo promedio
    if cumulative_pvs < investment:
        avg_cf = np.mean(cash_flows)
        if avg_cf <= 0:
            return float('inf') # Nunca recupera si el flujo promedio es <= 0
        
        current_t = len(cash_flows)
        while current_t < max_years:
            current_t += 1
            pv = factor_P_dado_F(avg_cf, i, current_t)
            cumulative_pvs += pv
            if cumulative_pvs >= investment:
                 prev_cum = cumulative_pvs - pv
                 fraction = (investment - prev_cum) / pv
                 return current_t - 1 + fraction
                 
    return float('inf')
