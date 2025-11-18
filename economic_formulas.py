import numpy as np

# Este módulo implementa las fórmulas de factores discretos del Capítulo 2
# de Blank & Tarquin, 6ta Edición, y funciones para calcular métricas económicas.

# --- Factores Fundamentales (Capítulo 2) ---

def factor_P_dado_F(f, i, n):
    """ (P/F, i, n) - Calcula el Valor Presente de una cantidad Futura """
    if (1 + i) == 0: return 0
    return f / ((1 + i) ** n)

def factor_A_dado_P(p, i, n):
    """ (A/P, i, n) - Factor de Recuperación de Capital """
    if n == 0 or i == 0: return p / n if n != 0 else 0
    numerador = i * ((1 + i) ** n)
    denominador = ((1 + i) ** n) - 1
    if denominador == 0: return 0
    return p * (numerador / denominador)

def factor_P_dado_G(g, i, n):
    """ (P/G, i, n) - Valor Presente de un Gradiente Aritmético """
    if n == 0 or i == 0: return 0
    numerador = ((1 + i) ** n) - (i * n) - 1
    denominador = (i ** 2) * ((1 + i) ** n)
    if denominador == 0: return 0
    return g * (numerador / denominador)

def factor_P_dado_A1_g(a1, g, i, n):
    """ Valor Presente de un Gradiente Geométrico """
    if n == 0: return 0
    if abs(i - g) < 1e-9: # Si i es muy cercano a g
        return a1 * n / (1 + i)
    
    numerador = 1 - (((1 + g) / (1 + i)) ** n)
    denominador = i - g
    if denominador == 0: return 0
    return a1 * (numerador / denominador)

# --- Funciones para Métricas Económicas (Capítulos 5 y 6) ---

def calculate_npv_from_series(cash_flows, discount_rate):
    """ Calcula el VPN de una serie de flujos de caja usando el factor P/F """
    npv = 0.0
    for t, cf in enumerate(cash_flows, 1):
        npv += factor_P_dado_F(cf, discount_rate, t)
    return npv

def calculate_irr_from_series(cash_flows, initial_guess=0.1):
    """ Calcula la TIR de una serie de flujos de caja """
    # La TIR requiere al menos un flujo positivo y uno negativo
    if not any(cf > 0 for cf in cash_flows) or not any(cf < 0 for cf in cash_flows):
        return None # Retorna None si no se puede calcular
    
    try:
        # np.irr es un solver numérico eficiente para esta tarea
        return np.irr(cash_flows)
    except Exception:
        return None # Retorna None si el solver falla

def calculate_aw_from_pv(pv, i, n):
    """ Calcula el Valor Anual Equivalente (VA) desde un Valor Presente (VP) """
    return factor_A_dado_P(pv, i, n)

def calculate_arithmetic_gradient_g(cash_flows):
    """ Estima el gradiente aritmético promedio G de una serie """
    if len(cash_flows) < 2:
        return 0
    # diff() calcula la diferencia entre elementos consecutivos
    return np.mean(np.diff(cash_flows))

def calculate_geometric_gradient_g(cash_flows):
    """ Estima el gradiente geométrico promedio g de una serie """
    if len(cash_flows) < 2:
        return 0
    # pct_change() calcula el cambio porcentual
    # Se suma 1e-9 para evitar división por cero si un flujo es 0
    changes = [(y - x) / (x + 1e-9) for x, y in zip(cash_flows, cash_flows[1:]) if x != 0]
    return np.mean(changes) if changes else 0