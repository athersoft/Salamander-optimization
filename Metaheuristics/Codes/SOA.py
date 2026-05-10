import numpy as np

def iterarSOA(maxIter, iter, dim, population, fitness, best, fo=None, lb=None, ub=None, lb0=None, ub0=None, objective_type='MIN', **kwargs):
    """
    Salamander Optimization Algorithm (SOA)
    Implementación estrictamente basada en el paper oficial.
    """
    N = population.shape[0]
    X = np.copy(population) # Población actual (X)
    new_fitness = np.copy(fitness) if fitness is not None else np.zeros(N)
    
    # Contadores de iteración: t (actual) y T (máximo) según el paper 
    t = iter if iter > 0 else 1
    T = maxIter if maxIter > 0 else 100
    
    # ---------------------------------------------------
    # Extraer la mejor solución global (X_best) 
    # ---------------------------------------------------
    if isinstance(best, tuple) or (isinstance(best, list) and len(best) == 2 and not isinstance(best[0], (int, float))):
        X_best = np.array(best[0], dtype=float)
    elif isinstance(best, np.ndarray) and best.dtype == object:
        X_best = np.array(best[0], dtype=float)
    else:
        X_best = np.array(best, dtype=float)
        
    if X_best.ndim > 1:
        X_best = X_best.flatten()
    X_best = X_best[:dim]
    
    # ---------------------------------------------------
    # Límites del problema (lb y ub) [cite: 134]
    # ---------------------------------------------------
    lower = lb if lb is not None else (lb0 if lb0 is not None else -100.0)
    upper = ub if ub is not None else (ub0 if ub0 is not None else 100.0)
    
    LB = np.asarray(lower) * np.ones(dim)
    UB = np.asarray(upper) * np.ones(dim)

    # Bucle principal para cada individuo i en la población
    for i in range(N):
        # ===================================================
        # Phase 1: Regeneration (Exploration phase)
        # ===================================================
        # Según el paper, I es 1 o 2
        I = np.random.choice([1, 2])
        
        # r es un número aleatorio entre 0 y 1
        r = np.random.rand(dim) 
        
        # Ecuación (4) corregida visualmente del paper:
        # X_i^(P1) = X_i + (r + (T-t)/T) * (X_best - I * X_i)
        X_P1 = X[i] + (r + ((T - t) / T)) * (X_best - I * X[i])
        
        X_P1 = np.clip(X_P1, LB, UB)
        
        # Ecuación (5) del paper: Actualización Greedy [cite: 161]
        if fo is not None:
            val_P1 = fo(X_P1)
            raw_fit_P1 = val_P1[0] if isinstance(val_P1, tuple) else val_P1
            fit_P1 = float(raw_fit_P1.flatten()[0]) if isinstance(raw_fit_P1, np.ndarray) else float(raw_fit_P1)
            current_fit = float(new_fitness[i]) if isinstance(new_fitness[i], np.ndarray) else new_fitness[i]
            
            # F_i^(P1) <= F_i [cite: 160]
            if (fit_P1 <= current_fit) if objective_type == 'MIN' else (fit_P1 >= current_fit):
                X[i] = X_P1
                new_fitness[i] = fit_P1
        else:
            X[i] = X_P1
            
        # ===================================================
        # Phase 2: Adaptive movement (Exploitation phase)
        # ===================================================
        # Sorteamos un nuevo factor I (1 o 2) para el movimiento local
        I2 = np.random.choice([1, 2])
        
        # r es un número aleatorio entre 0 y 1
        r2 = np.random.rand(dim)
        
        # Ecuación (6) visual real: X_i^(P2) = X_i + (1 - 2r) * I * ((ub - lb) / t)
        X_P2 = X[i] + (1 - 2 * r2) * I2 * ((UB - LB) / t)
        
        X_P2 = np.clip(X_P2, LB, UB)
        
        # Ecuación (7) del paper: Actualización Greedy [cite: 175]
        if fo is not None:
            val_P2 = fo(X_P2)
            raw_fit_P2 = val_P2[0] if isinstance(val_P2, tuple) else val_P2
            fit_P2 = float(raw_fit_P2.flatten()[0]) if isinstance(raw_fit_P2, np.ndarray) else float(raw_fit_P2)
            current_fit = float(new_fitness[i]) if isinstance(new_fitness[i], np.ndarray) else new_fitness[i]
            
            # F_i^(P2) <= F_i [cite: 172]
            if (fit_P2 <= current_fit) if objective_type == 'MIN' else (fit_P2 >= current_fit):
                X[i] = X_P2
                new_fitness[i] = fit_P2
        else:
            if np.random.rand() < 0.5:
                X[i] = X_P2

    return X