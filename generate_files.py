#!/usr/bin/env python3
import sys
import pandas as pd
import os

def main():
    if len(sys.argv) != 4:
        print("Uso: python make_runs_from_delta.py <T0.csv> <delta.csv> <prefijo_salida>")
        sys.exit(1)

    base_file, delta_file, out_prefix = sys.argv[1:]

    # Leer T0
    base = pd.read_csv(base_file)
    base_cols = list(base.columns)

    # Leer delta
    delta = pd.read_csv(delta_file)

    # Alinear columnas de delta a las de base
    for c in base_cols:
        if c not in delta.columns:
            delta[c] = pd.NA
    delta = delta[base_cols]  # descarta columnas extra

    # Generar exactamente 1 fichero por fila de delta, sin acumular
    n_runs = min(len(delta), 5)
    if n_runs == 0:
        print("El fichero delta no contiene filas")
        sys.exit(0)

    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for i in range(1, n_runs + 1):
        # solo la fila i del delta
        delta_i = delta.iloc[[i - 1]]
        df_out = pd.concat([base, delta_i], ignore_index=True)
        out_file = f"{out_prefix}_{i}.csv"
        df_out.to_csv(out_file, index=False)
        print(f"Guardado {out_file} -> base + fila {i} del delta")

    print(f"Generadas {n_runs} ejecuciones independientes")

if __name__ == "__main__":
    main()
