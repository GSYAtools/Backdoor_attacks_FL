#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, energy_distance

EPS = 1e-12

# ----------------------------- Utilidades -----------------------------
def select_columns(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    # Solo 'l2' y 'proj*'
    return [c for c in num if (c == "l2") or c.startswith("proj")]

def get_proj_cols(df):
    return [c for c in df.columns if c.startswith("proj")]

def resolve_n_add(n0, n_add=None, pstar=None):
    if n_add is not None:
        return int(n_add), None
    if pstar is not None:
        if not (0.0 < pstar < 0.9):
            raise ValueError("p* debe estar en (0, 0.9)")
        return int(round(pstar / (1.0 - pstar) * n0)), float(pstar)
    # fallback p*=0.20
    return int(round((0.20 / (1.0 - 0.20)) * n0)), 0.20

def orth_debias_projections_inplace(df, proj_cols):
    """
    Estima la dirección dominante 'v' como la media en el subespacio de proyecciones
    y proyecta cada fila a la componente ortogonal: y_perp = y - (y·v)/(v·v) * v.
    Modifica in-place las columnas 'proj*'. Devuelve el vector 'v' (lista) y un flag 'applied'.
    """
    if not proj_cols:
        return None, False

    Y = df[proj_cols].to_numpy(dtype=float, copy=True)  # shape (n, K)
    mu = Y.mean(axis=0)  # dirección benigna media
    norm = np.linalg.norm(mu)
    if norm < 1e-10:  # sin dirección clara
        return None, False

    v = mu / (norm + EPS)  # normalizado
    # Proyección ortogonal fila a fila
    # alpha = (Y @ v)  (porque v está normalizado, v·v ≈ 1)
    alpha = Y.dot(v)
    Y_perp = Y - np.outer(alpha, v)

    # Escribe de vuelta
    df.loc[:, proj_cols] = Y_perp
    return v.tolist(), True

def maybe_replace_l2_with_l2perp_inplace(df, proj_cols):
    """
    Reemplaza 'l2' por la norma en el subespacio ortogonal estimado (ya aplicado).
    Supone que 'proj*' ya están des-sesgadas. Si no hay 'proj*', no hace nada.
    """
    if ("l2" in df.columns) and proj_cols:
        Yp = df[proj_cols].to_numpy(dtype=float, copy=False)
        l2_perp = np.sqrt((Yp ** 2).sum(axis=1))
        df.loc[:, "l2"] = l2_perp
        return True
    return False

def calibrate_column(values, B, n0, n_add, rng, pctl):
    e_vals, w_vals = [], []
    n = len(values)
    if n < max(n0, 1):
        return None, None
    for _ in range(B):
        # baseline A (n0 con reemplazo)
        idx_A = rng.integers(0, n, size=n0)
        A = values[idx_A]
        # muestra "actual" (k=1) y réplica hasta n_add
        idx_new = rng.integers(0, n, size=1)
        new_val = values[idx_new]
        new_eff = np.repeat(new_val, n_add)
        Bwin = np.concatenate([A, new_eff])

        e_vals.append(energy_distance(A, Bwin))
        w_vals.append(wasserstein_distance(A, Bwin))

    tau_e = float(np.percentile(e_vals, pctl))
    tau_w = float(np.percentile(w_vals, pctl))
    return tau_e, tau_w

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Calibración bootstrap Energy+W1 con opción de des-sesgar proyecciones por proyección ortogonal. Salida en /results"
    )
    ap.add_argument("csv_file", help="CSV con P0 (baseline benigno)")
    ap.add_argument("--B", type=int, default=500, help="Repeticiones bootstrap (default 500)")
    ap.add_argument("--n0", type=int, default=50, help="Tamaño baseline por réplica (default 50)")
    ap.add_argument("--n-add", type=int, default=None, dest="n_add",
                    help="Tamaño efectivo de la muestra actual (réplica). Si se omite, se usa p*=0.20")
    ap.add_argument("--pstar", type=float, default=None,
                    help="Proporción efectiva p*; n_add = p*/(1-p*) * n0")
    ap.add_argument("--pctl", type=float, default=95.0, help="Percentil del umbral (default 95)")
    ap.add_argument("--debias-proj", action="store_true",
                    help="Quita la componente benigna dominante en el espacio de proyecciones (idea 'notch').")
    ap.add_argument("--use-l2-perp", action="store_true",
                    help="Sustituye l2 por la norma en el subespacio ortogonal (requiere --debias-proj).")
    ap.add_argument("--out-name", default=None,
                    help="Nombre de salida (opcional). Si no se da, se genera automáticamente.")
    args = ap.parse_args()

    # Semilla fija y reproducible
    seed = 42
    rng = np.random.default_rng(seed)

    # Cargar datos
    df = pd.read_csv(args.csv_file)
    cols = select_columns(df)
    if not cols:
        raise ValueError("No se encontraron columnas 'l2' o 'proj*' en el CSV.")

    proj_cols = get_proj_cols(df)

    # Des-sesgo por proyección ortogonal (si se solicita)
    debias_info = None
    if args.debias_proj:
        v, applied = orth_debias_projections_inplace(df, proj_cols)
        if applied and args.use_l2_perp:
            _ = maybe_replace_l2_with_l2perp_inplace(df, proj_cols)
        debias_info = {
            "applied": bool(applied),
            "type": "proj_mean_orth",
            "proj_cols": proj_cols,
            "v": v if applied else None,
            "eps": EPS,
            "use_l2_perp": bool(args.use_l2_perp and applied)
        }

    # Resolver n_add
    n_add, pstar_used = resolve_n_add(args.n0, args.n_add, args.pstar)
    pstar_equiv = n_add / (args.n0 + n_add)

    # Calibración por columna (sobre los datos ya transformados si procede)
    results, skipped = {}, []
    for col in cols:
        vals = df[col].dropna().values
        tau_e, tau_w = calibrate_column(vals, B=args.B, n0=args.n0, n_add=n_add, rng=rng, pctl=args.pctl)
        if tau_e is None:
            skipped.append(col)
            continue
        results[col] = {"tau_energy": tau_e, "tau_w1": tau_w}

    # Meta
    results["meta"] = {
        "n0": int(args.n0),
        "n_add": int(n_add),
        "pstar": float(pstar_equiv),
        "B": int(args.B),
        "pctl": float(args.pctl),
        "seed": int(seed)
    }
    if debias_info is not None:
        results["meta"]["debias"] = debias_info

    # Carpeta y nombre de salida
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.csv_file))[0]

    suffix = f"nadd{n_add}_pctl{int(args.pctl)}"
    if args.debias_proj:
        suffix += "_debias"
        if debias_info and debias_info.get("use_l2_perp"):
            suffix += "_l2perp"

    out_path = os.path.join(out_dir, args.out_name) if args.out_name \
        else os.path.join(out_dir, f"thresholds_{base}_{suffix}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("------------------------------------------------")
    print(f"Calibración completada  B={args.B}  n0={args.n0}  n_add={n_add}  pctl={args.pctl}")
    if debias_info is not None:
        print(f"Des-sesgo proyecciones: {debias_info}")
    print(f"meta: {results['meta']}")
    if skipped:
        print(f"Omitidas por falta de datos: {skipped}")
    print(f"Umbrales guardados en: {out_path}")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()
