#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, energy_distance

# -------------------------
# Umbrales y utilidades
# -------------------------
def load_thresholds(thresholds_path):
    with open(thresholds_path, "r") as f:
        th = json.load(f)
    meta = th.get("meta", {}) if isinstance(th, dict) else {}
    return th, meta

def resolve_n0_nadd(meta):
    n0 = int(meta.get("n0", 50))
    if "n_add" in meta:
        n_add = int(meta["n_add"])
    else:
        # fallback p*=0.20
        n_add = int(round((0.20 / (1.0 - 0.20)) * n0))
        print(f"  thresholds sin meta.n_add: usando fallback n0={n0}, n_add={n_add} (p*=0.20)")
    return n0, n_add

def feature_order(thresholds_dict):
    feats = [k for k in thresholds_dict.keys() if k != "meta"]
    l2 = ["l2"] if "l2" in feats else []
    projs = sorted([c for c in feats if c.startswith("proj")],
                   key=lambda s: int(s[4:]) if s[4:].isdigit() else 10**9)
    others = [c for c in feats if c not in set(l2 + projs)]
    return l2 + projs + sorted(others)

def effective_new_block(new_vals, n_add):
    """Replica las k nuevas hasta alcanzar n_add."""
    k = len(new_vals)
    if k <= 0 or n_add <= 0:
        return np.array([], dtype=float)
    reps = int(np.ceil(n_add / k))
    tiled = np.tile(new_vals, reps)
    return tiled[:n_add]

def compute_energy_w1(P0_vals, new_vals, n_add):
    """Compara P0 vs (P0 + nuevas_replicadas)."""
    new_eff = effective_new_block(new_vals, n_add=n_add)
    B = np.concatenate([P0_vals, new_eff])
    e = energy_distance(P0_vals, B)
    w = wasserstein_distance(P0_vals, B)
    return float(e), float(w), int(len(new_vals)), int(len(new_eff))

# -------------------------
# De-drift por ronda (proyecciones)
# -------------------------
def load_drift_row(prefix, i):
    """
    Lee un CSV con una sola fila que contenga columnas proj* (y opcionalmente otras).
    Ej: drift/t1_drift_1.csv
    """
    if not prefix:
        return None
    path = f"{prefix}_{i}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if len(df) < 1:
        return None
    return df.iloc[0]  # devuelve una Series

def apply_projection_dedrift_to_row(row_series, drift_row):
    """
    Resta drift en columnas proj* de la ÚLTIMA fila (la “nueva”).
    No toca l2. Devuelve una copia modificada de la Series.
    """
    if drift_row is None:
        return row_series
    row = row_series.copy()
    for c in row.index:
        if c.startswith("proj") and (c in drift_row.index):
            try:
                row[c] = float(row[c]) - float(drift_row[c])
            except Exception:
                pass
    return row

# -------------------------
# Núcleo comparación
# -------------------------
def compare_file_tx(tx_path, thresholds, n_add, drift_row=None, rule="any"):
    """
    tx_path: contiene P0 en todas las filas menos la última.
    La última fila es la nueva actualización a ponderar (k=1).
    Aplica de-drift de proyecciones SOLO en la última fila si hay drift_row.
    rule: 'any' (OR) o 'both' (AND) para disparar alerta.
    """
    df = pd.read_csv(tx_path)
    if len(df) < 2:
        raise ValueError(f"{os.path.basename(tx_path)} debe tener al menos 2 filas (P0 + 1 nueva).")

    feats = feature_order(thresholds)

    df_P0 = df.iloc[:-1, :].copy()
    new_row = df.iloc[-1, :]

    # de-drift en la nueva observación (proyecciones)
    new_row = apply_projection_dedrift_to_row(new_row, drift_row)
    df_new = pd.DataFrame([new_row])

    results = {}
    for col in feats:
        if col not in df.columns:
            print(f"  Columna {col} no encontrada en {os.path.basename(tx_path)}, se omite.")
            continue

        P0_vals = df_P0[col].dropna().values
        new_vals = df_new[col].dropna().values  # normalmente tamaño 1
        if len(P0_vals) == 0 or len(new_vals) == 0:
            print(f"  Columna {col} sin datos tras dropna(), se omite.")
            continue

        # Umbrales esperados: tau_energy y tau_w1
        if "tau_energy" not in thresholds.get(col, {}) or "tau_w1" not in thresholds.get(col, {}):
            raise ValueError(f"El thresholds JSON no contiene tau_energy/tau_w1 para '{col}'. Recalibra con Energy+W1.")

        energy, w1, k_obs, n_eff = compute_energy_w1(P0_vals, new_vals, n_add=n_add)
        tau_energy = float(thresholds[col]["tau_energy"])
        tau_w1     = float(thresholds[col]["tau_w1"])

        if rule == "both":
            alert = (energy > tau_energy) and (w1 > tau_w1)
        else:
            alert = (energy > tau_energy) or  (w1 > tau_w1)

        results[col] = {
            "k_observed": k_obs,               # debería ser 1
            "n_add_effective": n_eff,          # ≈ n_add
            "energy": energy, "tau_energy": tau_energy,
            "w1": w1, "tau_w1": tau_w1,
            "alert": alert
        }

    return results

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Comparar data/<case>_1.._5.csv con Energy+W1. "
                    "P0 = todas las filas menos la última; nueva = última con peso n_add. "
                    "Opcionalmente resta drift de proyecciones en la última fila."
    )
    ap.add_argument("case", choices=["T1", "T2", "T3"], help="Caso a comparar")
    ap.add_argument("thresholds_json", help="JSON con umbrales calibrados (Energy + W1)")
    ap.add_argument("--data-dir", default="data", help="Carpeta de CSVs (default: data)")
    ap.add_argument("--outdir", default="results", help="Carpeta de salida (default: results)")
    ap.add_argument("--drift-prefix", default=None,
                    help="Prefijo de ficheros de drift por ronda (ej: drift/t1_drift genera drift/t1_drift_1.csv ... _5.csv)")
    ap.add_argument("--rule", choices=["any","both"], default="any",
                    help="Regla de disparo: any=OR (por defecto), both=AND")
    args = ap.parse_args()

    thresholds, meta = load_thresholds(args.thresholds_json)
    n0, n_add = resolve_n0_nadd(meta)

    os.makedirs(args.outdir, exist_ok=True)
    case_lower = args.case.lower()
    combined = {
        "case": args.case,
        "thresholds_meta": meta,
        "n0_used": n0,
        "n_add_used": n_add,
        "rule": args.rule,
        "runs": {}
    }

    found_any = False
    for i in range(1, 6):
        tx_path = os.path.join(args.data_dir, f"{case_lower}_{i}.csv")
        if not os.path.exists(tx_path):
            print(f"* Aviso: no existe {tx_path}, se omite.")
            continue

        drift_row = load_drift_row(args.drift_prefix, i) if args.drift_prefix else None
        print(f"== {os.path.basename(tx_path)} | drift={'sí' if drift_row is not None else 'no'} ==")

        run_results = compare_file_tx(tx_path, thresholds, n_add=n_add, drift_row=drift_row, rule=args.rule)
        combined["runs"][str(i)] = {
            "file": tx_path,
            "drift_file": f"{args.drift_prefix}_{i}.csv" if (args.drift_prefix and drift_row is not None) else None,
            "results": run_results
        }
        found_any = True

    if not found_any:
        raise FileNotFoundError("No se encontró ningún fichero con sufijos _1.csv ... _5.csv en la carpeta de datos.")

    out_path = os.path.join(args.outdir, f"results_{args.case}.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"→ Resultados combinados guardados en {out_path}")

if __name__ == "__main__":
    main()
