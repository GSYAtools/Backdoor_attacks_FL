#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, re
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Orden de features: l2, proj0..projN, resto
# ---------------------------
def feature_sort_key(name: str):
    if name.lower() == "l2":
        return (0, 0)
    m = re.fullmatch(r"(proj)(\d+)", name)
    if m:
        return (1, int(m.group(2)))
    return (2, name)

def is_feature_entry(obj) -> bool:
    """Entrada válida de feature: tiene energy, w1, tau_energy, tau_w1."""
    return (
        isinstance(obj, dict)
        and "energy" in obj and "w1" in obj
        and "tau_energy" in obj and "tau_w1" in obj
    )

# ---------------------------
# Carga y organización
# ---------------------------
def load_combined(path):
    with open(path, "r") as f:
        data = json.load(f)
    if "runs" not in data or not isinstance(data["runs"], dict):
        raise ValueError("El JSON no parece ser el combinado (falta 'runs').")

    # IDs de runs
    run_ids = sorted((int(k) for k in data["runs"].keys()))
    if not run_ids:
        raise ValueError("No hay runs en 'runs'.")

    # Determinar features válidos a partir de la primera run disponible
    first_run = data["runs"][str(run_ids[0])]["results"]
    feats = [k for k, v in first_run.items() if is_feature_entry(v)]
    feats = sorted(feats, key=feature_sort_key)
    if not feats:
        raise ValueError("No se encontraron features válidos (energy/w1/taus).")

    # Taus (umbrales por feature, tomados de la primera run que los tenga)
    tau_energy, tau_w1 = [], []
    for feat in feats:
        found_energy, found_w1 = None, None
        for rid in run_ids:
            rfeat = data["runs"][str(rid)]["results"].get(feat)
            if rfeat:
                if found_energy is None and "tau_energy" in rfeat:
                    found_energy = float(rfeat["tau_energy"])
                if found_w1 is None and "tau_w1" in rfeat:
                    found_w1 = float(rfeat["tau_w1"])
            if found_energy is not None and found_w1 is not None:
                break
        tau_energy.append(found_energy if found_energy is not None else np.nan)
        tau_w1.append(found_w1 if found_w1 is not None else np.nan)

    # Series por run
    energy_by_run, w1_by_run = {}, {}
    for rid in run_ids:
        rname = str(rid)
        r = data["runs"][rname]["results"]
        e_vals, w_vals = [], []
        for feat in feats:
            rf = r.get(feat, {})
            if not is_feature_entry(rf):
                raise ValueError(f"Faltan 'energy/w1/taus' para '{feat}' en run {rname}.")
            e_vals.append(float(rf["energy"]))
            w_vals.append(float(rf["w1"]))
        energy_by_run[rname] = e_vals
        w1_by_run[rname] = w_vals

    case = data.get("case", os.path.splitext(os.path.basename(path))[0])
    return {
        "case": case,
        "features": feats,
        "run_ids": [str(k) for k in run_ids],
        "energy_by_run": energy_by_run,
        "w1_by_run": w1_by_run,
        "tau_energy": tau_energy,
        "tau_w1": tau_w1
    }

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

# ---------------------------
# Plot: líneas + umbral negro grueso
# ---------------------------
def plot_lines(features, series_by_run, taus, ylabel, out_path, title=None):
    ensure_dir(os.path.dirname(out_path))
    x = np.arange(len(features))

    # Ajustes globales de tamaño de fuente
    plt.rcParams.update({
        "font.size": 20,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18
    })

    fig, ax = plt.subplots(figsize=(14, 6))

    # Curvas por run
    for run_id, vals in series_by_run.items():
        ax.plot(x, vals, marker="o", linewidth=2.3, label=f"run {run_id}", zorder=2)

    # Umbral: línea negra gruesa
    ax.plot(x, taus, color="black", linewidth=4.0, label="Threshold", zorder=5)

    # Ejes y límites
    vmax_series = max(max(v) for v in series_by_run.values()) if series_by_run else 0.0
    ymax = max(vmax_series, max(taus) if taus else 0.0)
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35, zorder=1)
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Pinta 5 líneas (runs) y el umbral negro grueso a partir de results/results_Tx.json."
    )
    ap.add_argument("combined_json", help="Ruta a results/results_Tx.json")
    ap.add_argument("--outdir", default="results", help="Carpeta de salida (default: results)")
    args = ap.parse_args()

    data = load_combined(args.combined_json)
    case = data["case"]
    feats = data["features"]

    out_energy = os.path.join(args.outdir, f"{case}_energy.png")
    out_w1     = os.path.join(args.outdir, f"{case}_w1.png")

    plot_lines(
        features=feats,
        series_by_run=data["energy_by_run"],
        taus=data["tau_energy"],
        ylabel="Energy",
        out_path=out_energy
    )

    plot_lines(
        features=feats,
        series_by_run=data["w1_by_run"],
        taus=data["tau_w1"],
        ylabel="W1",
        out_path=out_w1
    )

    print(f"Guardado: {out_energy}")
    print(f"Guardado: {out_w1}")

if __name__ == "__main__":
    main()
