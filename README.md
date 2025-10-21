# Early Detection of Backdoor Attacks in Federated Learning via Ecosystemic Symmetry Breaking

Implementation and reproducibility package for the experiments described in:

**Carlos Mario Braga, Manuel A. Serrano, and Eduardo FernÃ¡ndez-Medina (2025)**  
*Early Detection of Backdoor Attacks in Federated Learning via Ecosystemic Symmetry Breaking*  
Submitted to the **First International Workshop on Security and Privacy in Federated and Distributed Architectures (FEDAS'25)**,  
in conjunction with **BDCAT 2025**, ACM, Nantes, France.

This repository provides the full pipeline for reproducing the experiments, figures, and statistical analyses described in the paper.

---

## Base Software

This implementation extends the canonical framework by **Bagdasaryan et al. (2020)**:

> *How To Backdoor Federated Learning*, AISTATS 2020, PMLR v108, pp. 2938â€“2948.

The original code includes:
- **Federated Averaging (FedAvg)** implementation in PyTorch.
- Datasets: **CIFAR-10** and **MNIST**.
- Attack strategies: **model replacement** and **semantic backdoor**.
- Multi-client simulation with benign and adversarial participants.

The official baseline repository cited in the paper is:  
`https://github.com/GSYAtools/Backdoor_attacks_FL`

This project preserves the training, aggregation, and attack logic from the canonical framework to ensure comparability with prior results.

---

## Repository Structure

```
â”œâ”€â”€ generate_files.py              # Generates baseline + delta combinations (e.g., t1_1.csv ... t1_5.csv)
â”œâ”€â”€ calibrate_thresholds_proj.py   # Calibrates Energy/Wasserstein-1 thresholds using bootstrap
â”œâ”€â”€ compare_distribution_proj.py   # Compares runs against the baseline and triggers anomaly alerts
â”œâ”€â”€ plot_graphs.py                 # Plots Energy and W1 values with threshold overlays
â”œâ”€â”€ data/                          # Input CSVs (T0.csv, T1_delta.csv, etc.)
â”œâ”€â”€ results/                       # Output results (JSON + PNG files)
â””â”€â”€ README.md
```

Dependencies:
- **Python â‰¥ 3.8**
- `numpy`, `pandas`, `scipy`, `matplotlib`

---

## Execution Pipeline

1. **Baseline calibration (benign reference)**  
   Calibrate thresholds using the benign dataset `T0.csv`:
   ```bash
   python calibrate_thresholds_proj.py data/T0.csv --B 500 --n0 50 --pctl 99
   ```

   This produces a JSON file in `results/` such as:
   ```
   results/thresholds_T0_nadd12_pctl99.json
   ```

2. **Generate per-run datasets (T1, T2, T3)**  
   Using a delta file representing new updates (benign or adversarial):
   ```bash
   python generate_files.py data/T0.csv data/T2_delta.csv data/t2
   ```

   This creates files: `data/t2_1.csv ... data/t2_5.csv`

3. **Compare distributions and detect deviations**  
   Evaluate deviations using the calibrated thresholds:
   ```bash
   python compare_distribution_proj.py T2 results/thresholds_T0_nadd12_pctl99.json --data-dir data
   ```

   Output:
   ```
   results/results_T2.json
   ```

4. **Plot summary graphs**  
   Generate the figures for Energy and W1 metrics:
   ```bash
   python plot_graphs.py results/results_T2.json
   ```

   Output files in `results/`:
   - `T2_energy.png`
   - `T2_w1.png`

---

## Attack Scenarios

The experiments reproduce the three canonical cases described in the paper.  
Each scenario corresponds to a different type or intensity of client update.

| Scenario | Description | Type of Update | Attack Strength (Î³) | Expected Behavior |
|-----------|--------------|----------------|----------------------|-------------------|
| **T1** | Benign control (no adversarial activity). | Regular client updates only. | â€“ (none) | Distances remain below threshold (no alerts). |
| **T2** | Strong attack: semantic backdoor using model replacement. | Adversarial updates scaled aggressively. | **Î³ = 100** | Clear multi-projection deviations; early detection triggered. |
| **T3** | Stealthy attack: reduced-scale backdoor. | Adversarial updates with minimal scaling. | **Î³ = 20** | Moderate deviations; detectable but weaker signals. |

The scaling factor **Î³** controls the amplitude of the malicious update relative to the global model.  
Higher values of Î³ lead to faster convergence of the backdoor but make detection easier.

---

## Command-Line Parameters

Each stage in the pipeline can be executed independently. Below are the main options.

### `generate_files.py`

Creates CSVs combining the baseline (`T0.csv`) and delta updates.

**Usage**
```bash
python generate_files.py <T0.csv> <delta.csv> <output_prefix>
```

| Argument | Description |
|-----------|-------------|
| `<T0.csv>` | Baseline benign updates. |
| `<delta.csv>` | Delta file with new updates (benign or adversarial). |
| `<output_prefix>` | Output prefix for generated CSVs. |

---

### `calibrate_thresholds_proj.py`

Performs bootstrap calibration of Energy and Wasserstein-1 thresholds.

**Usage**
```bash
python calibrate_thresholds_proj.py data/T0.csv [options]
```

| Option | Default | Description |
|---------|----------|-------------|
| `--B` | 500 | Bootstrap replications. |
| `--n0` | 50 | Baseline size per replication. |
| `--n-add` | â€“ | Effective size of new (weighted) sample. |
| `--pstar` | â€“ | Equivalent to `--n-add`, defines effective proportion p*. |
| `--pctl` | 95 | Percentile used for threshold calibration. |
| `--debias-proj` | â€“ | Removes the dominant benign projection direction. |
| `--use-l2-perp` | â€“ | Replaces l2 with the orthogonal norm (requires `--debias-proj`). |
| `--out-name` | auto | Output file name. |

---

### `compare_distribution_proj.py`

Compares each generated run against the benign baseline and checks for deviations.

**Usage**
```bash
python compare_distribution_proj.py <CASE> <thresholds.json> [options]
```

| Option | Default | Description |
|---------|----------|-------------|
| `--data-dir` | `data` | Folder with generated CSVs. |
| `--outdir` | `results` | Output folder for results. |
| `--drift-prefix` | â€“ | Optional drift correction prefix. |
| `--rule` | `any` | Trigger rule: `any` (OR) or `both` (AND). |

---

### `plot_graphs.py`

Generates the figures for Energy and W1 metrics.

**Usage**
```bash
python plot_graphs.py results/results_T2.json
```

| Option | Default | Description |
|---------|----------|-------------|
| `--outdir` | `results` | Output folder for figures. |

---

## Reproducibility & Review Notice

This repository accompanies the submission:

> *Early Detection of Backdoor Attacks in Federated Learning via Ecosystemic Symmetry Breaking*  
> Submitted to the **First International Workshop on Security and Privacy in Federated and Distributed Architectures (FEDASâ€™25)**,  
> in conjunction with **BDCAT 2025**, ACM, Nantes, France.

The repository is provided **solely for peer review and reproducibility evaluation**.  
It reproduces all experiments and figures reported in the paper (Sections 3â€“5).

At this stage, **no public license is granted**. Redistribution or reuse is not allowed until final publication and licensing.

If you are a reviewer:
- All scripts can be executed with Python â‰¥ 3.8 and standard dependencies.
- Default parameters reproduce the benign (`T1`), strong attack (`T2`), and stealthy attack (`T3`) scenarios.
- Output files (`.json`, `.png`) will appear in the `results/` folder.

For questions related to this reproducibility package, please contact the corresponding author through the submission system.

---

## Acknowledgments

This reproducibility package was developed as part of the research submitted to FEDAS â€™25.  
The underlying research received support from the following projects:

- **Di4SPDS (PCI2023145980-2)** â€” funded by MCIN/AEI/10.13039/501100011033 and by the European Union (Chist-ERA Program).  
- **KOSMOS-UCLM (PID2024-155363OB-C44)** â€” funded by MCIN/AEI/10.13039/501100011033/FEDER, EU.  
- **AURORA (SBPLY/24/180225/000074)** â€” funded by the Regional Government of Castilla-La Mancha and the European Regional Development Fund (FEDER).  
- **RADAR (2025-GRIN-38447)** â€” funded by FEDER.  
- **RED2024-154240-T** â€” funded by MICIU/AEI/10.13039/501100011033.
