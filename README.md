# Early Detection of Backdoor Attacks in Federated Learning via Ecosystemic Symmetry Breaking

Implementation and replication package for the experiments presented in:

> **Carlos Mario Braga, Manuel A. Serrano, and Eduardo Fernádez-Medina (2025)**  
> *Early detection of backdoor attacks in federated learning via ecosystemic symmetry breaking.*  
> In *Proceedings of the First International Workshop on Security and Privacy in Federated and Distributed Architectures (FEDAS 2025)*, ACM, Nantes, France.  

---

## Base Software

This implementation extends the canonical framework of **Bagdasaryan et al. (2020)**:  
> *How To Backdoor Federated Learning*, AISTATS 2020, PMLR v108, pp. 2938-2948.  

The original software implements:
- **Federated averaging (FedAvg)** in PyTorch.  
- Datasets: **CIFAR-10** and **MNIST**.  
- **Model-replacement** and **semantic backdoor** attacks.  
- Simulation of multiple clients (benign and adversarial).  

The original repository is referenced in the article as:  
`https://github.com/GSYAtools/Backdoor_attacks_FL`

This work preserves all training, aggregation and attack logic from that baseline to ensure full comparability with canonical experiments.

---

## Modifications Introduced

Two minimal but crucial changes were introduced to `training.py`:

1. **Fixed client participation**  
   Client 99 is explicitly forced to participate in every round to allow consistent per-node monitoring across all scenarios.

2. **Compact logging of local updates**  
   A new block of code records per-round, per-client update statistics into a CSV file (`updates_log.csv`):

```python
# --- Logging of compact updates (L2 + projections) per client ---
deltas = []
for (pname, pdata) in model.state_dict().items():
    if not torch.is_floating_point(pdata):
        continue
    if helper.params.get('tied', False) and pname == 'decoder.weight' or '__' in pname:
        continue
    target_value = target_model.state_dict()[pname]
    delta = (pdata - target_value).view(-1)
    deltas.append(delta)

if deltas:
    delta_vec = torch.cat(deltas)

    global PROJ_VECS
    if PROJ_VECS is None:
        dim = delta_vec.numel()
        PROJ_VECS = [torch.tensor(np.random.randn(dim), dtype=torch.float32) for _ in range(NUM_PROJ)]

    l2_norm = delta_vec.norm().item()
    proj_vals = [torch.dot(delta_vec, r[:delta_vec.numel()]).item() for r in PROJ_VECS]

    with open("updates_log.csv", "a") as f:
        line = [epoch, current_data_model, l2_norm] + proj_vals
        f.write(",".join(map(str, line)) + "\n")
```

This addition creates a lightweight, privacy-preserving record of the client'ynamics for later analysis.

---

## Repository Structure

```
fl/
calibrate_thresholds_proj.py           # Script to calibrate Energy/Wasserstein distance in baseline T0
compare_distribution_proj.py           # Script to compute Energy/Wasserstein distances in cases T1, T2, T3
generate_files.py                      # Script to generate de files from the delta generated in the training   
plot_graphs.py                         # Graphs generator
data/                                  # Input data: updates_log.csv and related logs
output/                                # Output results and analysis plots
README.md
```

---

## Method Overview

Each client update is summarized locally by:
- **L2 norm** of its update vector, and  
- **10 fixed Gaussian projections** of that same vector.  

These compact statistics form the ecosystem representation of the client.  
A benign reference distribution `T0` (50 benign rounds) is built to calibrate thresholds via bootstrap (99th percentile).  
Then, for every new update:

- **Energy distance** (Székely & Rizzo)  
- **Wasserstein-1 distance** (Villani)

are computed between the new observation and `T0`.  
Exceeding either threshold signals **ecosystemic symmetry breaking**, interpreted as a potential poisoning event.

---

## Experimental Scenarios

| Scenario | Description | Î³ |
|-----------|--------------|--|
| **T1** | Additional benign rounds (control) | -- |
| **T2** | Strong semantic backdoor (model replacement) | 100 |
| **T3** | Stealthy backdoor (lower scaling) | 20 |

Each scenario is repeated five times with different random seeds.  
Detection is declared when multiple projections simultaneously exceed their thresholds.

---

## Results (as reported in the paper)

- **T1 (Benign)** â†’ isolated threshold exceedances only; no alerts.  
- **T2 (Strong attack)** â†’ immediate multi-projection exceedances in the first local round.  
- **T3 (Stealthy attack)** â†’ consistent, though subtler, exceedances across runs.  

These results confirm **early, unsupervised, trigger-free detection**, fully compatible with secure aggregation.

---

## Running the Experiments

Train the federated model (CIFAR-10 example):

After each round, `updates_log.csv` will be written in `/data`.

Compute statistical distances and visualize results:

```bash
python analyze_distances.py --input data/updates_log.csv --output output/
```

---

## ðŸ“œ License

This repository is provided for academic and research purposes only.  
The base code from Bagdasaryan et al. (2020) retains its original license and attribution.
