# Experiments: Distribution of Gene Tree Topologies with Duplication, Loss, and Coalescence

This repository contains scripts for reproducing the computational experiments and figures presented in the paper. The experiments investigate gene tree topology probabilities and expected branch lengths under a model incorporating gene duplication, loss, and coalescence.

## Requirements

- Python 3.x
- NumPy and Matplotlib (for plotting and numerical calculations)
- [dupcoal](https://github.com/meganlsmith/dupcoal)

Install the required Python packages, if needed:

```bash
pip install numpy matplotlib
```

## Reproducing the Figures and Tables

### Figure 3: Distribution of Gene Tree Topologies

Run the following command to generate Figure 3:

```bash
python3 ./figure_3.py
```

The script generates the plot showing the probabilities of different gene tree topology groups as a function of duplication time.

### Figure 4 and Table 3: Expected Branch Lengths

To calculate the theoretical expected branch lengths for a species-tree height of \(t_s=2\) and duplication rates \(\lambda=0.01,0.02,\ldots,0.1\), run:

```bash
python3 ./theory.py --ts 2 --output <DESTINATION>
```

Replace `<DESTINATION>` with the desired output path.

### Table 4: Empirical Branch Lengths from DupCoal Simulations

The empirical branch lengths are obtained by simulating gene trees using DupCoal with a fixed loss rate \(\mu=0\) and varying duplication rates \(\lambda\).

#### Step 1: Run the simulations

The following commands use 500,000 replicates per parameter setting and the species tree specified in `2_sp.txt`.

For duplication rates from 0.01 to 0.1 in increments of 0.01:

```bash
mkdir -p data

for lambda in 0.01 0.02 0.03 0.04 0.05 \
              0.06 0.07 0.08 0.09 0.1
do
    python3 dupcoal_v3.0.0.py \
        --stree 2_sp.txt \
        --mu_par 0 \
        --lambda_par "$lambda" \
        --reps 500000 \
        --output "data/dist_exte_${lambda}"
done
```

For additional intermediate duplication rates:

```bash
for lambda in 0.015 0.025 0.035 0.045 0.055 \
              0.065 0.075 0.085 0.095
do
    python3 dupcoal_v3.0.0.py \
        --stree 2_sp.txt \
        --mu_par 0 \
        --lambda_par "$lambda" \
        --reps 500000 \
        --output "data/dist_exte_${lambda}"
done
```

These simulations produce output for duplication rates between 0.01 and 0.1, including intermediate values at increments of 0.005.

#### Step 2: Extract empirical branch lengths

After completing the simulations, run:

```bash
python3 ./extract_from_dupcoal.py
```

This script extracts the branch lengths from the DupCoal simulation outputs for comparison with the theoretical expectations.

## Repository Contents

| File | Description |
|---|---|
| `figure_3.py` | Generates Figure 3, showing gene tree topology probabilities. |
| `theory.py` | Calculates theoretical expected branch lengths. |
| `dupcoal_v3.0.0.py` | Simulates gene trees under the duplication–loss–coalescence model. |
| `extract_from_dupcoal.py` | Extracts empirical branch lengths from simulation outputs. |
| `2_sp.txt` | Species-tree input for the two-species simulations. |
| `data/` | Directory for simulation outputs. |

## Reproducibility Notes

- The default simulation settings use \(\mu=0\), \(t_s=2\), and 500,000 replicates per duplication-rate setting.
- Run the simulations before executing `extract_from_dupcoal.py`.
- Ensure that the output directory and filenames match the paths expected by the extraction script.
- Simulation results may vary slightly between runs because of stochastic sampling.

