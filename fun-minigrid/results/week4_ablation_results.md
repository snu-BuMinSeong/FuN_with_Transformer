# Week 4 Baseline vs Ablation Results

## Table 1. Best checkpoint sample evaluation

| Model | Seed | Sample Success Rate | Sample Mean Return | Sample Episode Length | Argmax Success Rate |
|---|---:|---:|---:|---:|---:|
| baseline | 1 | 0.320000 | 0.127364 | 223.510000 | 0.000000 |
| baseline | 11 | 0.480000 | 0.243948 | 195.570000 | 0.000000 |
| baseline | 44 | 0.450000 | 0.237780 | 196.450000 | 0.000000 |
| ablation | 1 | 0.350000 | 0.160604 | 215.110000 | 0.000000 |
| ablation | 11 | 0.820000 | 0.496936 | 134.740000 | 0.000000 |
| ablation | 44 | 0.640000 | 0.362080 | 167.200000 | 0.000000 |

## Table 2. Model mean comparison

| Model | Mean Sample Success Rate | Mean Sample Return | Mean Episode Length | Mean Argmax Success Rate |
|---|---:|---:|---:|---:|
| baseline | 0.416667 | 0.203031 | 205.176667 | 0.000000 |
| ablation | 0.603333 | 0.339873 | 172.350000 | 0.000000 |

## Table 3. Difference

Ablation - baseline.

| Metric | Ablation - Baseline |
|---|---:|
| Success rate difference | 0.186667 |
| Mean return difference | 0.136843 |
| Episode length difference | -32.826667 |
| Argmax success difference | 0.000000 |
