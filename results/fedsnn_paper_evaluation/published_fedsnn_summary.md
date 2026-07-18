# Fed-SNN Table I evaluation summary

Execution completion: **true**

Scientific status: **equivalence not established**. IID and non-IID treatments are not pooled.

## cifar10_fedsnn_paper_reported_iid_evaluation

Distribution: `iid`; alpha: `n/a`; seeds completed: 3/3.

| Seed | Final round | Official test | Macro-F1 | Paper reference | Signed pp | Absolute pp |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 100 | 0.815 | 0.814874 | 0.7644 | 5.06 | 5.06 |
| 17 | 100 | 0.8216 | 0.82112 | 0.7644 | 5.72 | 5.72 |
| 27 | 100 | 0.8155 | 0.815215 | 0.7644 | 5.11 | 5.11 |

Settings: local epochs `5`, clients `10/2`, timesteps `20`, momentum `0.95`, weight decay `0.0001`, aggregation `uniform`.

| Training examples | Internal validation | Official test | Official-test accesses | Status |
|---:|---:|---:|---:|---|
| 50000 | 0 | 10000 | 1 | equivalence_not_established |

## cifar10_fedsnn_paper_reported_noniid_evaluation

Distribution: `label_dirichlet_non_iid`; alpha: `0.5`; seeds completed: 3/3.

| Seed | Final round | Official test | Macro-F1 | Paper reference | Signed pp | Absolute pp |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 100 | 0.7201 | 0.72054 | 0.7394 | -1.93 | 1.93 |
| 17 | 100 | 0.758 | 0.7544 | 0.7394 | 1.86 | 1.86 |
| 27 | 100 | 0.7332 | 0.730468 | 0.7394 | -0.62 | 0.62 |

Settings: local epochs `5`, clients `10/2`, timesteps `20`, momentum `0.95`, weight decay `0.0001`, aggregation `uniform`.

| Training examples | Internal validation | Official test | Official-test accesses | Status |
|---:|---:|---:|---:|---|
| 50000 | 0 | 10000 | 1 | equivalence_not_established |
