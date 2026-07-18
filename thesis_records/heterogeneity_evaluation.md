# Heterogeneity evaluation

## Scientific objective

Measure the effect of client data and event-stream heterogeneity on federated SNN accuracy, convergence, and fairness.

## Methods

Each treatment must specify the partition, corruption mechanism, client assignment, seeds, and comparison baseline.

## Experiment protocol

Only validated scientific records with compatible protocols may enter the analysis. Exclusions and missing observations must be reported explicitly.

## Evidence collected

The validated [SHD FedAvg baseline](federated_baseline.md) is available as the prerequisite reference.

The [corrected CIFAR-10 Fed-SNN summary](../results/fedsnn_paper_evaluation/published_fedsnn_summary.md) provides a completed distribution comparison with the model, optimizer, local work, participation, and evaluation policy held fixed. Seeds 7, 17, and 27 all completed at round 100.

IID official-test accuracies were 81.50%, 82.16%, and 81.55%, giving 81.7367% ± 0.3675 percentage points. Alpha-0.5 non-IID accuracies were 72.01%, 75.80%, and 73.32%, giving 73.7100% ± 1.9249 percentage points. The descriptive mean reduction from IID to non-IID was 8.0267 percentage points. Macro-F1 means of 81.7070% and 73.5136% closely follow accuracy and provide no evidence of class collapse.

This corrected evidence remains separate from the SHD heterogeneity record and from the unsuccessful superseded 18–27% Fed-SNN implementation. It supplies protocol-aligned motivation for later heterogeneity-aware methods, but it is not pooled into SHD results and does not establish a cross-dataset comparison.

## Unresolved scientific limitations

Only three seeds are available, so statistical significance is not claimed. The evidence does not isolate a causal mechanism for the observed reduction, and cross-dataset, energy, and exact source-equivalence conclusions remain unsupported. Fed-SNN scientific status remains `equivalence_not_established`.

## Conclusions permitted by the evidence

Within the declared CIFAR-10 matrix, non-IID allocation is descriptively associated with an 8.0267-point lower mean official-test accuracy than IID allocation. This motivates, but does not validate, later heterogeneity-aware methods. No claim of novelty, causality, statistical significance, energy efficiency, or exact reproduction is permitted.
