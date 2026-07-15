# Federated baseline

## Scientific objective

Evaluate federated SNN baselines under explicit client partition and aggregation rules.

## Methods

Methods and hyperparameters must be recorded with the resolved partition, client sampling, aggregation weights, and random seeds.

## Experiment protocol

Only validated scientific records with compatible protocols may enter the analysis. Exclusions and missing observations must be reported explicitly.

## Evidence collected

The centralized prerequisite is now available in the [centralized summary](../results/centralized/centralized_summary.md). The principal centralized comparison for the initial FedAvg evaluation is the SHD LIF `independent_evaluation` result: 76.3693% ± 2.26431% official-test accuracy across seeds 7, 17, and 27.

No federated evidence has yet been collected. Neither federated implementation completion nor federated evaluation completion is claimed here.

## Unresolved scientific limitations

Client partition validity, convergence, communication volume, runtime, memory, utilization, energy, and scheduler-resource accounting remain unmeasured. The analysis must establish whether centralized and federated checkpoint-selection rules are comparable before interpreting any accuracy difference.

## Conclusions permitted by the evidence

The centralized reference permits a predefined comparison target for FedAvg. No conclusion about federated accuracy, convergence, communication efficiency, or resource behaviour is currently permitted.
