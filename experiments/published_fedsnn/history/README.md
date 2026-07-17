# Prior CIFAR-10 Fed-SNN implementation

The retained configuration records the computationally complete independent implementation that produced 18.23–26.79% official-test accuracy. It used unsigned inputs, augmentation, an unsigned Poisson encoder, summed output voltage, default PyTorch initialization, 25 timesteps, BNTT epsilon 1e-5, retained local remainders, and example-count aggregation.

This configuration is intentionally absent from the active manifest. Its existing run directories and generated evidence remain unchanged under `runs/published_fedsnn`; the corrected Table I configurations use different names and `runs/fedsnn_paper_evaluation`.
