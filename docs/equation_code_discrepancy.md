# Equation versus public-code PfA behaviour

Let x be current with neuron-wise mean m, deviation d = x - m, variance v = mean(d²), threshold θ, and positive λ.

Equation PfA computes:

`weight = θd / (d² + 2λ + 2v)`

The behaviour independently reconstructed from public commit `0898fc22480c86bccd7f6fccb0d43fdfbd579797` computes:

`weight = θd / (3d² + 2λ)`

The second denominator arises because the public path uses each element squared deviation in both denominator terms where the equation uses neuron-wise variance for one term. For heterogeneous currents, the outputs differ. Both paths then compute `bias = (θ - weight(x + m)) / 2`, apply `sigmoid(weight x + bias)`, and multiply x by the attention.

The implementations are intentionally separate, independently tested, and named `equation` and `public_behavior`. Equation PfA is evaluated in SHD and SSC `independent_evaluation` experiments. Public behaviour is evaluated only in the explicitly labelled SHD `published_protocol` experiment. Their metrics are not pooled.

This discrepancy does not authorize combining PfA with DCLS. No verified repository source specifies that architecture, so the existing `dcls_shd` attention prohibition remains.
