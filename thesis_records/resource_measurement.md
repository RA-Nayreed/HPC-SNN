# Client resource measurement and cost estimation

## Scientific question and evidence status

This evaluation asks whether the runtime and device energy of an unseen federated SNN client can be predicted before process assignment, and whether causal spike history improves that prediction beyond example count, event count, sequence length, and padding.

The committed collection passed its declared consistency gates. Six accepted executions produced 6,000 accepted client measurements, and the measurement and cost-estimation system operated successfully. This establishes descriptive evidence under the declared protocol. It does not establish statistical significance, causality, universal generalization, energy advantage, production scheduling superiority, literature reproduction, or thesis novelty.

## Integration with FedAvg

Measurement extends the established client path rather than introducing another trainer. The distributed entry point constructs the established dataset bundle, split, partition, model, client selections, local Adam optimizer, sample-count aggregation, checkpoint selection, and one-time official-test evaluation. When measurement is disabled, the hook is absent and the established random-number consumption, data-loader seeds, iterator order, client selection, update order, aggregation, checkpoints, and record fields are unchanged.

The scientific topology is rank zero on one visible GH200, one distributed process, and one client process per GPU. Rank zero invokes the established `train_client` function through a measurement wrapper. Measurements never control a tensor operation. Sampler threads do not use Torch random functions. The client features needed before assignment are computed from the client training indices and resolved training seed before rank zero assigns work.

Client intervals are identified by dataset, experiment, scientific seed, communication round, selected position, client ID, training seed, execution attempt, and GPU UUID. Accepted client intervals cannot overlap within an attempt. Other interval categories cover model distribution, result collection, aggregation, validation, checkpoint writing, a communication round, and the training execution.

## Scientific matrix

The canonical [manifest](../experiments/resource_measurement/manifest.yaml) expands to exactly six tasks.

| Dataset | Experiment | Seeds | Model | Batch size | Clients selected / total | Rounds |
|---|---|---|---|---:|---:|---:|
| SHD | `shd_lif_client_resource` | 7, 17, 27 | 256/256 LIF | 32 | 10 / 20 | 100 |
| SSC | `ssc_lif_128_client_resource` | 7, 17, 27 | 128/128 LIF | 256 | 10 / 20 | 100 |

Both datasets use one local epoch, Adam at 0.001, sample-count FedAvg, and label-Dirichlet alpha 0.5. The accepted client-record count is exactly `2 × 3 × 100 × 10 = 6,000`: 1,000 records for every dataset/seed execution.

SHD uses the established stratified validation split, 10 ms integration, 700-to-140 input reduction, and the established 256/256 LIF model. Its official test collection is constructed only after validation has selected a checkpoint. SSC uses all 75,466 official training examples, all 9,981 official validation examples, and all 20,382 official test examples, with 10 ms integration, the same input reduction, and the established 128/128 LIF model. Official validation selects the SSC checkpoint, and the official test collection is accessed once afterward. Dataset results remain separate; cross-dataset rows are reported only in the declared joint and transfer evaluations.

The validator rejects another dataset, seed, topology, backend, process count, GPU count, MPS setting, sampling interval, idle duration, model, batch size, client protocol, round count, example cap, batch cap, validation policy, or official-test policy.

## Timing boundaries

Every client interval begins with a monotonic nanosecond timestamp. Data-loader iterator waits are timed around batch retrieval. CUDA events delimit GPU work on the active stream, and CUDA is synchronized before the final event duration is read. The interval ends with another monotonic nanosecond timestamp.

For an accepted interval,

\[
t_{wall}=t_{data\ wait}+t_{CUDA}+t_{residual\ host}.
\]

The residual is computed from the other three values, and reconciliation uses a declared floating-point tolerance. The CPU test adapter supplies controlled event durations; it does not represent CUDA availability. CUDA-event behavior and NVML sampling require a compatible accelerator host.

## Persistent power sampling and idle reference

One persistent NVML sampler resolves the allocated GPU by UUID rather than assuming CUDA index zero is NVML index zero. It writes each sample to `device_samples.jsonl` and flushes incrementally. A sample contains monotonic nanoseconds, UTC time, GPU UUID, power, GPU and memory utilization, allocated device memory, temperature, optional graphics and memory clocks, optional cumulative hardware energy, backend, configured interval, execution attempt, and error status.

The scientific sampling interval is exactly 100 ms. Requested measurement fails when NVML is unavailable, the visible-device count is not one, UUID resolution is ambiguous, or required fields cannot be sampled. Shutdown closes and flushes records on ordinary completion, Python exceptions, `SIGTERM`, `SIGINT`, and Slurm cancellation.

Each dataset/seed attempt records 30 seconds of idle power after model, CUDA, and measurement initialization and another 30 seconds after evaluation. Data loading, training, validation, and checkpoint I/O are excluded from these idle intervals. Pre-execution and post-execution medians and temperature drift remain separate. The client idle reference is the median of the combined accepted idle samples, while every original idle sample is preserved.

## Device-energy integration

Power is interpolated at each interval boundary from adjacent samples. A sample immediately before and after the interval is mandatory, duplicated or nonmonotonic timestamps are rejected, and a gap above 2.5 times the configured interval invalidates coverage. Nanoseconds are converted to seconds once.

Gross device energy is the trapezoidal integral

\[
E_{gross}=\sum_i \frac{P_i+P_{i+1}}{2}(t_{i+1}-t_i),
\]

and idle-adjusted energy is retained separately as

\[
E_{dynamic}=\sum_i \max\left(\frac{P_i+P_{i+1}}{2}-P_{idle},0\right)(t_{i+1}-t_i).
\]

Both are expressed in joules. Gross energy is never replaced by the idle-adjusted quantity. Sample count, coverage, maximum gap, and an optional cumulative-device-energy difference are recorded. Missing samples are not fabricated.

Execution energy is reconciled into client training, aggregation, validation, checkpoint writing, other measured work, idle-reference contribution, and any unattributed interval. Model distribution and result collection belong to other measured work. A declared interpolation tolerance permits only numerical boundary effects; reconciliation does not force an unexplained remainder to zero. Allocated GPU-hours, utilization, billing units, and device energy are distinct quantities.

## Execution evidence and resource accounting

The [summary JSON](../results/resource_measurement/resource_measurement_summary.json), [summary CSV](../results/resource_measurement/resource_measurement_summary.csv), and [Slurm accounting](../results/resource_measurement/provenance/slurm-accounting.txt) report:

| Quantity | Committed value |
|---|---:|
| Slurm job | 291481 |
| Execution commit | `3ddae173c89125bc69922d80bde5732ed6cd050e` |
| State and exit code | `COMPLETED`, `0:0` |
| Physical GH200 devices | 1 |
| Processes | 1 |
| CPU cores | 72 |
| Slurm memory | 217086M |
| Slurm allocation elapsed | 34,051 s |
| Allocated GPU-hours | 9.45861111111111 h |
| Internal execution time | 33217.69320165331 s |
| Summed accepted-client wall time | 11457.10155192 s |
| Summed accepted-client CUDA-event time | 8159.062962005615 s |
| Pending time | null; not part of allocation execution |

Slurm duration and allocated GPU-hours are scheduler accounting quantities. Internal execution time, summed client wall time, and CUDA-event time are measurements with different boundaries; they must not be substituted for one another. Billing quantities and pending time remain separate.

The six accepted executions are:

| Dataset | Seed | Accepted clients | Internal time (s) | Client wall time (s) | CUDA-event time (s) | Gross energy (J) | Idle-adjusted energy (J) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SHD | 7 | 1000 | 2040.813377056038 | 1347.072477339 | 1179.1335129432678 | 218183.40788355345 | 3322.7407473576236 |
| SHD | 17 | 1000 | 2064.5288088772213 | 1373.212066732 | 1205.274860939026 | 222467.1850438589 | 3071.4136503444015 |
| SHD | 27 | 1000 | 2132.0926596346544 | 1430.063606405 | 1262.6056069602967 | 231701.10518099155 | 2964.2536725687632 |
| SSC | 7 | 1000 | 8940.54641737556 | 2418.340784185 | 1488.929653968811 | 400834.88317312335 | 14306.394297468409 |
| SSC | 17 | 1000 | 8991.110143224243 | 2413.011474932 | 1467.6087600479127 | 400205.4943967287 | 14585.120256270595 |
| SSC | 27 | 1000 | 9048.601795485592 | 2475.401142327 | 1555.5105671463014 | 410220.6191966648 | 14130.676037046904 |

Across accepted client-training intervals, gross device energy is `1883612.6948749206 J`, which converts to `1883.6126948749206 kJ` and `0.5232257485763668 kWh`. Idle-adjusted device energy is `52380.5986610567 J`, which converts to `52.380598661056695 kJ` and `0.014550166294737972 kWh`. These totals cover only accepted client-training intervals. They do not measure the entire Slurm allocation, and they do not support an energy or power advantage over another system.

## Measurement calibration

The [SHD resource configuration](../experiments/resource_measurement/shd/lif_client_resource.yaml) defines the training-only calibration client. The calibration never constructs or accesses the official test dataset. At least ten paired repetitions alternate measured-first and unmeasured-first order. Before each repetition it restores identical model parameters, Python state, NumPy state, Torch CPU state, Torch CUDA state, and deterministic loader order.

Calibration passes only if all of these conditions hold:

- median paired runtime overhead is at most 2%;
- at least 90% of representative measured intervals contain ten or more samples;
- there are no sampling errors;
- exactly one GPU UUID is observed;
- measured and unmeasured parameter updates are numerically identical.

All individual durations, relative overheads, sample counts, UUIDs, and update comparisons are recorded in `instrumentation_calibration.json`. The scientific runner requires a passing artifact and never changes the 100 ms interval in response to calibration.

### Observed calibration

The [committed calibration](../results/resource_measurement/instrumentation_calibration.json) contains one accepted artifact referenced by all six executions:

| Calibration field | Observed value |
|---|---|
| Outcome | passed |
| Paired repetitions | 10 |
| Warm-up executions | one measured, then one unmeasured |
| Warm-up inclusion | excluded from paired observations and overhead statistic |
| State handling | restored before every warm-up and paired execution |
| Median relative overhead | 0.01713823842158517 (about 1.7138%) |
| Declared overhead threshold | 0.02 |
| Minimum samples per measured client | 10 |
| Sample coverage | 1.0 |
| Numerical update identity | true |
| Sampling errors | none |
| GPU UUIDs | `GPU-f50ec698-57aa-4b59-108c-fa678abbc391` only |
| Training data only | true |
| Official-test accesses | 0 |

The observed overhead is nonzero and below the declared threshold. The calibration result therefore supports the accepted instrumentation decision without implying cost-free measurement.

## Client-resource schema

Features available before any client execution are example count, local batch count, raw input events, mean/median/maximum sequence length, valid time bins, estimated padded time bins, padding fraction, event density, represented class count, label entropy, communication round, dataset identity, model identity, and parameter count. They are computed only from that client's training indices. Deterministic padded-work estimation reproduces the client's seeded batch order without constructing the official test dataset.

Before assignment, the record also indicates whether earlier accepted observations exist and how many. Features available only from prior observations are previous duration, gross energy, idle-adjusted energy, layer spike rates, spikes per example, exponentially weighted duration, energy, and spike rates, observation count, and missing-history indicators. Coefficient candidates 0.10, 0.30, 0.50, 0.70, and 0.90 are declared before execution. The selected coefficient is chosen using fitting and client-grouped validation observations only, with seed 27 excluded.

Values available only after the current client finishes are actual batches, presented examples, valid and padded time bins, padding fraction, input events, layer-one and layer-two spike counts and rates, wall time, data-wait time, CUDA-event time, residual host time, gross and idle-adjusted energy, and peak allocated and reserved CUDA memory. Current-execution spike values are never scheduler inputs.

Each record carries an availability dictionary so these three feature times are machine-readable. Validation and official-test examples are rejected from client features.

## Causal history and leakage protection

Rows are ordered by dataset, scientific seed, communication round, selected position, and client ID. Before predicting round `t`, a client's history is constructed only from accepted observations for the same dataset, seed, and client in rounds less than `t`. Later rows are therefore unable to alter an earlier feature row. Client ID is used for grouping and history lookup but is never a predictor.

Seeds 7 and 17 provide fitting and client-grouped model selection. Seed 27 remains outside fitting and selection. The evaluation settings are SHD within-dataset, SSC within-dataset, joint fitting and evaluation, SHD-to-SSC transfer, SSC-to-SHD transfer, and prequential seed-27 evaluation. Exact row identities and hashes are stored for fitting, validation, and evaluation collections.

## Cost models and evaluation

The candidates are:

1. a fitting-set median constant;
2. a size model using example and batch count;
3. an event-structure model adding events, sequence statistics, valid and padded time bins, padding fraction, and density;
4. a historical-spike model adding only causal spike, duration, energy, and missing-history features;
5. a diagnostic oracle that may use current-execution spike information and is prohibited from scheduler export.

Implemented regressors are the median predictor, standardized ridge regression over declared penalties, robust linear regression, and optional log-target variants chosen by grouped validation. Means and scales come only from fitting rows; constant features use a safe scale. JSON model artifacts contain feature order, coefficients, intercept, standardization, target transform, fitting row hashes, dataset and seed identities, validation decision, and software and Git provenance. They contain no executable serialization. Reloaded JSON must reproduce stored predictions within strict tolerance.

The required targets are client wall seconds, CUDA-event seconds, gross device joules, and idle-adjusted device joules. Wall time is primary for assignment and gross energy is primary for energy interpretation. Every target and setting reports mean absolute error, root mean squared error, median and 90th-percentile absolute percentage error, R-squared, Spearman correlation, mean and median signed error, and count. Percentage errors use a declared positive denominator floor. Slices cover dataset, seed, joint rows, client-size quartiles, sequence-length quartiles, round intervals, clients without history, and clients with history.

### Feature availability and data separation

| Candidate | Information available to the model | Assignment use |
|---|---|---|
| `constant` | fitting-collection target median | available before assignment |
| `size` | example count and local batch count | available before assignment |
| `event_structure` | size plus input events, sequence statistics, valid and estimated padded time bins, padding fraction, and event density | available before assignment |
| `historical_spike` | static features plus causal values from earlier accepted observations of the same client | available before assignment when history exists |
| `diagnostic_oracle` | current-execution spike measurements | available only after the current execution; prohibited from deployment-oriented export |

Client identity supports grouping and causal-history lookup but is not a predictor. Current-execution measurements are unavailable before assignment and are excluded from deployable models. Seeds 7 and 17 supplied fitting and client-grouped selection; seed 27 was excluded from fitting, historical-weight selection, and model selection. The committed evaluation settings are within-SHD, within-SSC, joint, SHD-to-SSC transfer, SSC-to-SHD transfer, and prequential seed-27 evaluation.

### Exported models

The [scheduling model](../results/resource_measurement/client_cost_model.json) is a ridge `event_structure` model with target `client_wall_time_seconds`, regularization `0.0001`, and fitting seeds 7 and 17. Its feature order is:

1. `example_count`;
2. `local_batch_count`;
3. `total_raw_input_events`;
4. `mean_sequence_length`;
5. `median_sequence_length`;
6. `maximum_sequence_length`;
7. `total_valid_time_bins`;
8. `estimated_padded_time_bins`;
9. `padding_fraction`;
10. `event_density`.

`client_id` and current-execution spike fields are absent. The [gross-energy model](../results/resource_measurement/energy_cost_model.json) uses the same `event_structure` feature order and ridge family with target `gross_energy_joules` and fitting seeds 7 and 17. The committed evaluation reports `model_json_roundtrip_verified: true`: reloading each stored JSON reproduces predictions. The energy model is an empirical model for this collection, not a universal physical law.

### Joint seed-27 evaluation

The following tables reproduce the authoritative joint evaluation of the untouched 2,000 seed-27 client records. Absolute-error units match the target; APE values are fractions. The diagnostic oracle is included only as a post-execution diagnostic.

#### Client wall time (seconds)

| Model | MAE | Median AE | P90 AE | RMSE | Median APE | P90 APE | R² | Spearman ρ | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 0.535811294804 | 0.5437057485000001 | 0.9373968046 | 0.603987933367345 | 0.2604593096703142 | 0.5808101324133589 | -0.0018474037417719469 | 0.0 | 2000 |
| size | 0.08459618490282145 | 0.0730184372472844 | 0.16736925691565915 | 0.10933737172145008 | 0.03874939898807017 | 0.09066487334433164 | 0.9671691359638711 | 0.992728111005165 | 2000 |
| event_structure | 0.06884469267330857 | 0.066351972627951 | 0.11281469013244072 | 0.08285417813628894 | 0.033442242048620285 | 0.06660915772954716 | 0.9811472859662216 | 0.994406546671811 | 2000 |
| historical_spike | 0.04278198963016118 | 0.03657851881133589 | 0.0780543551093871 | 0.05922437086851806 | 0.019143486165949034 | 0.045585288774428176 | 0.9903673459208833 | 0.995512964878241 | 2000 |
| diagnostic_oracle | 0.0958338163946473 | 0.09132422922644301 | 0.15781277735913107 | 0.11005693423851735 | 0.04302817461585441 | 0.10435917009215079 | 0.9667355862171975 | 0.9941398300349574 | 2000 |

#### CUDA-event time (seconds)

| Model | MAE | Median AE | P90 AE | RMSE | Median APE | P90 APE | R² | Spearman ρ | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 0.26645063631629945 | 0.24198390579223616 | 0.5551142776489258 | 0.3442734162193094 | 0.1662752722570181 | 0.3748354581172131 | -0.133293604775653 | 0.0 | 2000 |
| size | 0.0763083546843217 | 0.07250780909104382 | 0.12143988442002693 | 0.08617665560153755 | 0.05153802572320113 | 0.08506272274927805 | 0.9289907817735998 | 0.988846068727149 | 2000 |
| event_structure | 0.07553541764813324 | 0.07235978633440726 | 0.11167373378190251 | 0.08199524081961747 | 0.05268777117471792 | 0.07236216311224308 | 0.9357145406671056 | 0.9933739515652682 | 2000 |
| historical_spike | 0.07946785055198213 | 0.07656706420503623 | 0.11881703667612328 | 0.08649454985776153 | 0.05387025453621168 | 0.09147338328944689 | 0.9284659283167848 | 0.9911430092857524 | 2000 |
| diagnostic_oracle | 0.0931686127228165 | 0.08965905204284841 | 0.13386792811689405 | 0.09939937294070582 | 0.06566630077333824 | 0.09724099235910331 | 0.9055280783854702 | 0.9919708669927167 | 2000 |

#### Gross device energy (joules)

| Model | MAE | Median AE | P90 AE | RMSE | Median APE | P90 APE | R² | Spearman ρ | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 91.10198270640485 | 92.93444353012168 | 156.82492801404214 | 102.50321847816257 | 0.27024922252376216 | 0.6011277920779611 | -0.0016640821659030802 | 0.0 | 2000 |
| size | 13.671509464251521 | 11.802625431632592 | 27.152791919182217 | 17.651978490035876 | 0.037697605602695146 | 0.09017856454791359 | 0.9702947045453406 | 0.9930329468269733 | 2000 |
| event_structure | 11.216858334971583 | 10.85648933302295 | 18.579205432184253 | 13.47694736940239 | 0.03329355789126251 | 0.06645202484805976 | 0.9826846943594206 | 0.994749131529932 | 2000 |
| historical_spike | 7.281786352286803 | 6.368535005436371 | 13.15520050613449 | 9.830491646239004 | 0.020456792261673823 | 0.04811656076067036 | 0.9907870741846558 | 0.9958873349718337 | 2000 |
| diagnostic_oracle | 15.672264934536354 | 14.991273070938178 | 25.701233134850646 | 17.954602063856743 | 0.04381325121861285 | 0.10484531674765998 | 0.969267445151161 | 0.9945215951303987 | 2000 |

#### Idle-adjusted device energy (joules)

| Model | MAE | Median AE | P90 AE | RMSE | Median APE | P90 APE | R² | Spearman ρ | n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant | 5.588457311704704 | 3.8479327930819993 | 10.818624571938262 | 6.687197716538244 | 0.6467977595441463 | 1.5131888515358587 | -0.2013673698181937 | 0.0 | 2000 |
| size | 0.6768774366028601 | 0.3591551302279925 | 1.3671488585911176 | 1.173808646084354 | 0.053755260032856064 | 0.359608567155403 | 0.9629846285094928 | 0.9588942400121795 | 2000 |
| event_structure | 0.6729312828219791 | 0.3604094094492716 | 1.357045479752213 | 1.1730769410503314 | 0.05178313537956877 | 0.3621792610597913 | 0.9630307619096935 | 0.9607420713455627 | 2000 |
| historical_spike | 0.8218589605092875 | 0.6471485467360596 | 1.4123275589977677 | 1.149689143448548 | 0.10998326362726059 | 0.30005337381316605 | 0.9644901886758258 | 0.9593774608443651 | 2000 |
| diagnostic_oracle | 0.714068782645882 | 0.4252872269438267 | 1.4101663328214342 | 1.174751340866848 | 0.06146977501037278 | 0.3946023303671487 | 0.9629251499742267 | 0.957613935903484 | 2000 |

### Prequential feature ablation

The committed [feature-ablation source](../results/resource_measurement/source_feature_ablation_comparison.csv) gives these seed-27 median absolute client-wall-time errors for the joint prequential setting:

| Dataset | constant | size | event_structure | historical_spike | diagnostic_oracle |
|---|---:|---:|---:|---:|---:|
| SHD | 0.5649508759999999 | 0.059673647276199504 | 0.07027049376196515 | 0.04088364796993127 | 0.12312305529387013 |
| SSC | 0.5142956855 | 0.08629442668652731 | 0.06209416433775661 | 0.03218091726004002 | 0.06842086281668758 |

These values describe this prequential setting only. They do not replace the separately predeclared within-dataset comparisons used by the adoption rule.

### Cross-dataset transfer

The exact seed-27 client-wall-time metrics for fits transferred between datasets are:

| Direction | Model | MAE | Median AE | P90 AE | RMSE | Median APE | P90 APE | R² | Spearman ρ | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SHD → SSC | constant | 1.101120593827 | 1.118683833 | 1.6331692197999998 | 1.149558057054281 | 0.44873634765011516 | 0.5430411184498057 | -11.121795076500447 | 0.0 | 1000 |
| SHD → SSC | size | 2.6295786012024776 | 2.5463936449601485 | 3.683046449450923 | 2.717753861117883 | 1.0484449231796886 | 1.239653038566619 | -66.75250568158359 | 0.9843651611192329 | 1000 |
| SHD → SSC | event_structure | 0.8114480654068984 | 0.8166645570853253 | 1.178278501419302 | 0.8821221063206054 | 0.33841880127611496 | 0.43654731810539676 | -6.137769678970589 | 0.9726498561200385 | 1000 |
| SHD → SSC | historical_spike | 0.09641492816808389 | 0.06702076038644145 | 0.21957298413852647 | 0.12516543746224884 | 0.02750415647987159 | 0.09715199954303365 | 0.8562943108730273 | 0.9850407730407732 | 1000 |
| SHD → SSC | diagnostic_oracle | 1.5431929903860964 | 1.5109896132234242 | 1.8602206660891396 | 1.5606310383211845 | 0.6242146273395641 | 0.6792110022815812 | -21.34115381218493 | 0.7737364617364618 | 1000 |
| SSC → SHD | constant | 0.872439579095 | 0.8887853655 | 1.2250193145 | 0.9132501489329973 | 0.6286866003497271 | 1.1369266664296513 | -10.444605952629407 | 0.0 | 1000 |
| SSC → SHD | size | 0.6156863117107885 | 0.620830074784912 | 0.672683074248295 | 0.6176346549932662 | 0.44181895004193084 | 0.5817351462751781 | -4.234613882302911 | 0.961026651295563 | 1000 |
| SSC → SHD | event_structure | 0.14468585138549325 | 0.10299698390706924 | 0.31514606192067945 | 0.18965367626425308 | 0.07484543451876008 | 0.23966950617919677 | 0.5064355493376136 | 0.9174195494519449 | 1000 |
| SSC → SHD | historical_spike | 61.629432221811534 | 55.2600272422501 | 121.91967974404096 | 74.1659971011098 | 38.417393438108135 | 94.9725015881036 | -75478.84860422405 | -0.2488070128070128 | 1000 |
| SSC → SHD | diagnostic_oracle | 0.07266869005190471 | 0.05929990147773123 | 0.15310271414904225 | 0.09213180841660303 | 0.04276024294415422 | 0.09823532128187866 | 0.8835229045125901 | 0.9105340185340186 | 1000 |

Transfer was asymmetric. The event-structure model retained high rank correlation in both directions, but SHD-to-SSC transfer had negative R² and substantial scale error; SSC-to-SHD transfer had positive R² but was less accurate than within-dataset evaluation. Historical-spike transfer was accurate from SHD to SSC and failed strongly from SSC to SHD. These observations do not establish generalization beyond SHD and SSC, the evaluated LIF models, this GH200 system, or this execution.

## Spike-history decision

Historical spikes are adopted only if the untouched seed-27 evaluation meets every predeclared condition: median absolute runtime error improves by at least 5%, 90th-percentile runtime error does not worsen, rank correlation improves or remains within tolerance, both SHD and SSC benefit, prediction time is negligible beside client training, and offline assignment is closer to the measured oracle. If any condition fails, the recorded decision is `spike_history_not_adopted`, the strongest event/size model is exported, and the spike result remains valid negative evidence. The decision values cannot be altered after inspecting seed 27.

The [committed decision](../results/resource_measurement/cost_model_evaluation.json) records every condition:

| Dataset | Median runtime-error improvement fraction | Improvement ≥ 0.05 | P90 error did not worsen | Rank maintained within 0.01 |
|---|---:|---|---|---|
| SHD | 0.3331679722893203 | passed | passed | passed |
| SSC | -2.233354617899182 | failed | failed | passed |

| Global condition | Observed value | Result |
|---|---:|---|
| Benefit required on both datasets | SSC did not meet the error conditions | failed |
| Prediction time fraction | 0.000001492612371393861 | negligible; passed |
| Historical-spike assignment closer to measured oracle | true | passed |
| Diagnostic oracle export prohibited | no deployment-oriented oracle export | passed |
| Complete adoption rule | not every condition passed | failed |
| Final decision | `spike_history_not_adopted` | strongest non-spike `event_structure` selected |

Historical spike features helped runtime prediction on SHD but performed substantially worse on SSC under the predeclared adoption comparisons. They were therefore not sufficiently robust across both evaluated datasets and were not adopted. This is valid negative evidence, not an execution failure. It does not show that historical-spike prediction is generally beneficial, generally harmful, disproven, or without use; the observed behavior is dataset-dependent.

## Deterministic offline assignment evaluation

No production scheduler is introduced. For the already selected clients in every measured round, the analysis compares round-robin, example-count longest-first, predicted-cost longest-first, and measured-cost oracle assignment for two and four candidate processes. Every client is assigned once. Reports include predicted and measured process loads, predicted and measured makespan, oracle makespan, regret, load imbalance, and assignment-computation time.

The [assignment evidence](../results/resource_measurement/assignment_readiness.json) contains 9,600 offline records and marks every record `offline_evaluation_only: true`. The table below gives arithmetic means across the 100 untouched seed-27 rounds for the exported `event_structure` cost model. Makespan, regret, and assignment-computation values are seconds.

| Dataset | Processes | Assignment | Predicted makespan | Measured makespan | Oracle makespan | Regret | Load imbalance | Assignment time |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| SHD | 2 | round robin | 7.103745537475048 | 7.476756084730002 | 7.210967868969999 | 0.26578821576 | 0.0849254453255951 | 0.0000009283 |
| SHD | 2 | example-count longest first | 6.876413430164928 | 7.240618597809999 | 7.210967868969999 | 0.029650728840000005 | 0.024789627176986122 | 0.0000061018500000000016 |
| SHD | 2 | predicted-cost longest first | 6.861767834803338 | 7.224955433929997 | 7.210967868969999 | 0.013987564960000034 | 0.02052954436011147 | 0.000006106290000000002 |
| SHD | 2 | measured-cost oracle | 6.8677414002789785 | 7.210967868969999 | 7.210967868969999 | 0 | 0.016772703624540853 | 0.000005912119999999999 |
| SHD | 4 | round robin | 4.326563347317584 | 4.558922448439999 | 4.078251417870001 | 0.48067103057000005 | 0.40819542731105196 | 0.0000011596499999999996 |
| SHD | 4 | example-count longest first | 3.896530679570373 | 4.09218089578 | 4.078251417870001 | 0.013929477909999996 | 0.2612843378014937 | 0.000007781150000000001 |
| SHD | 4 | predicted-cost longest first | 3.8887684601074217 | 4.085179624940001 | 4.078251417870001 | 0.006928207070000072 | 0.2599759455676595 | 0.000007772190000000003 |
| SHD | 4 | measured-cost oracle | 3.9002868085700038 | 4.078251417870001 | 4.078251417870001 | 0 | 0.25637778148692936 | 0.00046193204 |
| SSC | 2 | round robin | 12.454054778210045 | 12.799691978829996 | 12.431727338830001 | 0.36796464000000007 | 0.06506286641653851 | 0.0000013670100000000003 |
| SSC | 2 | example-count longest first | 12.110820981386519 | 12.456027668200004 | 12.431727338830001 | 0.02430032936999993 | 0.012603888268670309 | 0.000006262189999999999 |
| SSC | 2 | predicted-cost longest first | 12.107695389392118 | 12.44817860924 | 12.431727338830001 | 0.0164512704099997 | 0.011378062234823998 | 0.000006027600000000001 |
| SSC | 2 | measured-cost oracle | 12.118747042174613 | 12.431727338830001 | 12.431727338830001 | 0 | 0.008761286914255007 | 0.000006103140000000005 |
| SSC | 4 | round robin | 7.520585917951505 | 7.741922843169999 | 7.18215947396 | 0.5597633692100001 | 0.39546273119378783 | 0.0000011285900000000004 |
| SSC | 4 | example-count longest first | 6.999521461465747 | 7.21387283932 | 7.18215947396 | 0.03171336536 | 0.28377212582022737 | 0.000008072990000000003 |
| SSC | 4 | predicted-cost longest first | 6.994234892565401 | 7.19226245761 | 7.18215947396 | 0.010102983650000032 | 0.2797790741581492 | 0.000008113930000000002 |
| SSC | 4 | measured-cost oracle | 7.000689596889194 | 7.18215947396 | 7.18215947396 | 0 | 0.27666603272713625 | 0.000007757160000000001 |

Relative to round robin, predicted-cost longest-first reduced the mean measured offline makespan by fractions `0.033677793945192915` (SHD, two processes), `0.1039155258414422` (SHD, four), `0.027462642864483054` (SSC, two), and `0.07099791572385859` (SSC, four). In every displayed setting it also had lower regret and load imbalance than round robin, and regret remained positive relative to the measured-cost oracle.

This assignment evaluation is offline. No production scheduling policy changed. The measured-cost oracle uses costs unavailable before execution, and the diagnostic current-execution oracle cannot be deployed. The observed offline differences do not establish an end-to-end distributed speedup or an end-to-end latency result from predicted assignments.

## Interruption, resumption, and acceptance

Every power sample, interval, idle reference, and client record carries an execution-attempt identity. An interval interrupted during a client is retained in attempt records but is not accepted. The established training rule repeats an incomplete round; accepted prior rounds remain usable. An interrupted attempt uses only its accepted 30-second pre-execution idle samples for its attempt-local baseline, allowing immediate sampler shutdown. The completed resumed execution must still contain a 30-second post-evaluation idle interval. Exclusion records state why interrupted clients or rounds were omitted. Resumed accepted records keep their original attempt and GPU UUID, and samples are filtered by attempt so traces from separate attempts are never integrated together.

Each scientific run writes `measurement_config.json`, `measurement_acceptance.json`, `idle_power.json`, `device_samples.jsonl`, `execution_intervals.jsonl`, `client_resource_records.jsonl`, `excluded_intervals.jsonl`, and `calibration_reference.json`, beside established resolved configuration, Git provenance, checkpoints, and federated records. Raw power samples are not stored in `final_metrics.json`.

The predeclared collection gate required six completed compatible executions, exactly 6,000 accepted client records, three seeds per dataset, passing attempt-specific calibration, complete timing and energy coverage, no leakage, one official-test access per execution, finite metrics, model JSON prediction reproduction, and complete Git, configuration, hardware, Slurm, and input-hash provenance. The committed summary reports every gate satisfied and `valid: true`. Execution completion, measurement completeness, energy completeness, and the scientific hypothesis outcome remain distinct. Accuracy was not a completion gate, and the valid collection did not predetermine the spike-history decision.

## Authoritative artifacts and figures

Collection, model, and provenance records:

- [resource summary JSON](../results/resource_measurement/resource_measurement_summary.json), [CSV](../results/resource_measurement/resource_measurement_summary.csv), and [Markdown](../results/resource_measurement/resource_measurement_summary.md);
- [cost evaluation JSON](../results/resource_measurement/cost_model_evaluation.json), [CSV](../results/resource_measurement/cost_model_evaluation.csv), and [Markdown](../results/resource_measurement/cost_model_evaluation.md);
- [scheduling model](../results/resource_measurement/client_cost_model.json);
- [gross-energy model](../results/resource_measurement/energy_cost_model.json);
- [offline assignment evidence](../results/resource_measurement/assignment_readiness.json);
- [instrumentation calibration](../results/resource_measurement/instrumentation_calibration.json);
- [Slurm accounting](../results/resource_measurement/provenance/slurm-accounting.txt);
- [execution commit](../results/resource_measurement/provenance/execution-commit.txt);
- [Slurm job ID](../results/resource_measurement/provenance/slurm-job-id.txt);
- [accounting SHA-256](../results/resource_measurement/provenance/accounting-sha256.json);
- [preserved input SHA-256 records](../results/resource_measurement/provenance/run-input-sha256.txt).

Every figure is paired with its traceable source table:

- [predicted versus measured runtime](../results/resource_measurement/predicted_versus_measured_runtime.png) and [source CSV](../results/resource_measurement/source_predicted_versus_measured_runtime.csv);
- [predicted versus measured gross energy](../results/resource_measurement/predicted_versus_measured_gross_energy.png) and [source CSV](../results/resource_measurement/source_predicted_versus_measured_gross_energy.csv);
- [feature ablation comparison](../results/resource_measurement/feature_ablation_comparison.png) and [source CSV](../results/resource_measurement/source_feature_ablation_comparison.csv);
- [error distribution by model](../results/resource_measurement/error_distribution_by_model.png) and [source CSV](../results/resource_measurement/source_error_distribution_by_model.csv);
- [cross-dataset transfer comparison](../results/resource_measurement/cross_dataset_transfer_comparison.png) and [source CSV](../results/resource_measurement/source_cross_dataset_transfer_comparison.csv);
- [observed load versus predicted assignment load](../results/resource_measurement/observed_load_versus_predicted_assignment_load.png) and [source CSV](../results/resource_measurement/source_observed_load_versus_predicted_assignment_load.csv);
- [representative power trace](../results/resource_measurement/representative_power_trace.png) and [source CSV](../results/resource_measurement/source_representative_power_trace.csv);
- [runtime residual versus example count](../results/resource_measurement/runtime_residual_versus_example_count.png) and [source CSV](../results/resource_measurement/source_runtime_residual_versus_example_count.csv);
- [runtime residual versus input event count](../results/resource_measurement/runtime_residual_versus_input_event_count.png) and [source CSV](../results/resource_measurement/source_runtime_residual_versus_input_event_count.csv);
- [runtime residual versus padded timesteps](../results/resource_measurement/runtime_residual_versus_padded_timesteps.png) and [source CSV](../results/resource_measurement/source_runtime_residual_versus_padded_timesteps.csv).

Submission, accounting, fitting, and summary-regeneration commands are in the [reproducibility guide](../docs/reproducibility.md#client-resource-measurement-and-cost-estimation). Roihu allocation and interpreter details are in the [environment guide](../environment/roihu/README.md#client-resource-measurement-allocation).

## Limitations

- The scientific matrix has three seeds.
- It covers two datasets, SHD and SSC, and two LIF model sizes, 256/256 and 128/128.
- It used one GH200 system, one physical GPU, and one process.
- Power was sampled every 100 ms.
- NVML reports device-level rather than component-level power and energy.
- Accepted client-training energy does not represent whole-allocation energy.
- Idle-adjusted energy depends on the accepted idle baseline and its measurement conditions.
- No direct external power meter was used.
- No multinode resource measurement was performed.
- No production scheduling result exists.
- Predicted assignments have no end-to-end distributed latency result.
- The descriptive evidence supports neither a causal claim nor a statistical-significance claim.
- The two directed transfer evaluations do not establish universal cross-dataset or cross-system behavior.
- The measurements do not establish energy efficiency, reduced billing, reduced physical network traffic, production scheduling superiority, or multinode scalability.
- These measurements alone do not establish a thesis-novelty claim.

## Conclusions

The measurement and client-cost-estimation system operated successfully under its declared consistency gates. Event-structure features were selected for deployment-oriented wall-time and gross-energy prediction, with a ridge scheduling model exported for `client_wall_time_seconds`.

Causal spike history was beneficial in the declared SHD runtime comparison but failed the two-dataset adoption rule because the SSC median and tail-error conditions failed. The resulting `spike_history_not_adopted` decision is dataset-dependent negative evidence and motivates non-spike event-structure cost estimation for subsequent scheduling experiments.

The offline assignment records show lower measured makespan, regret, and load imbalance for predicted-cost longest-first than round robin in the displayed SHD and SSC seed-27 aggregates. They do not constitute a deployed scheduling result, and no production scheduling or end-to-end latency result exists.
