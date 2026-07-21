# Scheduling Evaluation

Evidence status: **valid**

Evidence complete: **True**

Datasets are summarized separately; three seeds do not establish statistical significance.

Decision: `event_structure_scheduler_not_adopted`

## Grouped measurements

### SHD — event_structure_longest_processing_time

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.705535924617; sample SD 0.00655637340348; range [0.69832155477, 0.711130742049]
- official_test_macro_f1: mean 0.694748868287; sample SD 0.00728841461958; range [0.690481212617, 0.703164521015]
- total_runtime_seconds: mean 579.13392729; sample SD 23.8670484819; range [565.332310846, 606.693209339]
- mean_round_time_seconds: mean 5.7913392729; sample SD 0.238670484819; range [5.65332310846, 6.06693209339]
- client_wall_time_seconds: mean 439.639394668; sample SD 23.5320868969; range [422.171139166, 466.398620227]
- scheduler_overhead_seconds: mean 9.92840231346; sample SD 0.175326597619; range [9.75850945106, 10.1086992943]
- scheduler_overhead_fraction: mean 0.0171552754939; sample SD 0.00045011719656; range [0.0166619621559, 0.017543660613]
- predicted_load_imbalance: mean 0.226736064415; sample SD 0.032874914199; range [0.199436849694, 0.263229153128]
- observed_load_imbalance: mean 0.27112430032; sample SD 0.0286345581758; range [0.252433513427, 0.304090209974]
- aggregation_time_seconds: mean 4.14984638748; sample SD 0.12047905944; range [4.01462472801, 4.24576853774]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 4.14984638748; sample SD 0.12047905944; range [4.01462472801, 4.24576853774]
- logical_intra_node_bytes: mean 336781440; sample SD 3564744.77745; range [333927360, 340777152]
- logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- predicted_logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- client_result_collection_bytes: mean 336781440; sample SD 3564744.77745; range [333927360, 340777152]
- model_distribution_bytes: mean 128433600; sample SD 0; range [128433600, 128433600]
- model_sized_payloads_crossing_node_boundaries: mean 0; sample SD 0; range [0, 0]
- maximum_peak_allocated_bytes: mean 116810240; sample SD 244106.974059; range [116529152, 116968960]
- maximum_peak_reserved_bytes: mean 127926272; sample SD 0; range [127926272, 127926272]
- gpu_utilization_percent: mean 38.9007521393; sample SD 1.29477670905; range [37.9172794118, 40.3676975945]
- internal_treatment_duration_seconds: mean 591.676977549; sample SD 23.9514312971; range [577.746597084, 619.333456679]
- derived_treatment_gpu_exposure_hours: mean 0.657418863943; sample SD 0.0266127014412; range [0.641940663427, 0.688148285199]
- paired_speedup: mean 1.05986496633; sample SD 0.0599615515393; range [0.993241297025, 1.10949711591]
- paired_runtime_reduction: mean 0.054419072815; sample SD 0.0547526483779; range [-0.00680469388026, 0.0986907620959]
- maximum_absolute_parameter_difference: mean 0; sample SD 0; range [0, 0]
- maximum_relative_parameter_difference: mean 0; sample SD 0; range [0, 0]
- runtime_regression_fraction: mean -0.054419072815; sample SD 0.0547526483779; range [-0.0986907620959, 0.00680469388026]

### SHD — example_count_longest_processing_time

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.705535924617; sample SD 0.00655637340348; range [0.69832155477, 0.711130742049]
- official_test_macro_f1: mean 0.694748868287; sample SD 0.00728841461958; range [0.690481212617, 0.703164521015]
- total_runtime_seconds: mean 566.864926091; sample SD 4.5461556607; range [561.851052168, 570.718463404]
- mean_round_time_seconds: mean 5.66864926091; sample SD 0.045461556607; range [5.61851052168, 5.70718463404]
- client_wall_time_seconds: mean 438.300677735; sample SD 7.66595655608; range [431.632083743, 446.676232565]
- scheduler_overhead_seconds: mean 1.35347069866; sample SD 0.0111063057643; range [1.34680417669, 1.36629174871]
- scheduler_overhead_fraction: mean 0.00238772379124; sample SD 2.44255560925e-05; range [0.00235983985634, 0.00240533623842]
- predicted_load_imbalance: mean 0.202340688404; sample SD 0.0340748809044; range [0.177338223028, 0.241152661342]
- observed_load_imbalance: mean 0.270107479951; sample SD 0.0366491992926; range [0.244010403246, 0.312006854503]
- aggregation_time_seconds: mean 4.16362680708; sample SD 0.154789048573; range [4.03826421173, 4.33663858252]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 4.16362680708; sample SD 0.154789048573; range [4.03826421173, 4.33663858252]
- logical_intra_node_bytes: mean 334783584; sample SD 7610285.37677; range [326221344, 340777152]
- logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- predicted_logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- client_result_collection_bytes: mean 334783584; sample SD 7610285.37677; range [326221344, 340777152]
- model_distribution_bytes: mean 128433600; sample SD 0; range [128433600, 128433600]
- model_sized_payloads_crossing_node_boundaries: mean 0; sample SD 0; range [0, 0]
- maximum_peak_allocated_bytes: mean 116798122.667; sample SD 223251.571008; range [116540416, 116932608]
- maximum_peak_reserved_bytes: mean 127926272; sample SD 0; range [127926272, 127926272]
- gpu_utilization_percent: mean 38.7161976547; sample SD 0.732269764848; range [38.0557620818, 39.5036764706]
- internal_treatment_duration_seconds: mean 575.242388709; sample SD 4.63701423365; range [570.09432668, 579.091172254]
- derived_treatment_gpu_exposure_hours: mean 0.639158209676; sample SD 0.00515223803738; range [0.633438140756, 0.643434635838]
- paired_speedup: mean 1.08135300986; sample SD 0.0314246507288; range [1.05584940523, 1.1164584089]
- paired_runtime_reduction: mean 0.0747178149434; sample SD 0.0265737786011; range [0.0528952376695, 0.104310566317]
- maximum_absolute_parameter_difference: mean 0; sample SD 0; range [0, 0]
- maximum_relative_parameter_difference: mean 0; sample SD 0; range [0, 0]
- runtime_regression_fraction: mean -0.0747178149434; sample SD 0.0265737786011; range [-0.104310566317, -0.0528952376695]

### SHD — round_robin

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.705535924617; sample SD 0.00655637340348; range [0.69832155477, 0.711130742049]
- official_test_macro_f1: mean 0.694748868287; sample SD 0.00728841461958; range [0.690481212617, 0.703164521015]
- total_runtime_seconds: mean 612.885949195; sample SD 12.846767943; range [602.59275014, 627.283331743]
- mean_round_time_seconds: mean 6.12885949195; sample SD 0.12846767943; range [6.0259275014, 6.27283331743]
- client_wall_time_seconds: mean 487.284198163; sample SD 10.9784874258; range [477.750531246, 499.286989229]
- scheduler_overhead_seconds: mean 0.0538718093497; sample SD 0.00382194288341; range [0.0514474647352, 0.0582775943913]
- scheduler_overhead_fraction: mean 8.79521225332e-05; sample SD 7.03875667024e-06; range [8.20163108627e-05, 9.57282193299e-05]
- predicted_load_imbalance: mean 0.333333333333; sample SD 0; range [0.333333333333, 0.333333333333]
- observed_load_imbalance: mean 0.409667735365; sample SD 0.028365538854; range [0.392876080826, 0.442417906146]
- aggregation_time_seconds: mean 4.32976979033; sample SD 0.303042871306; range [4.14643248927, 4.67955771054]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 4.32976979033; sample SD 0.303042871306; range [4.14643248927, 4.67955771054]
- logical_intra_node_bytes: mean 299678400; sample SD 0; range [299678400, 299678400]
- logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- predicted_logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- client_result_collection_bytes: mean 299678400; sample SD 0; range [299678400, 299678400]
- model_distribution_bytes: mean 128433600; sample SD 0; range [128433600, 128433600]
- model_sized_payloads_crossing_node_boundaries: mean 0; sample SD 0; range [0, 0]
- maximum_peak_allocated_bytes: mean 116913493.333; sample SD 45080.2359059; range [116867584, 116957696]
- maximum_peak_reserved_bytes: mean 128625322.667; sample SD 1210791.27173; range [127926272, 130023424]
- gpu_utilization_percent: mean 43.0092115257; sample SD 0.619155883722; range [42.3968531469, 43.6349480969]
- internal_treatment_duration_seconds: mean 620.837362035; sample SD 12.8745435435; range [610.502822388, 635.259459743]
- derived_treatment_gpu_exposure_hours: mean 0.68981929115; sample SD 0.0143050483817; range [0.67833646932, 0.705843844159]

### SSC — event_structure_longest_processing_time

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.437722827331; sample SD 0.00880051284571; range [0.427779413208, 0.444509861643]
- official_test_macro_f1: mean 0.410934690352; sample SD 0.0110407289204; range [0.404286621057, 0.423679450878]
- total_runtime_seconds: mean 1254.67224265; sample SD 21.3439959943; range [1240.19154141, 1279.18390283]
- mean_round_time_seconds: mean 12.5467224265; sample SD 0.213439959943; range [12.4019154141, 12.7918390283]
- client_wall_time_seconds: mean 765.981493054; sample SD 9.55632796235; range [759.30355196, 776.928172269]
- scheduler_overhead_seconds: mean 110.454778605; sample SD 2.56407537117; range [108.312502049, 113.295795658]
- scheduler_overhead_fraction: mean 0.0880303801514; sample SD 0.00087305139133; range [0.0870230671804, 0.0885688096977]
- predicted_load_imbalance: mean 0.27307573004; sample SD 0.00626474306541; range [0.266557251307, 0.279051415969]
- observed_load_imbalance: mean 0.29235996663; sample SD 0.00895867474735; range [0.282794753793, 0.300554006878]
- aggregation_time_seconds: mean 1.58703841316; sample SD 0.00951527609189; range [1.57732071134, 1.59633744229]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 1.58703841316; sample SD 0.00951527609189; range [1.57732071134, 1.59633744229]
- logical_intra_node_bytes: mean 122278700; sample SD 2616955.56516; range [119256900, 123789600]
- logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- predicted_logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- client_result_collection_bytes: mean 122278700; sample SD 2616955.56516; range [119256900, 123789600]
- model_distribution_bytes: mean 46890000; sample SD 0; range [46890000, 46890000]
- model_sized_payloads_crossing_node_boundaries: mean 0; sample SD 0; range [0, 0]
- maximum_peak_allocated_bytes: mean 208718336; sample SD 0; range [208718336, 208718336]
- maximum_peak_reserved_bytes: mean 276824064; sample SD 0; range [276824064, 276824064]
- gpu_utilization_percent: mean 45.6465328297; sample SD 0.213570017286; range [45.4467005076, 45.8715986395]
- internal_treatment_duration_seconds: mean 1276.985224; sample SD 21.775449538; range [1262.45260909, 1302.02152106]
- derived_treatment_gpu_exposure_hours: mean 1.41887247112; sample SD 0.0241949439311; range [1.40272512121, 1.44669057896]
- paired_speedup: mean 0.94738085517; sample SD 0.012978716499; range [0.932492674295, 0.956309410032]
- paired_runtime_reduction: mean -0.0556747784529; sample SD 0.0145709516846; range [-0.0723944836953, -0.0456866674211]
- maximum_absolute_parameter_difference: mean 0; sample SD 0; range [0, 0]
- maximum_relative_parameter_difference: mean 0; sample SD 0; range [0, 0]
- runtime_regression_fraction: mean 0.0556747784529; sample SD 0.0145709516846; range [0.0456866674211, 0.0723944836953]

### SSC — example_count_longest_processing_time

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.437722827331; sample SD 0.00880051284571; range [0.427779413208, 0.444509861643]
- official_test_macro_f1: mean 0.410934690352; sample SD 0.0110407289204; range [0.404286621057, 0.423679450878]
- total_runtime_seconds: mean 1130.67622401; sample SD 20.4669919172; range [1116.51689739, 1154.1428596]
- mean_round_time_seconds: mean 11.3067622401; sample SD 0.204669919172; range [11.1651689739, 11.541428596]
- client_wall_time_seconds: mean 756.202764922; sample SD 10.4887472722; range [748.259500082, 768.092239668]
- scheduler_overhead_seconds: mean 1.76479627134; sample SD 0.107986380706; range [1.64110529912, 1.84029701143]
- scheduler_overhead_fraction: mean 0.00156037319976; sample SD 7.91860942044e-05; range [0.00146984367451, 0.00161676186949]
- predicted_load_imbalance: mean 0.237790489441; sample SD 0.0137612147834; range [0.224354772919, 0.251855603889]
- observed_load_imbalance: mean 0.295640073151; sample SD 0.00967165867551; range [0.285483233717, 0.304739679394]
- aggregation_time_seconds: mean 1.58209214445; sample SD 0.0102592191304; range [1.57490665431, 1.59384136705]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 1.58209214445; sample SD 0.0102592191304; range [1.57490665431, 1.59384136705]
- logical_intra_node_bytes: mean 122174500; sample SD 2663222.35835; range [119100600, 123789600]
- logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- predicted_logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- client_result_collection_bytes: mean 122174500; sample SD 2663222.35835; range [119100600, 123789600]
- model_distribution_bytes: mean 46890000; sample SD 0; range [46890000, 46890000]
- model_sized_payloads_crossing_node_boundaries: mean 0; sample SD 0; range [0, 0]
- maximum_peak_allocated_bytes: mean 208718336; sample SD 0; range [208718336, 208718336]
- maximum_peak_reserved_bytes: mean 276824064; sample SD 0; range [276824064, 276824064]
- gpu_utilization_percent: mean 42.7799621491; sample SD 0.343841447896; range [42.3886278195, 43.0336812144]
- internal_treatment_duration_seconds: mean 1148.63046402; sample SD 20.5979029273; range [1134.73593058, 1172.29538788]
- derived_treatment_gpu_exposure_hours: mean 1.27625607113; sample SD 0.0228865588082; range [1.26081770064, 1.30255043098]
- paired_speedup: mean 1.05128056197; sample SD 0.013450374879; range [1.03578327367, 1.05991697064]
- paired_runtime_reduction: mean 0.0486745654587; sample SD 0.0122603520351; range [0.0345470665304, 0.0565298719635]
- maximum_absolute_parameter_difference: mean 0; sample SD 0; range [0, 0]
- maximum_relative_parameter_difference: mean 0; sample SD 0; range [0, 0]
- runtime_regression_fraction: mean -0.0486745654587; sample SD 0.0122603520351; range [-0.0565298719635, -0.0345470665304]

### SSC — round_robin

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.437722827331; sample SD 0.00880051284571; range [0.427779413208, 0.444509861643]
- official_test_macro_f1: mean 0.410934690352; sample SD 0.0110407289204; range [0.404286621057, 0.423679450878]
- total_runtime_seconds: mean 1188.77735028; sample SD 33.4678296351; range [1156.46952708, 1223.29560343]
- mean_round_time_seconds: mean 11.8877735028; sample SD 0.334678296351; range [11.5646952708, 12.2329560343]
- client_wall_time_seconds: mean 814.91463394; sample SD 25.3328263902; range [789.074475329, 839.707654967]
- scheduler_overhead_seconds: mean 0.0546764716661; sample SD 0.00233372171831; range [0.0521775542293, 0.0567993236473]
- scheduler_overhead_fraction: mean 4.60304738949e-05; sample SD 2.71996306161e-06; range [4.39735453058e-05, 4.91144144459e-05]
- predicted_load_imbalance: mean 0.333333333333; sample SD 0; range [0.333333333333, 0.333333333333]
- observed_load_imbalance: mean 0.407121985502; sample SD 0.0253203992652; range [0.378133413661, 0.424913186029]
- aggregation_time_seconds: mean 1.66820642251; sample SD 0.0289765215119; range [1.6494143653, 1.70157707809]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 1.66820642251; sample SD 0.0289765215119; range [1.6494143653, 1.70157707809]
- logical_intra_node_bytes: mean 109410000; sample SD 0; range [109410000, 109410000]
- logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- predicted_logical_inter_node_bytes: mean 0; sample SD 0; range [0, 0]
- client_result_collection_bytes: mean 109410000; sample SD 0; range [109410000, 109410000]
- model_distribution_bytes: mean 46890000; sample SD 0; range [46890000, 46890000]
- model_sized_payloads_crossing_node_boundaries: mean 0; sample SD 0; range [0, 0]
- maximum_peak_allocated_bytes: mean 208718336; sample SD 0; range [208718336, 208718336]
- maximum_peak_reserved_bytes: mean 276824064; sample SD 0; range [276824064, 276824064]
- gpu_utilization_percent: mean 45.8211983301; sample SD 0.360587012377; range [45.4508196721, 46.1711229947]
- internal_treatment_duration_seconds: mean 1206.48493504; sample SD 33.2845038754; range [1174.60446667, 1241.01508587]
- derived_treatment_gpu_exposure_hours: mean 1.34053881671; sample SD 0.0369827820838; range [1.30511607408, 1.37890565096]

## Paired comparisons

| Dataset | Seed | Reference | Treatment | Speedup | Runtime reduction | Execution structure | Model-state structure | Mathematical | Bitwise | Maximum absolute difference | Prediction identity | Checkpoint identity | First divergence |
|---|---:|---|---|---:|---:|---|---|---|---|---:|---|---|---|
| SHD | 37 | round_robin | example_count_longest_processing_time | 1.05584940523 | 0.0528952376695 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SHD | 37 | round_robin | event_structure_longest_processing_time | 0.993241297025 | -0.00680469388026 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SHD | 47 | round_robin | example_count_longest_processing_time | 1.1164584089 | 0.104310566317 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SHD | 47 | round_robin | event_structure_longest_processing_time | 1.10949711591 | 0.0986907620959 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SHD | 57 | round_robin | example_count_longest_processing_time | 1.07175121545 | 0.0669476408435 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SHD | 57 | round_robin | event_structure_longest_processing_time | 1.07685648604 | 0.0713711502292 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SSC | 37 | round_robin | example_count_longest_processing_time | 1.0581414416 | 0.0549467578823 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SSC | 37 | round_robin | event_structure_longest_processing_time | 0.953340481184 | -0.0489431842424 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SSC | 47 | round_robin | example_count_longest_processing_time | 1.03578327367 | 0.0345470665304 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SSC | 47 | round_robin | event_structure_longest_processing_time | 0.932492674295 | -0.0723944836953 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SSC | 57 | round_robin | example_count_longest_processing_time | 1.05991697064 | 0.0565298719635 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |
| SSC | 57 | round_robin | event_structure_longest_processing_time | 0.956309410032 | -0.0456866674211 | True | True | True | True | 0 | True | True | no_recorded_trajectory_divergence |

## Decision conditions

- exact_dataset_seed_pair_coverage: True
- all_structural_and_scientific_equivalence: True
- scheduler_overhead_below_one_percent: False
- shd_runtime_improvement_at_least_five_percent: True
- ssc_runtime_improvement_at_least_five_percent: False
- not_slower_than_example_count_each_dataset: False
- two_of_three_seeds_improve_each_dataset: False
- no_dataset_seed_pair_more_than_two_percent_slower: False
- predictions_and_checkpoints_identical: True
- permitted_pre_execution_information_only: True
