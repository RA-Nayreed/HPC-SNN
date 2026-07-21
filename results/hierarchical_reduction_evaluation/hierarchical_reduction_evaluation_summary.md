# Hierarchical Reduction Evaluation

Evidence status: **valid**

Evidence complete: **True**

Datasets are summarized separately; three seeds do not establish statistical significance.

Decision: `node_hierarchical_reduction_not_retained`

## Grouped measurements

### SHD — flat_ordered

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.705535924617; sample SD 0.00655637340348; range [0.69832155477, 0.711130742049]
- official_test_macro_f1: mean 0.694748868287; sample SD 0.00728841461958; range [0.690481212617, 0.703164521015]
- total_runtime_seconds: mean 651.001961749; sample SD 14.9090311753; range [634.629703402, 663.796895329]
- mean_round_time_seconds: mean 6.51001961749; sample SD 0.149090311753; range [6.34629703402, 6.63796895329]
- client_wall_time_seconds: mean 467.993194164; sample SD 18.702618855; range [449.563592766, 486.95736901]
- scheduler_overhead_seconds: mean 10.3778370885; sample SD 0.0273141580648; range [10.3555144587, 10.4082944751]
- scheduler_overhead_fraction: mean 0.015946977879; sample SD 0.000371833598232; range [0.0156004261719, 0.0163397683346]
- predicted_load_imbalance: mean 0.226736064415; sample SD 0.032874914199; range [0.199436849694, 0.263229153128]
- observed_load_imbalance: mean 0.240750675537; sample SD 0.0368354232942; range [0.217348318319, 0.283210522302]
- aggregation_time_seconds: mean 27.1994720806; sample SD 0.672010914452; range [26.5081528181, 27.8503479091]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 27.1994720806; sample SD 0.672010914452; range [26.5081528181, 27.8503479091]
- logical_intra_node_bytes: mean 106742592; sample SD 4113738.28444; range [102318768, 110452896]
- logical_inter_node_bytes: mean 230038848; sample SD 7662287.93174; range [223474464, 238458384]
- predicted_logical_inter_node_bytes: mean 230038848; sample SD 7662287.93174; range [223474464, 238458384]
- client_result_collection_bytes: mean 336781440; sample SD 3564744.77745; range [333927360, 340777152]
- model_distribution_bytes: mean 128433600; sample SD 0; range [128433600, 128433600]
- model_sized_payloads_crossing_node_boundaries: mean 537.333333333; sample SD 17.8978583449; range [522, 557]
- maximum_peak_allocated_bytes: mean 116810240; sample SD 244106.974059; range [116529152, 116968960]
- maximum_peak_reserved_bytes: mean 127926272; sample SD 0; range [127926272, 127926272]
- gpu_utilization_percent: mean 39.7570468468; sample SD 0.498774470301; range [39.302891933, 40.2908545727]
- internal_treatment_duration_seconds: mean 664.040329901; sample SD 14.7033997987; range [647.890184356, 676.650770883]
- derived_treatment_gpu_exposure_hours: mean 0.737822588779; sample SD 0.0163371108875; range [0.719877982618, 0.75183418987]

### SHD — node_hierarchical

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.713780918728; sample SD 0.0360976807103; range [0.676236749117, 0.748233215548]
- official_test_macro_f1: mean 0.707435053673; sample SD 0.0392508548353; range [0.669831844894, 0.748148495558]
- total_runtime_seconds: mean 641.737328483; sample SD 11.4522262617; range [631.382461722, 654.037685218]
- mean_round_time_seconds: mean 6.41737328483; sample SD 0.114522262617; range [6.31382461722, 6.54037685218]
- client_wall_time_seconds: mean 465.826643689; sample SD 12.9992127893; range [455.685336792, 480.480874442]
- scheduler_overhead_seconds: mean 10.3493621511; sample SD 0.205856330935; range [10.1421401234, 10.5538250485]
- scheduler_overhead_fraction: mean 0.0161328807959; sample SD 0.00054434815846; range [0.0155069659633, 0.0164957169086]
- predicted_load_imbalance: mean 0.226736064415; sample SD 0.032874914199; range [0.199436849694, 0.263229153128]
- observed_load_imbalance: mean 0.249446037853; sample SD 0.0323794826597; range [0.218522622501, 0.283107681917]
- aggregation_time_seconds: mean 12.994381783; sample SD 0.550935454478; range [12.4331495987, 13.5344081707]
- node_local_reduction_time_seconds: mean 5.56450574915; sample SD 0.213512566816; range [5.33580683346, 5.75860619859]
- inter_node_movement_time_seconds: mean 4.52588812235; sample SD 0.294318683203; range [4.20188050275, 4.77670513]
- global_reduction_time_seconds: mean 2.51914619193; sample SD 0.0780145175252; range [2.44562969857, 2.60099056561]
- logical_intra_node_bytes: mean 220763088; sample SD 3006959.86629; range [217909008, 223902576]
- logical_inter_node_bytes: mean 85622400; sample SD 0; range [85622400, 85622400]
- predicted_logical_inter_node_bytes: mean 85622400; sample SD 0; range [85622400, 85622400]
- client_result_collection_bytes: mean 306385488; sample SD 3006959.86629; range [303531408, 309524976]
- model_distribution_bytes: mean 128433600; sample SD 0; range [128433600, 128433600]
- model_sized_payloads_crossing_node_boundaries: mean 100; sample SD 0; range [100, 100]
- maximum_peak_allocated_bytes: mean 116810240; sample SD 244106.974059; range [116529152, 116968960]
- maximum_peak_reserved_bytes: mean 127926272; sample SD 0; range [127926272, 127926272]
- gpu_utilization_percent: mean 39.4254382212; sample SD 0.809628044411; range [38.4913249211, 39.9252336449]
- internal_treatment_duration_seconds: mean 654.919901391; sample SD 11.3301475784; range [644.706814036, 667.107538145]
- derived_treatment_gpu_exposure_hours: mean 0.727688779323; sample SD 0.0125890528649; range [0.716340904484, 0.741230597939]
- paired_speedup: mean 1.01439248362; sample SD 0.00899659137259; range [1.00514306601, 1.02311290503]
- paired_runtime_reduction: mean 0.0141365398972; sample SD 0.00875072623268; range [0.00511675022965, 0.0225907667865]
- maximum_absolute_parameter_difference: mean 0.219044437011; sample SD 0.0590077808436; range [0.162821352482, 0.280489772558]
- maximum_relative_parameter_difference: mean 18479.9634847; sample SD 19280.9532529; range [5411.23806285, 40623.9857123]
- runtime_regression_fraction: mean -0.0141365398972; sample SD 0.00875072623268; range [-0.0225907667865, -0.00511675022965]

### SSC — flat_ordered

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.437722827331; sample SD 0.00880051284571; range [0.427779413208, 0.444509861643]
- official_test_macro_f1: mean 0.410934690352; sample SD 0.0110407289204; range [0.404286621057, 0.423679450878]
- total_runtime_seconds: mean 1286.49999901; sample SD 14.2105345303; range [1271.47264429, 1299.72097426]
- mean_round_time_seconds: mean 12.8649999901; sample SD 0.142105345303; range [12.7147264429, 12.9972097426]
- client_wall_time_seconds: mean 800.307727687; sample SD 13.0234875577; range [785.849216323, 811.118158459]
- scheduler_overhead_seconds: mean 109.656248898; sample SD 1.78597721012; range [108.503854306, 111.7135622]
- scheduler_overhead_fraction: mean 0.0852344308457; sample SD 0.00077401654115; range [0.0844141828399, 0.0859519576995]
- predicted_load_imbalance: mean 0.27307573004; sample SD 0.00626474306541; range [0.266557251307, 0.279051415969]
- observed_load_imbalance: mean 0.304701534312; sample SD 0.0109198411397; range [0.296641748108, 0.317129213746]
- aggregation_time_seconds: mean 1.32386333682; sample SD 0.18128301841; range [1.15817802015, 1.51749775768]
- node_local_reduction_time_seconds: mean 0; sample SD 0; range [0, 0]
- inter_node_movement_time_seconds: mean 0; sample SD 0; range [0, 0]
- global_reduction_time_seconds: mean 1.32386333682; sample SD 0.18128301841; range [1.15817802015, 1.51749775768]
- logical_intra_node_bytes: mean 38189300; sample SD 1063912.10633; range [37355700, 39387600]
- logical_inter_node_bytes: mean 84089400; sample SD 2049855.2827; range [81901200, 85965000]
- predicted_logical_inter_node_bytes: mean 84089400; sample SD 2049855.2827; range [81901200, 85965000]
- client_result_collection_bytes: mean 122278700; sample SD 2616955.56516; range [119256900, 123789600]
- model_distribution_bytes: mean 46890000; sample SD 0; range [46890000, 46890000]
- model_sized_payloads_crossing_node_boundaries: mean 538; sample SD 13.1148770486; range [524, 550]
- maximum_peak_allocated_bytes: mean 208718336; sample SD 0; range [208718336, 208718336]
- maximum_peak_reserved_bytes: mean 276824064; sample SD 0; range [276824064, 276824064]
- gpu_utilization_percent: mean 46.029825199; sample SD 0.0957709322813; range [45.919921875, 46.0954077593]
- internal_treatment_duration_seconds: mean 1308.27656633; sample SD 14.4770614431; range [1293.10504191, 1321.94130403]
- derived_treatment_gpu_exposure_hours: mean 1.45364062926; sample SD 0.0160856238257; range [1.4367833799, 1.46882367115]

### SSC — node_hierarchical

Seeds completed: 37, 47, 57

- official_test_accuracy: mean 0.436839695156; sample SD 0.0105187391535; range [0.424982827985, 0.445049553528]
- official_test_macro_f1: mean 0.409240711882; sample SD 0.0146680517653; range [0.400093604714, 0.426159287355]
- total_runtime_seconds: mean 1315.95339931; sample SD 8.20061259116; range [1309.29821804, 1325.11463962]
- mean_round_time_seconds: mean 13.1595339931; sample SD 0.0820061259116; range [13.0929821804, 13.2511463962]
- client_wall_time_seconds: mean 820.922574841; sample SD 4.46077994193; range [816.848145601, 825.688860267]
- scheduler_overhead_seconds: mean 110.536239951; sample SD 3.36245308528; range [108.278064221, 114.400574154]
- scheduler_overhead_fraction: mean 0.083989320155; sample SD 0.00206452935282; range [0.0824380703374, 0.0863325864294]
- predicted_load_imbalance: mean 0.27307573004; sample SD 0.00626474306541; range [0.266557251307, 0.279051415969]
- observed_load_imbalance: mean 0.310602004681; sample SD 0.0117571748277; range [0.29936087944, 0.322814910392]
- aggregation_time_seconds: mean 3.36221343283; sample SD 0.166612119285; range [3.16989469435, 3.46281413245]
- node_local_reduction_time_seconds: mean 1.69497048054; sample SD 0.0521784891307; range [1.64813187812, 1.75121056358]
- inter_node_movement_time_seconds: mean 1.14728492128; sample SD 0.126420580043; range [1.00302456925, 1.2387509754]
- global_reduction_time_seconds: mean 0.185205930068; sample SD 0.0101840695726; range [0.173453375814, 0.191433454165]
- logical_intra_node_bytes: mean 80129800; sample SD 1365576.68038; range [78618900, 81276000]
- logical_inter_node_bytes: mean 31260000; sample SD 0; range [31260000, 31260000]
- predicted_logical_inter_node_bytes: mean 31260000; sample SD 0; range [31260000, 31260000]
- client_result_collection_bytes: mean 111389800; sample SD 1365576.68038; range [109878900, 112536000]
- model_distribution_bytes: mean 46890000; sample SD 0; range [46890000, 46890000]
- model_sized_payloads_crossing_node_boundaries: mean 100; sample SD 0; range [100, 100]
- maximum_peak_allocated_bytes: mean 208718336; sample SD 0; range [208718336, 208718336]
- maximum_peak_reserved_bytes: mean 276824064; sample SD 0; range [276824064, 276824064]
- gpu_utilization_percent: mean 45.5533199053; sample SD 1.18581952055; range [44.3805104408, 46.7517347726]
- internal_treatment_duration_seconds: mean 1338.04652192; sample SD 8.76784107525; range [1331.04392608, 1347.88010029]
- derived_treatment_gpu_exposure_hours: mean 1.48671835769; sample SD 0.00974204563917; range [1.47893769564, 1.49764455588]
- paired_speedup: mean 0.977601821469; sample SD 0.00562205899575; range [0.971110039539, 0.98085879731]
- paired_runtime_reduction: mean -0.0229339801727; sample SD 0.0059023553002; range [-0.0297494200297, -0.0195147382508]
- maximum_absolute_parameter_difference: mean 0.398351088166; sample SD 0.148288024136; range [0.228151410818, 0.499680235982]
- maximum_relative_parameter_difference: mean 6673.63247324; sample SD 3512.20331341; range [4225.01668456, 10697.722976]
- runtime_regression_fraction: mean 0.0229339801727; sample SD 0.0059023553002; range [0.0195147382508, 0.0297494200297]

## Paired comparisons

| Dataset | Seed | Reference | Treatment | Speedup | Runtime reduction | Execution structure | Model-state structure | Mathematical | Bitwise | Maximum absolute difference | Prediction identity | Checkpoint identity | First divergence |
|---|---:|---|---|---:|---:|---|---|---|---|---:|---|---|---|
| SHD | 37 | flat_ordered | node_hierarchical | 1.0149214798 | 0.0147021026754 | True | True | False | False | 0.162821352482 | False | False | aggregation_grouping_first_divergence |
| SHD | 47 | flat_ordered | node_hierarchical | 1.00514306601 | 0.00511675022965 | True | True | False | False | 0.280489772558 | False | False | aggregation_grouping_first_divergence |
| SHD | 57 | flat_ordered | node_hierarchical | 1.02311290503 | 0.0225907667865 | True | True | False | False | 0.213822185993 | False | False | aggregation_grouping_first_divergence |
| SSC | 37 | flat_ordered | node_hierarchical | 0.971110039539 | -0.0297494200297 | True | True | False | False | 0.228151410818 | False | True | aggregation_grouping_first_divergence |
| SSC | 47 | flat_ordered | node_hierarchical | 0.98085879731 | -0.0195147382508 | True | True | False | False | 0.499680235982 | False | True | aggregation_grouping_first_divergence |
| SSC | 57 | flat_ordered | node_hierarchical | 0.98083662756 | -0.0195377822376 | True | True | False | False | 0.467221617699 | False | True | aggregation_grouping_first_divergence |

## Decision conditions

- every_update_once: True
- weights_and_denominators_correct: True
- structural_and_mathematical_equivalence: False
- parameter_differences_within_tolerance: False
- official_predictions_agree: False
- selected_checkpoints_agree: False
- logical_inter_node_movement_reduced: True
- no_material_runtime_regression: False
- official_test_ownership_preserved: True
