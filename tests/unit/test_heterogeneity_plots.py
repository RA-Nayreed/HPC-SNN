from fedapfa.cli.summarize_heterogeneity import _plots


def test_heterogeneity_plot_writer_creates_five_png_files(tmp_path):
    treatments = []
    for index, label in enumerate(("IID", "Dirichlet 1.0", "Dirichlet 0.5", "Dirichlet 0.1")):
        treatments.append(
            {
                "label": label,
                "contextual_evidence": index == 2,
                "metrics": {
                    "official_test_accuracy": {"mean": 0.6 + index / 20},
                    "mean_update_alignment": {"mean": 0.2 + index / 10},
                },
                "runs": [
                    {"validation_curve": [[1, 0.4 + seed / 100], [2, 0.5 + seed / 100]]}
                    for seed in (7, 17, 27)
                ],
                "partition_statistics": {
                    "jensen_shannon_divergence_bits": {"mean": {"mean": index / 10}}
                },
                "mean_client_spike_rates": {"layer1": {"mean": 0.1 + index / 100}},
            }
        )
    _plots(tmp_path, treatments)
    names = (
        "accuracy_by_partition.png",
        "convergence_by_partition.png",
        "label_distribution_by_partition.png",
        "update_alignment_by_partition.png",
        "spike_rate_by_partition.png",
    )
    assert all((tmp_path / name).read_bytes().startswith(b"\x89PNG") for name in names)
