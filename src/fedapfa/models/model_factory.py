"""Strict dataset/model factory."""

from .shd_dcls import DCLSSHDSNN
from .shd_lif import AudioLIFSNN
from .svgg9_bntt import SVGG9BNTT


def make_model(config):
    dataset = config["dataset"]
    model = config["model"]
    key = (dataset["name"], model["name"])
    if key == ("shd", "dcls_shd"):
        return DCLSSHDSNN(config)
    if key == ("cifar10", "svgg9_bntt"):
        return SVGG9BNTT(config)
    allowed = {("shd", "lif_2layer"), ("ssc", "lif_2layer_128"), ("ssc", "lif_2layer_512")}
    if key not in allowed:
        raise ValueError(f"unsupported dataset/model combination: {key}")
    return AudioLIFSNN(
        dataset["input_features"],
        model["hidden_dims"],
        dataset["classes"],
        model["neuron"],
        model["attention"],
        model["dropout"],
        model["batch_normalization"],
        model["bias"],
    )


def make_audio_snn(dataset, attention="none", lambda_=1e-2, dropout=0.4):
    raise TypeError("make_audio_snn legacy API removed; pass a completely resolved config to make_model")
