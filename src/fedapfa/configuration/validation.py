from collections.abc import Mapping
REQUIRED=("name","seed","dataset","model","attention","training","device")
def validate_config(config: Mapping[str,object]):
    missing=[key for key in REQUIRED if key not in config]
    if missing: raise ValueError("Missing required settings: "+", ".join(missing))
    if config["dataset"] not in {"shd","ssc"}: raise ValueError("invalid dataset")
    if config["attention"] not in {"none","equation","public_behavior"}: raise ValueError("invalid attention variant")
    if config["device"] not in {"cpu","cuda"}: raise ValueError("invalid device")
    training=config["training"]
    if not isinstance(training,Mapping) or training.get("batch_size",0)<=0 or training.get("epochs",0)<=0: raise ValueError("batch_size and epochs must be positive")
    if config.get("lambda",.01)<=0: raise ValueError("lambda must be positive")
    if config.get("input_features",140)!=140 or config.get("classes") not in {20,35}: raise ValueError("invalid feature or class count")
