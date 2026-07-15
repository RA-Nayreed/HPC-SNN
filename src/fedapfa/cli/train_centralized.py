import argparse, json, random, shutil, sys
from pathlib import Path
import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from fedapfa.configuration.loader import load_config
from fedapfa.configuration.experiment_id import experiment_id
from fedapfa.datasets import EventAudioDataset, pad_batch
from fedapfa.datasets.centralized_split import stratified_split
from fedapfa.models.model_factory import make_audio_snn
from fedapfa.training.centralized import train
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("config"); args=parser.parse_args(); config=load_config(args.config); random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    root=Path(config.get("data_root","data/raw"))/config["dataset"]; train_file=root/f"{config['dataset']}_train.h5"; test_file=root/f"{config['dataset']}_test.h5"; train_data=EventAudioDataset(train_file)
    if config["dataset"]=="ssc": validation_data=EventAudioDataset(root/"ssc_valid.h5")
    else:
        import h5py
        with h5py.File(train_file,"r") as f: train_indices,validation_indices=stratified_split(f["labels"][:],config.get("validation_fraction",.1),config["seed"])
        train_data=EventAudioDataset(train_file,train_indices); validation_data=EventAudioDataset(train_file,validation_indices)
    loader=lambda data,shuffle: DataLoader(data,batch_size=config["training"]["batch_size"],shuffle=shuffle,collate_fn=pad_batch,num_workers=0)
    run=Path(config.get("runs_root","runs"))/experiment_id(config)
    if run.exists(): raise FileExistsError(f"run exists: {run}")
    run.mkdir(parents=True); shutil.copy(args.config,run/"resolved_config.yaml"); (run/"command.txt").write_text(" ".join(sys.argv)); (run/"environment.json").write_text(json.dumps({"torch":torch.__version__,"cuda":torch.cuda.is_available()})); (run/"training.log").touch()
    model=make_audio_snn(config["dataset"],config["attention"],config["lambda"],config["dropout"]); train(model,loader(train_data,True),loader(validation_data,False),loader(EventAudioDataset(test_file),False),config,run)
