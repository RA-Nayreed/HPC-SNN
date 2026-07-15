from pathlib import Path
import h5py
from torch.utils.data import Dataset
from .temporal_binning import bin_events

EXPECTED={"shd_train.h5":(8156,20),"shd_test.h5":(2264,20),"ssc_train.h5":(75466,35),"ssc_valid.h5":(9981,35),"ssc_test.h5":(20382,35)}
class EventAudioDataset(Dataset):
    def __init__(self, path, indices=None):
        self.path=Path(path)
        if not self.path.is_file(): raise FileNotFoundError(self.path)
        with h5py.File(self.path,"r") as file: self.size=len(file["labels"])
        self.indices=list(range(self.size)) if indices is None else list(indices)
        self._file=None
    def _handle(self):
        if self._file is None: self._file=h5py.File(self.path,"r")
        return self._file
    def __len__(self): return len(self.indices)
    def __getitem__(self, item):
        file=self._handle(); index=self.indices[item]
        return bin_events(file["spikes/times"][index], file["spikes/units"][index]), int(file["labels"][index])
    def __getstate__(self):
        state=self.__dict__.copy(); state["_file"]=None; return state
    def metadata(self):
        with h5py.File(self.path,"r") as file:
            labels=file["labels"][:]; return {"examples":len(labels),"classes":int(labels.max())+1,"channels":700}
