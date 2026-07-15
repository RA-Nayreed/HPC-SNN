import numpy as np
from fedapfa.datasets.temporal_binning import bin_events,pad_batch
def test_binning_padding_and_mask_lengths():
    value=bin_events(np.array([.001,.011]),np.array([0,699])); assert value.shape==(2,140); assert value[0,0]==1 and value[1,139]==1
    batch,labels,lengths=pad_batch([(value,2),(value[:1],3)]); assert batch.shape==(2,2,140) and lengths.tolist()==[2,1] and labels.tolist()==[2,3]
