import numpy as np

def stratified_split(labels, validation_fraction=.1, seed=7):
    labels=np.asarray(labels); rng=np.random.default_rng(seed); train=[]; validation=[]
    for value in np.unique(labels):
        group=np.flatnonzero(labels==value); rng.shuffle(group); count=max(1, round(len(group)*validation_fraction)); validation.extend(group[:count]); train.extend(group[count:])
    return np.array(sorted(train)), np.array(sorted(validation))
