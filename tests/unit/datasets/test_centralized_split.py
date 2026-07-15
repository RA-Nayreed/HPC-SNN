import numpy as np
from fedapfa.datasets.centralized_split import stratified_split
def test_split_deterministic_and_stratified():
    labels=np.repeat(np.arange(2),10); a,b=stratified_split(labels,.2,7); c,d=stratified_split(labels,.2,7); assert np.array_equal(a,c) and np.array_equal(b,d) and len(b)==4
