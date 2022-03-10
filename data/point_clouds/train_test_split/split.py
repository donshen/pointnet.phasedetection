import random
random.seed(166)
import numpy as np
from dataclasses import dataclass, field, astuple

@dataclass
class Params:
    train_ratio: float = field(
        default=0.8,
        metadata={
            "help": 'train data split ratio'
        },
    )
    num_points: int = field(
        default=3000,
        metadata={
            "help": 'number of point clouds for each phase'
        },
    )
    phases: tuple = field(
        default=('lam','hpc','hpl','bcc','dis','sg','dg','dd','p'),
        metadata={
            "help": 'a tuple of names of different possible phases'
        },
    )

def __main__():
    train_ratio, n_points, phases = astuple(Params())
    n_train = int(n_points * train_ratio)
    n_test = n_points - n_train
    n_class = len(phases)
    train_id, test_id = (np.zeros((n_train, n_class)),
                         np.zeros((n_test, n_class)))
    data_train, data_test, data_val = [], [], []
    for idx, cls in enumerate(phases):
        cls_idx_train = random.sample(range(1, n_points + 1), n_train)
        cls_idx_test = [i for i in range(1, n_points + 1) if i not in cls_idx_train]
        train_id[:, idx] = cls_idx_train
        test_id[:, idx] =  cls_idx_test 
        for idx_train, idx_test in zip(train_id[:, idx], test_id[:, idx]):
            data_train.append(f'"shape_data/{cls}/coord_O_{cls}_{int(idx_train)}"')
            data_test.append(f'"shape_data/{cls}/coord_O_{cls}_{int(idx_test)}"')
    
    ftrain = open('shuffled_train_file_list.json','w')
    ftrain.write('[')
    for k in range(len(data_train) - 1):
        ftrain.write(data_train[k] + ',')
    ftrain.write(data_train[-1] + ']')
    ftrain.close()

    ftest = open('shuffled_test_file_list.json','w')
    ftest.write('[')
    for k in range(len(data_test) - 1):
        ftest.write(data_test[k] + ',')
    ftest.write(data_test[-1] + ']')
    ftest.close()
    
if __name__ == "__main__":
    __main__()