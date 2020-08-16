import argparse
from definitions import Topology

use_gpu = False
use_cnn = False

seed_count = 5
epoch_count = 1
topology = Topology.USANET

_batch_count = 1
_training_files_length = 10080

_algorithms = {
    low_load: ['K1SP_Random', 'K1SP_FirstFit',
               'K3SP_BestFit', '', '', '', '', '', '', ''],
    medium_load: ['K1SP_Random', 'K1SP_DP', 'K1SP_FirstFit', 'K3SP_BestFit',
                  'K3SP_PP', 'AP', 'K3SP_LastFit', 'K5SP_FirstFit', 'K5SP_CS', 'MAdapSPV'],
    high_load: ['K5SP_Random', 'K3SP_DP', 'K2SP_BestFit', 'K1SP_FirstFit',
                'K4SP_PP', 'K3SP_AP', 'K2SP_LastFit', 'K7SP_FirstFit', 'K5SP_CS', 'MAdapSPV'],
}

_algorithm_choice = 'high_load'

_loads = {
    low: ['75', '100', '125', '150', '175', '200', '225', '250', '275', '300', '325', '350', '375', '400'],
    high: ['150', '200', '250', '300', '350', '400', '450',
           '500', '550', '600', '650', '700', '750', '800']
}

_load_choice = 'high'


def batch_size():
    sample_amount = LEN_FILENAMES_TRAINING * SEED_COUNT
    return sample_amount // BATCH_COUNT


def algorithms():
    return _algorithms[_algorithm_choice]


def loads():
    return _loads[_load_choice]
