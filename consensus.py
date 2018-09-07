import yaml
import numpy as np

from pre_processing import sampling_normalization
from ensemble_clustering import do_clustering


def main():
    with open('config/consensus.yaml') as f:
        # use safe_load instead load
        c_info = yaml.safe_load(f)

    # =======================================================================================
    # Pre-processing on each data (PRE/ON/POST)
    # =======================================================================================
    input_path = 'input/' + c_info['data']
    # input_path = './' + c_info['data']
    label_list = c_info['markers']
    sample_num = c_info['samplesize']
    cluster_num = c_info['k']
    max_k = c_info['maxk']
    min_k = c_info['mink']

    print('Loading dataset...')
    sampled_data = sampling_normalization(input_path, label_list, sample_num)
    print(str(np.shape(sampled_data)[1]) + ' cells are sampled for clustering')
    print('Performing clustering...')
    do_clustering(sampled_data, label_list, cluster_num, min_k, max_k)


if __name__ == '__main__':
    main()
