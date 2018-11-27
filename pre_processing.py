# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
import numpy as np
# import matplotlib.pyplot as plt

# =========================================================================
# Extract user-defined markers
# =========================================================================
def marker_extraction(data_array, label_list, marker_list):
  extracted_marker = np.zeros((marker_list.__len__(), np.shape(data_array)[1]))
  for coords_id in range(0, marker_list.__len__()):
      inlist_id = label_list.index(marker_list[coords_id])
      extracted_marker[coords_id, :] = data_array[inlist_id, :]

  return extracted_marker

>>>>>>> 1766239020b50d79882ffea7abb3216ec407cdf0

# =========================================================================
# Perform sampling and normalization
# =========================================================================
def sampling_normalization(input_path, label_list, sample_ratio):
    # =========================================================================
    # Load input data
    # =========================================================================
    input_data = np.genfromtxt(input_path,
                               names=True, dtype=float, delimiter=',')
    input_label = input_data.dtype.names
    input_data = input_data.view((np.float, len(input_data.dtype.names)))
    # =========================================================================
    # Choose user-defined markers
    # =========================================================================
    chosen_data = np.zeros([np.shape(input_data)[0], len(label_list)])
    for label_id, label_name in enumerate(label_list):
        chosen_data[:, label_id] = input_data[:, input_label.index(label_name)]

    # =========================================================================
    # Log10 transform
    # =========================================================================
    norm_data = np.log10(chosen_data+1)

    # =========================================================================
    # Percentile normalization by mapping [0.1%, 99.9%] into [0, 1]
    # =========================================================================
    for marker_id in np.arange(np.shape(norm_data)[1]):
        min_tile, max_tile = np.percentile(norm_data[:, marker_id], [0.1, 99.9])
        norm_data[:, marker_id] = (norm_data[:, marker_id] - min_tile) / (max_tile - min_tile)
    
    for marker_id in np.arange(np.shape(norm_data)[1]):
        outlier_list = np.where(norm_data[:, marker_id] > 1)
        norm_data = np.delete(norm_data, outlier_list, axis=0)
        
        outlier_list = np.where(norm_data[:, marker_id] < 0)
        norm_data = np.delete(norm_data, outlier_list, axis=0)

    np_perm = np.arange(norm_data.shape[0])
    np.random.shuffle(np_perm)
    sample_num = int(round(sample_ratio*np_perm.shape[0]))

    sample_data = norm_data[np_perm[:sample_num]].copy()

    return sample_data
