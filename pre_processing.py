import numpy as np

# =========================================================================
# Extract user-defined markers
# =========================================================================
def marker_extraction(data_array, label_list, marker_list):
  extracted_marker = np.zeros((marker_list.__len__(), np.shape(data_array)[1]))
  for coords_id in range(0, marker_list.__len__()):
      inlist_id = label_list.index(marker_list[coords_id])
      extracted_marker[coords_id, :] = data_array[inlist_id, :]

  return extracted_marker


# =========================================================================
# Perform sampling and normalization
# =========================================================================
def sampling_normalization(input_path, label_list, sample_num):
    # =========================================================================
    # Load input data
    # =========================================================================
    input_data = np.genfromtxt(input_path,
                               names=True, dtype=float, delimiter=',')
    input_label = input_data.dtype.names
    input_data = input_data.view((np.float, len(input_data.dtype.names)))
    input_data = input_data.transpose()

    # Frame index
    marker_list = ['frame']
    frame_list = marker_extraction(input_data, input_label, marker_list)

    # =========================================================================
    # Remove useless markers
    # =========================================================================
    chosen_data = np.zeros([len(label_list), np.shape(input_data)[1]])
    for label_id, label_name in enumerate(label_list):
        chosen_data[label_id, :] = input_data[input_label.index(label_name), :]

    # =========================================================================
    # Log10 transform
    # =========================================================================
    norm_data = np.log10(chosen_data+1)

    # =========================================================================
    # Percentile normalization by mapping [1%, 99%] into [-1, 1]
    # =========================================================================
    for marker_id in np.arange(np.shape(norm_data)[0]):
        min_tile, max_tile = np.percentile(norm_data[marker_id, :], [1, 99])
        norm_data[marker_id, :] = (norm_data[marker_id, :] - min_tile) / \
                                  (max_tile - min_tile)
        norm_data[marker_id, :] = 2.0*(norm_data[marker_id, :] - 0.5)
        norm_data[marker_id, :] = np.clip(norm_data[marker_id, :], -1, 1)

    # =========================================================================
    # Perform sampling for each frame (local tile)
    # =========================================================================
    unq_frames = np.unique(frame_list)
    if sample_num == 0:
        sample_percent = 1
    else:
        sample_percent = float(sample_num)/np.shape(norm_data)[1]
    num_totalsample = 0
    for frame_id in unq_frames:
        cell_finder = np.where(frame_list == frame_id)[1]
        num_totalsample += int(np.ceil(np.shape(cell_finder)[0] * sample_percent))

    sampled_data = np.zeros([np.shape(norm_data)[0], num_totalsample])
    sampled_xy = np.zeros([2, num_totalsample])

    sample_id = 0
    for frame_id in unq_frames:
        cell_finder = np.where(frame_list == frame_id)[1]
        num_sample = int(np.ceil(np.shape(cell_finder)[0] * sample_percent))
        sampleid_list = np.random.choice(cell_finder, num_sample, replace=False)

        sampled_data[:, sample_id + np.arange(np.shape(sampleid_list)[0])] = \
            norm_data[:, sampleid_list]

        sample_id += np.shape(sampleid_list)[0]

    return sampled_data
