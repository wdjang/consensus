import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from plotly import tools
import plotly.graph_objs as go
import plotly.offline as plotlyoff
import h5py


def do_clustering(input_data, label_list, cluster_num, min_k, max_k):
 
  # ==================================================================================
  # Compute consensus affinity matrix
  # ==================================================================================
  # Try various number of clusters
  # Blend clustering result as consensus affinity matrix
  # ==================================================================================
  consensus_distance = np.zeros([np.size(input_data, 0), np.size(input_data, 0)])

  pdist_mat = squareform(pdist(input_data))
  pdist_mat = pdist_mat**2
  pdist_mat = np.exp(-pdist_mat/np.mean(pdist_mat))

  max_clusters = 10
  min_clusters = 2
  for n_clusters in range(min_clusters, max_clusters+1):
      print("Clustering with n_clusters =", n_clusters)

      # ==============================================================================
      # K-means
      # ==============================================================================
      # cc_cluster = KMeans(n_clusters=n_clusters, random_state=10).fit(input_data).labels_

      # ==============================================================================
      # Spectral clustering
      # ==============================================================================
      cc_cluster = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack',
                                      n_init=10, affinity='precomputed').fit(pdist_mat).labels_

      # ==============================================================================
      # Accumulate affinity matrix
      # ==============================================================================
      consensus_distance += squareform(
          pdist(np.reshape(cc_cluster, [len(cc_cluster), 1])) > 0)

  # ==================================================================================
  # Perform spectral clustering on the consensus_affinity
  # + Determine the best number of clusters using the silhouette coefficient
  # ==================================================================================
  consensus_distance = consensus_distance / np.max(consensus_distance)
  consensus_affinity = 1 - consensus_distance
  # consensus_distance = np.max(consensus_affinity)-consensus_affinity

  if cluster_num is not 0:
    cc_cluster = SpectralClustering(n_clusters=cluster_num, eigen_solver='arpack', n_init=10, affinity='precomputed').fit(consensus_affinity).labels_
  else:
    max_score = -1
    for num_clusters in range(min_k, max_k+1):
        temp_cluster = \
            SpectralClustering(n_clusters=num_clusters, eigen_solver='arpack',
                               n_init=10, affinity='precomputed').fit(consensus_affinity).labels_
        silhouette_avg = silhouette_score(X=consensus_distance, labels=temp_cluster,
                                          metric="precomputed")
        print("For n_clusters =", num_clusters,
              "The average silhouette_score is :", silhouette_avg)

        if max_score < silhouette_avg:
            max_score = silhouette_avg
            cc_cluster = temp_cluster
  # =======================================================================================
  # Compute signature of each cluster
  # =======================================================================================

  cluster_list = np.unique(cc_cluster)
  num_clusters = len(cluster_list)

  signature_mat = np.zeros([num_clusters, np.shape(input_data)[1]])
  signature_numel = np.zeros(num_clusters)
  trace_list = []
  for c_id in cluster_list:
      signature_mat[c_id, :] = np.mean(input_data[cc_cluster == c_id, :], axis=0)
      signature_numel[c_id] = np.sum(cc_cluster == c_id)
      trace_list.append(go.Scatter(x=label_list,
                                   y=np.mean(input_data[cc_cluster == c_id, :], axis=0),
                                   mode='markers', name='Cluster ' + str(c_id+1),
                                   error_y=dict(type='data', array=np.std(input_data[cc_cluster == c_id, :], axis=0), visible=True, color='#000000', thickness=0.7)
                                   ))
  signature_hf = h5py.File('output/cluster_signature.h5', 'w')
  signature_hf.create_dataset('signature', data=signature_mat)
  signature_hf.create_dataset('numel', data=signature_numel)
  signature_hf.close()

  num_rows = 3
  num_cols = int(np.ceil(num_clusters/num_rows))

  cluster_title = []
  cluster_diff = np.zeros([3, 1])
  for cluster_id in range(num_clusters):
      cluster_diff[0] = np.sum(cc_cluster == cluster_id) / \
                        np.shape(cc_cluster)[0]
      cluster_title.append('Cluster ' + str(cluster_id+1) + ' (' +
                           str(cluster_diff[0] * 100)[1:5] + '%)')
  fig = tools.make_subplots(rows=num_rows, cols=num_cols,
                            subplot_titles=cluster_title)
  for cluster_id in range(num_clusters):
      sub_yid = int(cluster_id / num_cols)
      sub_xid = int(cluster_id % num_cols)
      fig.append_trace(trace_list[sub_yid*num_cols+sub_xid], sub_yid+1, sub_xid+1)
  fig['layout'].update(height=1080, width=1920, title='Signatures of Clusters',
                       showlegend=False)
  for c_id in cluster_list:
      fig['layout']['yaxis'+str(c_id+1)].update(range=[0, 1])
  plotlyoff.plot(fig, filename='output/cluster_signature.html')

  # =====================================================================================
  # Save clustering result
  # =====================================================================================
  with open('output/cluster_labels.csv', 'w') as f:
      for c_list in range(np.size(cc_cluster, 0)):
          f.write(str(cc_cluster[c_list]))
          f.write('\n')

  # =====================================================================================
  # Save clustering statistics
  # =====================================================================================
  cell_cnt = 0
  sorted_data = np.zeros(np.shape(input_data))
  for c_id in cluster_list:
      num_cells = sum(cc_cluster == c_id)
      sorted_data[cell_cnt:cell_cnt+num_cells] = input_data[cc_cluster == c_id]
      intra_dist = squareform(pdist(sorted_data[cell_cnt:cell_cnt+num_cells], 'euclidean'))
      sorted_list = np.argsort(np.sum(intra_dist, axis=0))
      for sorted_id in range(len(sorted_list)):
          temp_data = sorted_data[cell_cnt+sorted_id]
          sorted_data[cell_cnt+sorted_id] = sorted_data[cell_cnt+sorted_list[sorted_id]]
          sorted_data[cell_cnt+sorted_list[sorted_id]] = temp_data
      cell_cnt += num_cells

  data_fig = plt.figure(figsize=[10, 10])
  plt.imshow(sorted_data, aspect='auto', cmap='coolwarm')
  x_ticks = np.arange(len(label_list))
  plt.xticks(x_ticks, label_list, fontsize=15)
  plt.ylabel('Cells')
  plt.xticks(rotation=80)
  plt.colorbar(fraction=0.049)
  data_fig.savefig('output/sorted_data.png')

  sorted_dist = squareform(pdist(sorted_data, 'euclidean'))

  dist_fig = plt.figure(figsize=[10, 10])
  plt.imshow(sorted_dist, cmap='YlGnBu')
  plt.xlabel('Cells')
  plt.ylabel('Cells')
  plt.colorbar(fraction=0.045)
  dist_fig.savefig('output/sorted_dist.png')
