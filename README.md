# consensus

The repository hosts source code for Docker containers used for making cell state calls on segmented CyCIF data:
* consensus - Performs consensus clustering on CyCIF data and produces signatures that define each cluster. The config file expects the following fields:
 * k - The desired number of clusters
 * mink, maxk - The minimum and maximum number of clusters to try (this is valid only when k=0)
 * data - Filename of segmented CyCIF data that will appears in input/ directory of the container
 * markers - List of markers that should be used for clustering
 * samplesize - Number of samples to use for clustering (0 to use all cells)
