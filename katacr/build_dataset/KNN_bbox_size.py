from katacr.utils.related_pkgs.utility import *
from katacr.build_dataset.utils.datapath_manager import PathManager
from katacr.build_dataset.constant import path_logs
from katacr.detection.cfg import image_shape

def get_bbox_size():
  path_manager = PathManager()
  paths = path_manager.sample('images', part=2, regex=r"^\d+.txt")
  ret = []
  for path in paths:
    with path.open('r') as file:
      params = file.read().split('\n')[:-1]
    for param in params:
      try:
        parts = param.split(' ')
        w, h = float(parts[3]) * image_shape[1], float(parts[4]) * image_shape[0]
        ret.append(np.array((w, h), dtype=np.float32))
      except:
        print(f"{path=}")
        # print(params)
        raise("Error")
    # print(path)
  return np.array(ret)

def knn_calc_bbox_size(data, k=9, verbose=False):
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=k)
  kmeans.fit(data)
  cluster_centroids = kmeans.cluster_centers_
  
  if verbose:
    anchors = list(cluster_centroids)
    anchors = sorted(anchors, key=lambda x: np.prod(x))
    print("anchors = [")
    for i in range(3):
      print("  ", end="")
      for j in range(3):
        cluster = anchors[i*3+j]
        print(f"({cluster[0]:.1f}, {cluster[1]:.1f}), ", end="")
      print("")
    print("]")

    import matplotlib.pyplot as plt
    # data_sample = data[:1000, ...]
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', label='Data')
    plt.scatter(cluster_centroids[:, 0], cluster_centroids[:, 1], c='red', marker='*', s=200, label='Cluster Centroid')
    plt.title("KMeans Clustering")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(path_logs.joinpath("KNN_bbox.jpg")), dpi=200)
    plt.show()

if __name__ == '__main__':
  data = get_bbox_size()
  knn_calc_bbox_size(data, verbose=True)