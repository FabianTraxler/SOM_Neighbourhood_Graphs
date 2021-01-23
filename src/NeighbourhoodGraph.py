import numpy as np
from scipy.spatial import distance_matrix, distance
from typing import List


class DistanceMatrix:

    def __init__(self, input_data: np.ndarray = []):
        self.input_data = input_data
        self.distances = distance_matrix(input_data, input_data)

    def get_knn(self, k: int, index: int = None) -> List:
        """ Get the indices of the k-neares neighbours
        If index is none return knn for all input data samples (matrix)

        Args
            k (int): Number of nearest neighbours to look for
            index (int): Index of the sample in focus
        
        Return
            List[int]: Integer indices of the neighbours
        """
        if index is not None:
            neighbours = self.distances[index].argsort()[
                         :k + 1]  # add 1 because distance to self is 0 and therefore always included
            return neighbours[neighbours != index]
        else:
            neighbours = self.distances.argsort()[:, :k + 1]
            return [row[row != idx] for idx, row in enumerate(neighbours)]

    def get_samples_in_radius(self, radius: float, index: int = None) -> List:
        """ Get the indices of the samples which lie within a specifies radius of the index-sample
        If index is none return samples in radius for all input data samples (matrix)

        Args
            index (int): Index of the sample in focus
            radius (float): radius of the Hypersphere around the index-sample
        
        Return
            List[int]: Integer indices of the close samples
        """
        if index is not None:
            close_samples = np.where(self.distances[index] < radius)[0]
            return close_samples[close_samples != index]
        else:
            close_samples = [np.where(row < radius)[0] for row in self.distances]
            return [row[row != idx] for idx, row in enumerate(close_samples)]



class NeighbourhoodGraph:
    def __init__(self, unit_weights=[], m=None, n=None, input_data:np.ndarray=[], distance_mat:np.ndarray=None):
        if m * n != len(unit_weights):
            print("Weigths and specified input dimensions m & n do not match")
            return
        
        if len(unit_weights)==0 or not m or not n or len(input_data)==0:
            print("You need to specify all relevant arguments: weights, m, n, input_data")

        self.unit_weights = unit_weights
        self.m = m
        self.n = n

        self.input_data = input_data
        if distance_mat is None:
            self.distance_mat = DistanceMatrix(input_data)
        else:
            self.distance_mat = distance_mat

    def create_graph(self, radius:float=None, knn:int=None) -> np.ndarray:
        """
        Create an adjecency matrix representing the Neighbourhood graph
        Use KNN or Radius Approach depending on which argument is specified
        If both are specifies KNN is used
        
        Args:
            radius (float): Radius size (if specified use radius method)
            knn (int): Number of KNN (if specifiec use KNN)
        
        Returns:
            np.ndarray: The Adjacancy Matrix representing the Neighbourhood Graph
        """
        adjacancy_mat = np.zeros((self.n,self.m))
        if knn is not None:
            neighbours = self.distance_mat.get_knn(knn)
        elif radius is not None:
            neighbours = self.distance_mat.get_samples_in_radius(radius)
        
        for index, neighbour_idxs in enumerate(neighbours):
            bmu = self.get_bmu(index)
            for neighbour_idx in neighbour_idxs:
                neighbour_bmu = self.get_bmu(neighbour_idx)
                if bmu != neighbour_bmu:
                    adjacancy_mat[bmu, neighbour_bmu] = 1

        return adjacancy_mat
    
    def get_bmu(self, index) -> int:
        """
        Get Best Matching Unit of a input samle
        Use the Eukleadian Distance (L2) to find BMU

        Args:
            index (int): Index of the input sample

        Returns
            int: Index of the Best Matching Unit
        """
        return np.argmin(np.sqrt(np.sum(np.power(self.unit_weights - self.input_data[index], 2), axis=1)))



if __name__ =="__main__":
    from Utilities import SOMToolBox_Parse
    from sklearn import datasets, preprocessing

    iris = datasets.load_iris().data
    min_max_scaler = preprocessing.MinMaxScaler()
    iris = min_max_scaler.fit_transform(iris)

    smap = SOMToolBox_Parse('../input/iris/iris.wgt.gz')
    smap, sdim, smap_x, smap_y = smap.read_weight_file()
    print(sdim)

    ng = NeighbourhoodGraph(smap._weights.reshape(-1,4), m, n, iris)

    adj_mat = ng.create_graph(3)

    print(adj_mat)