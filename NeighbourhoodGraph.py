import numpy as np
from scipy.spatial import distance_matrix, distance
from typing import List

class DistanceMatrix:

    def __init__(self, input_data:np.ndarray=[]):
        self.input_data = input_data
        self.distances = distance_matrix(input_data, input_data)

    def get_knn(self, index:int, k:int) -> List[int]:
        """ Get the indices of the k-neares neighbours

        Args
            index (int): Index of the sample in focus
            k (int): Number of nearest neighbours to look for
        
        Return
            List[int]: Integer indices of the neighbours
        """
        clostes_neighbours = self.distances[index].argsort()[:k+1] # add 1 because distance to self is 0 and therefore always included

        return clostes_neighbours[clostes_neighbours != index] # remove self
    

    def get_samples_in_radius(self, index:int, radius:float) -> List[int]:
        """ Get the indices of the samples which lie within a specifies radius of the index-sample

        Args
            index (int): Index of the sample in focus
            radius (float): radius of the Hypersphere around the index-sample
        
        Return
            List[int]: Integer indices of the neighbours
        """
        return np.where(distance_matrix(iris, iris)[0] < radius)[0]

    