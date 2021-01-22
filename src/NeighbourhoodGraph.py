import numpy as np
from scipy.spatial import distance_matrix, distance
from typing import List

class DistanceMatrix:

    def __init__(self, input_data:np.ndarray=[]):
        self.input_data = input_data
        self.distances = distance_matrix(input_data, input_data)

    def get_knn(self, k:int, index:int=None) -> List:
        """ Get the indices of the k-neares neighbours
        If index is none return knn for all input data samples (matrix)

        Args
            k (int): Number of nearest neighbours to look for
            index (int): Index of the sample in focus
        
        Return
            List[int]: Integer indices of the neighbours
        """
        if index is not None:
            neighbours = self.distances[index].argsort()[:k+1] # add 1 because distance to self is 0 and therefore always included
            return neighbours[neighbours != index]
        else:
            neighbours = self.distances.argsort()[:,:k+1]
            return [ row[row != idx] for idx, row in enumerate(neighbours)]

    def get_samples_in_radius(self, radius:float,  index:int=None) -> List:
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
            close_samples = [ np.where(row < radius)[0] for row in self.distances]
            return [ row[row != idx] for idx, row in enumerate(close_samples)]

