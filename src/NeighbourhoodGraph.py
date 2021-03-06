# -*- coding: utf-8 -*-
"""Neighbourhood Graph Visualization Tool

This module is used to create a Neighbourhood Graph based on a trained SOM.
Therefore the Class NeighbourhoodGraph should be used to initialize the Neighbourhood graph.

After initialization the instance can be used to generate a so called trace which can be displayed over any other visualization (eg. U-Matrix)

@author Lukas Lehner
@author Fabian Traxler
"""
import numpy as np
from scipy import spatial
from typing import List
import plotly.graph_objects as go
import networkx as nx


class DistanceMatrix:
    """ Class to calculate the (euclidean) distance between each input data sample
     and derive the knn- or radius-based neighbourhood of each sample in input space."""
    def __init__(self, input_data: np.ndarray = []):
        """
        Initialize the Distance Matrix with the input data.

        Args:
            input_data (np.ndarray): The input data used for training the SOM (also possible to use different data in same dimension)
        """
        self.distance_matrix = spatial.distance_matrix(input_data, input_data)

    def get_knn(self, k: int, index: int = None) -> List:
        """
        Get the indices of the k-nearest neighbours.
        If index is none return knn for all input data samples (matrix).

        Args
            k (int): Number of nearest neighbours to look for
            index (int): Index of the sample in focus

        Return
            List[int]: Integer indices of the neighbouring samples
        """
        if index is not None:
            neighbours = self.distance_matrix[index].argsort()[
                         :k + 1]  # add 1 because distance to self is 0 and therefore always included
            return neighbours[neighbours != index]
        else:
            neighbours = self.distance_matrix.argsort()[:, :k + 1]
            return [row[row != idx] for idx, row in enumerate(neighbours)]

    def get_samples_in_radius(self, radius: float, index: int = None) -> List:
        """
        Get the indices of the samples which lie within a specified radius of the index-sample.
        If index is none return samples in radius for all input data samples (matrix).

        Args
            radius (float): Radius of the hypersphere around the index-sample
            index (int): Index of the sample in focus

        Return
            List[int]: Integer indices of samples in the hypersphere
        """
        if index is not None:
            close_samples = np.where(self.distance_matrix[index] < radius)[0]
            return close_samples[close_samples != index]
        else:
            close_samples = [np.where(row < radius)[0] for row in self.distance_matrix]
            return [row[row != idx] for idx, row in enumerate(close_samples)]


class NeighbourhoodGraph:
    """Class representing the Neighbourhood Graph, including functions for visualization."""

    def __init__(self, unit_weights: np.ndarray, m: int, n: int, input_data: np.ndarray,
                 distance_mat: DistanceMatrix = None, bmu_array: np.ndarray = None):
        """
        Initialize the NeighbourhoodGraph Instance with the trained SOM and the used input data.

        Args:
            unit_weights (np.ndarray): The Weights of the Units in Input space
            m (int): Height of the SOM
            n (int): Width of the SOM
            input_data (np.ndarray): The Input data used for training the SOM (also possible to use different data in same dimension)
            distance_mat (DistanceMatrix): Optional: A Matrix representing the distance between each input data sample. If none is provided, the Distance Matrix is calculated using input_data.
            bmu_array (np.ndarray): Optional: Best Matching Unit for each input data sample. If none is provided, the array is calculated using unit_weights and input_data.
        """

        assert m * n == len(unit_weights), \
            "NeighbourhoodGraph __init__: Weights and specified input dimensions m & n do not match"

        self.unit_weights = unit_weights
        self.m = m
        self.n = n
        self.input_data = input_data

        self.distance_mat = DistanceMatrix(input_data) if distance_mat is None else distance_mat
        self.bmu_array = self.calc_bmu_array(unit_weights, input_data) if bmu_array is None else bmu_array

    def calc_bmu_array(self, _unit_weights: np.ndarray, _input_data: np.ndarray) -> np.ndarray:
        """
        Calculate Best Matching Unit for every sample.
        Use the Euclidean Distance (L2) to find BMUs.
        If multiple units have the same distance to a sample, the unit with the lowest index is chosen.
        Args:
            _unit_weights (ndarray): Trained unit weights
            _input_data (ndarray): Input data
        Returns
            np.ndarray: Array of indices of the Best Matching Units
        """
        return np.array(
            [np.argmin(np.sqrt(np.sum(np.power(_unit_weights - _input_data[index], 2), axis=1)))
             for index in np.arange(self.input_data.shape[0])])

    def calc_adjacency_matrix(self, radius: float = None, knn: int = None) -> np.ndarray:
        """
        Calculate the adjacency matrix of the Neighbourhood Graph corresponding to the specified neighbourhood.
        Use KNN or Radius Approach to define neighbourhood (depends on which argument is specified - if both are specified KNN is used)

        Args:
            radius (float): Radius size (if specified, use radius method)
            knn (int): Number of neighbours (if specified, use KNN)

        Returns:
            np.ndarray: The Adjacency Matrix of the Neighbourhood Graph
        """
        assert (radius is not None or knn is not None), \
            "NeighbourhoodGraph create_graph: radius or knn must be specified"
        assert (radius is None or radius > 0), "NeighbourhoodGraph create_graph: radius must be > 0"
        assert (knn is None or knn > 0), "NeighbourhoodGraph create_graph: knn must be > 0"

        adjacency_mat = np.zeros((self.n * self.m, self.n * self.m))

        if knn is not None:
            neighbours = self.distance_mat.get_knn(knn)
        else:
            neighbours = self.distance_mat.get_samples_in_radius(radius)

        for index, neighbour_idxs in enumerate(neighbours):
            bmu = self.bmu_array[index]
            for neighbour_idx in neighbour_idxs:
                neighbour_bmu = self.bmu_array[neighbour_idx]
                if bmu != neighbour_bmu:
                    adjacency_mat[bmu, neighbour_bmu] = 1

        return adjacency_mat

    def get_trace(self, radius: float = None, knn: int = None, adjacency_matrix: np.ndarray = None,
                  line_width: float = 3, line_color:str='#fff', pos: dict = None) -> go.Scatter:
        """
        Build a plotly Scatter Plot of the Neighbourhood Graph (can be used as trace in a FigureWidget).

        Args:
            radius (float): Radius size (if specified use radius method)
            knn (int): Number of neighbours (if specified use KNN)
            adjacency_matrix (np.ndarray): Optional: An adjacency matrix of a Neighbourhood Graph. If none is provided, an adjacency matrix is calculated based on the specified neighbourhood parameter (radius or knn).
            line_width (float): The width of the edges in the plot
            line_color (str): The color of the edges (in HEX Hash)
            pos (dict): Optional: The position of the Units in the 2D plot. If none is provided, the units are laid out on an m x n grid with cell size=1.
        Returns:
            plotly.graph_objects.Scatter: A Scatter Plot Object
        """
        if adjacency_matrix is None:
            adjacency_matrix = self.calc_adjacency_matrix(radius, knn)

        # Make networkx graph from input adjacency matrix
        G = nx.from_numpy_matrix(adjacency_matrix, parallel_edges=True)

        # # If no unit positions are provided, units are arranged on an m x n grid with grid cell size = 1
        if pos is None:
            pos = {}
            counter = 0
            for i in range(self.m):
                for j in range(self.n):
                    pos.update({counter: (j, i)})
                    counter += 1
        nx.set_node_attributes(G, pos, 'pos')

        # Convert nx graph to plotly trace data
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = G.nodes[edge[0]]['pos']
            x1, y1 = G.nodes[edge[1]]['pos']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        return go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=line_width, color=line_color),
            hoverinfo='skip',
            mode='lines')


if __name__ == "__main__":
    from src.som_vis import SOMToolBox_Parse
    from sklearn import datasets, preprocessing

    iris = datasets.load_iris().data
    # min_max_scaler = preprocessing.MinMaxScaler()
    # iris = min_max_scaler.fit_transform(iris)

    smap = SOMToolBox_Parse('../input/iris/iris.wgt.gz')
    s_weights, sdim, smap_x_dim, smap_y_dim = smap.read_weight_file()
    s_weights = s_weights.to_numpy()

    ng_iris = NeighbourhoodGraph(s_weights, smap_x_dim, smap_y_dim, input_data=iris)
    trace = ng_iris.get_trace(radius=0.5)
    go.FigureWidget(data=trace, layout=go.Layout(width=500, height=500, title="title")).show()
