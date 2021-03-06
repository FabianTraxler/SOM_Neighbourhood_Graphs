# %%
"""
This module contains copies of the classes SOMToolBox_Parse and SomViz provided by the lecturers.
"""
import pandas as pd
import numpy as np
import gzip

from scipy.spatial import distance_matrix, distance
from ipywidgets import Layout, HBox, Box, widgets, interact
import plotly.graph_objects as go


class SOMToolBox_Parse:

    def __init__(self, filename):
        self.filename = filename

    def read_weight_file(self, ):
        df = pd.DataFrame()
        if self.filename[-3:len(self.filename)] == '.gz':
            with gzip.open(self.filename, 'rb') as file:
                df, vec_dim, xdim, ydim = self._read_vector_file_to_df(df, file)
        else:
            with open(self.filename, 'rb') as file:
                df, vec_dim, xdim, ydim = self._read_vector_file_to_df(df, file)

        file.close()
        return df.astype('float64'), vec_dim, xdim, ydim

    def _read_vector_file_to_df(self, df, file):
        xdim, ydim, vec_dim, position = 0, 0, 0, 0
        for byte in file:
            line = byte.decode('UTF-8')
            if line.startswith('$'):
                xdim, ydim, vec_dim = self._parse_vector_file_metadata(line, xdim, ydim, vec_dim)
                if xdim > 0 and ydim > 0 and len(df.columns) == 0:
                    df = pd.DataFrame(index=range(0, ydim * xdim), columns=range(0, vec_dim))
            else:
                if len(df.columns) == 0 or vec_dim == 0:
                    raise ValueError('Weight file has no correct Dimensional information.')
                position = self._parse_weight_file_data(line, position, vec_dim, df)
        return df, vec_dim, xdim, ydim

    def _parse_weight_file_data(self, line, position, vec_dim, df):
        splitted = line.split(' ')
        try:
            df.values[position] = list(np.array(splitted[0:vec_dim]).astype(float))
            position += 1
        except:
            raise ValueError('The input-vector file does not match its unit-dimension.')
        return position

    def _parse_vector_file_metadata(self, line, xdim, ydim, vec_dim):
        splitted = line.split(' ')
        if splitted[0] == '$XDIM':
            xdim = int(splitted[1])
        elif splitted[0] == '$YDIM':
            ydim = int(splitted[1])
        elif splitted[0] == '$VEC_DIM':
            vec_dim = int(splitted[1])
        return xdim, ydim, vec_dim


# %%
class SomViz:

    def __init__(self, weights, m, n):
        self.weights = weights
        self.m = m
        self.n = n

    def umatrix(self, som_map=None, color="Viridis", interp="best", title=""):
        um = np.zeros((self.m * self.n, 1))
        neuron_locs = list()
        for i in range(self.m):
            for j in range(self.n):
                neuron_locs.append(np.array([i, j]))
        neuron_distmat = distance_matrix(neuron_locs, neuron_locs)

        for i in range(self.m * self.n):
            neighbor_idxs = neuron_distmat[i] <= 1
            neighbor_weights = self.weights[neighbor_idxs]
            um[i] = distance_matrix(np.expand_dims(self.weights[i], 0), neighbor_weights).mean()

        if som_map == None:
            return self.plot(um.reshape(self.m, self.n), color=color, interp=interp, title=title)
        else:
            som_map.data[0].z = um.reshape(self.m, self.n)

    def hithist(self, som_map=None, idata=[], color='RdBu', interp="best", title=""):
        hist = [0] * self.n * self.m
        for v in idata:
            position = np.argmin(np.sqrt(np.sum(np.power(self.weights - v, 2), axis=1)))
            hist[position] += 1

        if som_map == None:
            return self.plot(np.array(hist).reshape(self.m, self.n), color=color, interp=interp, title=title)
        else:
            som_map.data[0].z = np.array(hist).reshape(self.m, self.n)

    def component_plane(self, som_map=None, component=0, color="Viridis", interp="best", title=""):
        if som_map == None:
            return self.plot(self.weights[:, component].reshape(-1, self.n), color=color, interp=interp, title=title)
        else:
            som_map.data[0].z = self.weights[:, component].reshape(-1, self.n)

    def sdh(self, som_map=None, idata=[], sdh_type=1, factor=1, draw=True, color="Cividis", interp="best", title=""):

        import heapq
        sdh_m = [0] * self.m * self.n

        cs = 0
        for i in range(0, factor): cs += factor - i

        for vector in idata:
            dist = np.sqrt(np.sum(np.power(self.weights - vector, 2), axis=1))
            c = heapq.nsmallest(factor, range(len(dist)), key=dist.__getitem__)
            if (sdh_type == 1):
                for j in range(0, factor):  sdh_m[c[j]] += (factor - j) / cs  # normalized
            if (sdh_type == 2):
                for j in range(0, factor): sdh_m[c[j]] += 1.0 / dist[c[j]]  # based on distance
            if (sdh_type == 3):
                dmin = min(dist)
                for j in range(0, factor): sdh_m[c[j]] += 1.0 - (dist[c[j]] - dmin) / (max(dist) - dmin)

        if som_map == None:
            return self.plot(np.array(sdh_m).reshape(-1, self.n), color=color, interp=interp, title=title)
        else:
            som_map.data[0].z = np.array(sdh_m).reshape(-1, self.n)

    def project_data(self, som_m=None, idata=[], title=""):

        data_y = []
        data_x = []
        for v in idata:
            position = np.argmin(np.sqrt(np.sum(np.power(self.weights - v, 2), axis=1)))
            x, y = position % self.n, position // self.n
            data_x.extend([x])
            data_y.extend([y])

        if som_m != None: som_m.add_trace(
            go.Scatter(x=data_x, y=data_y, mode="markers", marker_color='rgba(255, 255, 255, 0.8)', ))

    def time_series(self, som_m=None, idata=[], wsize=50, title=""):  # not tested

        data_y = []
        data_x = [i for i in range(0, len(idata))]

        data_x2 = []
        data_y2 = []

        qmin = np.Inf
        qmax = 0

        step = 1

        ps = []
        for v in idata:
            matrix = np.sqrt(np.sum(np.power(self.weights - v, 2), axis=1))
            position = np.argmin(matrix)
            qerror = matrix[position]
            if qmin > qerror: qmin = qerror
            if qmax < qerror: qmax = qerror
            ps.append((position, qerror))

        markerc = []
        for v in ps:
            data_y.extend([v[0]])
            rez = v[1] / qmax

            markerc.append('rgba(0, 0, 0, ' + str(rez) + ')')

            x, y = v[0] % self.n, v[0] // self.n
            if x == 0:
                y = np.random.uniform(low=y, high=y + .1)
            elif x == self.m - 1:
                y = np.random.uniform(low=y - .1, high=y)
            elif y == 0:
                x = np.random.uniform(low=x, high=x + .1)
            elif y == self.n - 1:
                x = np.random.uniform(low=x - .1, high=x)
            else:
                x, y = np.random.uniform(low=x - .1, high=x + .1), np.random.uniform(low=y - .1, high=y + .1)

            data_x2.extend([x])
            data_y2.extend([y])

        ts_plot = go.FigureWidget(go.Scatter(x=[], y=[], mode="markers", marker_color=markerc,
                                             marker=dict(colorscale='Viridis', showscale=True,
                                                         color=np.random.randn(500))))
        ts_plot.update_xaxes(range=[0, wsize])

        ts_plot.data[0].x, ts_plot.data[0].y = data_x, data_y
        som_m.add_trace(go.Scatter(x=data_x2, y=data_y2, mode="markers", ))

        som_m.layout.height = 500
        ts_plot.layout.height = 500
        som_m.layout.width = 500
        ts_plot.layout.width = 1300

        return HBox([go.FigureWidget(som_m), go.FigureWidget(ts_plot)])

    def plot(self, matrix, color="Viridis", interp="best", title=""):
        return go.FigureWidget(go.Heatmap(z=matrix, zsmooth=interp, showscale=False, colorscale=color),
                               layout=go.Layout(width=700, height=700, title=title, title_x=0.5, ))


if __name__ == "__main__":
    from sklearn import datasets, preprocessing
    from src.NeighbourhoodGraph import NeighbourhoodGraph

    iris = datasets.load_iris().data
    # min_max_scaler = preprocessing.MinMaxScaler()
    # iris = min_max_scaler.fit_transform(iris)

    smap = SOMToolBox_Parse('../input/iris/iris.wgt.gz')
    s_weights, sdim, smap_x_dim, smap_y_dim = smap.read_weight_file()
    s_weights = s_weights.to_numpy()

    ng_iris = NeighbourhoodGraph(s_weights, smap_x_dim, smap_y_dim, input_data=iris)
    ng_iris_trace_3nn = ng_iris.get_trace(knn=3)

    go.FigureWidget(data=ng_iris_trace_3nn,
                    layout=go.Layout(width=700, height=700, title="Iris: NeighbourhoodGraph (3-NN)")).show()

    vis_iris = SomViz(s_weights, smap_x_dim, smap_y_dim)
    um_iris = vis_iris.umatrix(title="Iris: Umatrix + NeighbourhoodGraph (3-NN)")
    um_iris.add_trace(ng_iris_trace_3nn)
    um_iris.show()

    # We can reuse all traces
    ng_iris_trace_05r = ng_iris.get_trace(radius=0.5)

    um_iris.data = [um_iris.data[0]]
    um_iris.add_trace(ng_iris_trace_05r)
    um_iris.layout = go.Layout(width=700, height=700, title="Iris: Umatrix + NeighbourhoodGraph (0.5 Radius)",
                               title_x=0.5, )
    um_iris.show()

    hithist_iris = vis_iris.hithist(idata=iris, title="Iris: HisHist + NeighbourhoodGraph (3-NN)")
    hithist_iris.add_trace(ng_iris_trace_3nn)
    hithist_iris.show()