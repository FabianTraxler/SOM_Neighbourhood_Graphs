"""
This file define a utility function that is used to create multiple visualizations at once
"""
import numpy as np
from IPython.core.display import display
from ipywidgets import HBox

from NeighbourhoodGraph import NeighbourhoodGraph
from som_vis import SomViz


def create_visualizations(sweights: np.ndarray, smap_y: int, smap_x: int, idata,
                          color='viridis', interp=False, data_title='Chainlink',
                          k_list=[1, 2, 3, 4],
                          r_list=[0.03, 0.09, 0.15, 0.21],
                          width=700, height=None,
                          scale_to_mean=False,
                          show_hithist=False,
                          save_dir=None, save_file_type="svg",
                          display_mode=False):
    """
    Create multiple Visualizations with different Parameters at once
    Displays a plot for every parameter in k_list, and r_list
    Also Scales the plots to width 

    Args:
        sweights (np.ndarray): The unit weights of the SOM in input space
        smap_y (int): The y-size of the SOM 
        smap_x (int): The x-size of the SOM
        color (str): The color palette to be used,
        interp (bool): Use interpolation or not
        data_title (str): The Title of the dataset for the plots
        k_list (list): KNNs to be visualized
        r_list (list): Radii to be visualized
        width (int): Width of the plots
        height (int): Height of the plots
        scale_to_mean (bool): Specify if data should be scaled (only relevant for Radius method)
        show_hithist (bool): Specify if also a HitHistogram should be used
        save_dir (str):
        save_file_type (str):
        display_mode (bool):
    """
    # Visualization
    viz = SomViz(sweights, smap_y, smap_x)

    if scale_to_mean:  # scale radii to input data
        r_list = np.around(np.mean(idata) * np.array(r_list), decimals=8)

    if height is None: height = width * (smap_y / smap_x)

    save_file_title = "{}_umatrix_".format(data_title)
    if show_hithist: save_file_title += "hithist_"
    save_file_title += "neighbourhoodgraph-"

    # Prepare Neighbourhood Graph
    ng = NeighbourhoodGraph(viz.weights, viz.m, viz.n, input_data=idata)
    traces = []
    for val in k_list:
        traces.append(ng.get_trace(knn=val))
    for val in r_list:
        traces.append(ng.get_trace(radius=val))

    # Prepare underlying visualizations
    umatrix_title = '{}: U-Matrix'.format(data_title)
    hithist_title = '{}: Hit hist'.format(data_title)

    # Show underlying visualizations with Neighbourhood Graph overlay
    for i, trace in enumerate(traces):
        if i < len(k_list):
            title_suffix = " + Neighbourhood Graph ({}-NN)".format(k_list[i])
            save_file_suffix = "-knn-{}".format(k_list[i])
        else:
            title_suffix = " + Neighbourhood Graph (radius: {})".format(r_list[i - len(k_list)])
            save_file_suffix = "-rad-{}".format(r_list[i - len(k_list)])

        umatrix = viz.umatrix(color=color, interp=interp, title=umatrix_title + title_suffix)
        umatrix.layout.width = width
        umatrix.layout.height = height
        umatrix.add_trace(trace)

        fig = umatrix

        if show_hithist:
            hithist = viz.hithist(idata=idata, color=color, interp=interp, title=hithist_title + title_suffix)
            hithist.layout.width = width
            hithist.layout.height = height
            hithist.add_trace(trace)

            fig = HBox([umatrix, hithist])

        if save_dir is not None:
            fig.write_image(save_dir + save_file_title + save_file_suffix + "." + save_file_type)

        if display_mode:
            display(fig)
        else:
            fig.show()
