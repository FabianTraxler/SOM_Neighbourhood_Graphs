def create_visualizations(sweights, smap_y:int, smap_x:int, idata,
                          color='viridis', interp=False, data_title='Chainlink',
                          k_list = [1,2,3,4],
                          r_list = [0.03,0.09,0.15,0.21],
                          width = 700, height = None,
                          scale_to_mean = False,
                          show_hithist = False):
    # Visualization
    viz = SomViz(sweights, smap_y, smap_x)

    if scale_to_mean: # scale radii to input data
        r_list = np.around(np.mean(idata)*np.array(r_list), decimals=8)

    if height is None: height = width * (smap_y/smap_x)

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
        else:
            title_suffix = " + Neighbourhood Graph (radius: {})".format(r_list[i-len(k_list)])

        umatrix = viz.umatrix(color=color, interp=interp, title=umatrix_title + title_suffix)
        umatrix.layout.width = width
        umatrix.layout.height = height
        umatrix.add_trace(trace)

        if show_hithist:
            hithist = viz.hithist(idata=idata, color=color, interp=interp, title=hithist_title + title_suffix)
            hithist.layout.width = width
            hithist.layout.height = height
            hithist.add_trace(trace)

            display(HBox([umatrix, hithist]))
        else:
            display(umatrix)