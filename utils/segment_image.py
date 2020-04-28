import numpy as np
import gc
from skimage.filters import rank
from skimage.morphology import watershed, disk

from scipy import ndimage as ndi
from skimage.future import graph


def segment_image(input_data):

    segmented = watershed_transformation(input_data)

    #    return segmented_data2, gradient, markers_nolabel
    return segmented


# --------Region Adjacency Graphs (RAGs) Merging process ---------------

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])


# ------------------------------------------------------------------------------

def watershed_transformation(image_data):
    '''
    Runs a watershed transform on the main dataset (Revised by Xin Miao 1/7/2019)
        1. Create an edge-detection layer using the sobel algorithm
        2. Find the local minimum based on distance transform and place a marker
        3. Adjust the gradient image based on given threshold and amplification.
        4. Construct watersheds on top of the gradient image starting at the
            markers.
    '''
    # If this block has no data, return a placeholder watershed.
    if np.amax(image_data[0]) <= 1:
        # We just need the dimensions from one band
        return np.zeros(np.shape(image_data[0]))

    denoised = rank.median(image_data, disk(5))
    gradient = rank.gradient(denoised, disk(5))  # <----------------- Parameter 1: gradient searching radius

    markers = gradient < 5  # <----------------- Parameter 2: gradient marker threshold
    markers = ndi.label(markers)[0]  # Label each marker 1,2,3...
    im_watersheds = watershed(gradient, markers)  # call watershed transform internal function

    labels = im_watersheds
    img = np.stack((image_data,) * 3, axis=-1)
    g = graph.rag_mean_color(img, labels)
    labels2 = graph.merge_hierarchical(labels, g, thresh=20, rag_copy=False,  # <-- Parameter 3: merging threshold
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)

    im_watersheds2 = labels2
    gc.collect()
    # return im_watersheds
    #    return im_watersheds2, im_watersheds, gradient, markers_nolabel
    return im_watersheds2