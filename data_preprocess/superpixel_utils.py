import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import h5py
from sklearn.cluster import MiniBatchKMeans, KMeans
import random 
from PIL import Image
from dgl.data.utils import save_graphs

from histocartography.utils import download_example_data
from histocartography.preprocessing import (
    ColorMergedSuperpixelExtractor,
    DeepFeatureExtractor
)
from histocartography.visualization import OverlayGraphVisualization

from skimage.measure import regionprops
import joblib
import cv2

import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import pandas as pd
from tqdm.auto import tqdm

import logging
import multiprocessing
import os
import sys
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
import pandas as pd
from tqdm.auto import tqdm

class PipelineStep(ABC):
    """Base pipelines step"""

    def __init__(
        self,
        save_path: Union[None, str, Path] = None,
        precompute: bool = True,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Abstract class that helps with saving and loading precomputed results
        Args:
            save_path (Union[None, str, Path], optional): Base path to save results to.
                When set to None, the results are not saved to disk. Defaults to None.
            precompute (bool, optional): Whether to perform the precomputation necessary
                for the step. Defaults to True.
            link_path (Union[None, str, Path], optional): Path to link the output directory
                to. When None, no link is created. Only supported when save_path is not None.
                Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save the output of
                the precomputation to. If not specified it defaults to the output directory
                of the step when save_path is not None. Defaults to None.
        """
        assert (
            save_path is not None or link_path is None
        ), "link_path only supported when save_path is not None"

        name = self.__repr__()
        self.save_path = save_path
        if self.save_path is not None:
            self.output_dir = Path(self.save_path) / name
            self.output_key = "default_key"
            self._mkdir()
            if precompute_path is None:
                precompute_path = save_path

        if precompute:
            self.precompute(
                link_path=link_path,
                precompute_path=precompute_path)

    def __repr__(self) -> str:
        """Representation of a pipeline step.
        Returns:
            str: Representation of a pipeline step.
        """
        variables = ",".join(
            [f"{k}={v}" for k, v in sorted(self.__dict__.items())])
        return (
            f"{self.__class__.__name__}({variables})".replace(" ", "")
            .replace('"', "")
            .replace("'", "")
            .replace("..", "")
            .replace("/", "_")
        )

    def _mkdir(self) -> None:
        """Create path to output files"""
        assert (
            self.save_path is not None
        ), "Can only create directory if base_path was not None when constructing the object"
        if not self.output_dir.exists():
            self.output_dir.mkdir()

    def _link_to_path(self, link_directory: Union[None, str, Path]) -> None:
        """Links the output directory to the given directory.
        Args:
            link_directory (Union[None, str, Path]): Directory to link to
        """
        if link_directory is None or Path(
                link_directory).parent.resolve() == Path(self.output_dir):
            logging.info("Link to self skipped")
            return
        assert (
            self.save_path is not None
        ), f"Linking only supported when saving is enabled, i.e. when save_path is passed in the constructor."
        if os.path.islink(link_directory):
            if os.path.exists(link_directory):
                logging.info("Link already exists: overwriting...")
                os.remove(link_directory)
            else:
                logging.critical(
                    "Link exists, but points nowhere. Ignoring...")
                return
        elif os.path.exists(link_directory):
            os.remove(link_directory)
        os.symlink(self.output_dir, link_directory, target_is_directory=True)

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information for this step
        Args:
            link_path (Union[None, str, Path], optional): Path to link the output to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to load/save the precomputation outputs. Defaults to None.
        """
        pass

    def process(
        self, *args: Any, output_name: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Main process function of the step and outputs the result. Try to saves the output when output_name is passed.
        Args:
            output_name (Optional[str], optional): Unique identifier of the passed datapoint. Defaults to None.
        Returns:
            Any: Result of the pipeline step
        """
        if output_name is not None and self.save_path is not None:
            return self._process_and_save(
                *args, output_name=output_name, **kwargs)
        else:
            return self._process(*args, **kwargs)

    @abstractmethod
    def _process(self, *args: Any, **kwargs: Any) -> Any:
        """Abstract method that performs the computation of the pipeline step
        Returns:
            Any: Result of the pipeline step
        """

    def _get_outputs(self, input_file: h5py.File) -> Union[Any, Tuple]:
        """Extracts the step output from a given h5 file
        Args:
            input_file (h5py.File): File to load from
        Returns:
            Union[Any, Tuple]: Previously computed output of the step
        """
        outputs = list()
        nr_outputs = len(input_file.keys())

        # Legacy, remove at some point
        if nr_outputs == 1 and self.output_key in input_file.keys():
            return tuple([input_file[self.output_key][()]])

        for i in range(nr_outputs):
            outputs.append(input_file[f"{self.output_key}_{i}"][()])
        if len(outputs) == 1:
            return outputs[0]
        else:
            return tuple(outputs)

    def _set_outputs(self, output_file: h5py.File,
                     outputs: Union[Tuple, Any]) -> None:
        """Save the step output to a given h5 file
        Args:
            output_file (h5py.File): File to write to
            outputs (Union[Tuple, Any]): Computed step output
        """
        if not isinstance(outputs, tuple):
            outputs = tuple([outputs])
        for i, output in enumerate(outputs):
            output_file.create_dataset(
                f"{self.output_key}_{i}",
                data=output,
                compression="gzip",
                compression_opts=9,
            )

    def _process_and_save(
        self, *args: Any, output_name: str, **kwargs: Any
    ) -> Any:
        """Process and save in the provided path as as .h5 file
        Args:
            output_name (str): Unique identifier of the the passed datapoint
        Raises:
            read_error (OSError): When the unable to read to self.output_dir/output_name.h5
            write_error (OSError): When the unable to write to self.output_dir/output_name.h5
        Returns:
            Any: Result of the pipeline step
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        output_path = self.output_dir / f"{output_name}.h5"
        if output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(output_path, "r") as input_file:
                    output = self._get_outputs(input_file=input_file)
            except OSError as read_error:
                print(f"\n\nCould not read from {output_path}!\n\n")
                raise read_error
        else:
            output = self._process(*args, **kwargs)
            try:
                with h5py.File(output_path, "w") as output_file:
                    self._set_outputs(output_file=output_file, outputs=output)
            except OSError as write_error:
                print(f"\n\nCould not write to {output_path}!\n\n")
                raise write_error
        return output

def fast_histogram(input_array: np.ndarray, nr_values: int) -> np.ndarray:
    """Calculates a histogram of a matrix of the values from 0 up to (excluding) nr_values
    Args:
        x (np.array): Input tensor
        nr_values (int): Possible values. From 0 up to (exclusing) nr_values.
    Returns:
        np.array: Output tensor
    """
    output_array = np.empty(nr_values, dtype=int)
    for i in range(nr_values):
        output_array[i] = (input_array == i).sum()
    return output_array


def load_image(image_path: Path) -> np.ndarray:
    """Loads an image from a given path and returns it as a numpy array
    Args:
        image_path (Path): Path of the image
    Returns:
        np.ndarray: Array representation of the image
    """
    assert image_path.exists()
    try:
        with Image.open(image_path) as img:
            image = np.array(img)
    except OSError as e:
        logging.critical("Could not open %s", image_path)
        raise OSError(e)
    return image

"""This module handles all the graph building"""

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import cv2
import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import load_graphs, save_graphs
from skimage.measure import regionprops
from sklearn.neighbors import kneighbors_graph

# from ..pipeline import PipelineStep
# from .utils import fast_histogram



LABEL = "label"
CENTROID = "centroid"
FEATURES = "feat"


def two_hop_neighborhood(graph: dgl.DGLGraph) -> dgl.DGLGraph:
    """Increases the connectivity of a given graph by an additional hop
    Args:
        graph (dgl.DGLGraph): Input graph
    Returns:
        dgl.DGLGraph: Output graph
    """
    A = graph.adjacency_matrix().to_dense()
    A_tilde = (1.0 * ((A + A.matmul(A)) >= 1)) - torch.eye(A.shape[0])
    ngraph = nx.convert_matrix.from_numpy_matrix(A_tilde.numpy())
    new_graph = dgl.DGLGraph()
    new_graph.from_networkx(ngraph)
    for k, v in graph.ndata.items():
        new_graph.ndata[k] = v
    for k, v in graph.edata.items():
        new_graph.edata[k] = v
    return new_graph


class BaseGraphBuilder(PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(
            self,
            nr_annotation_classes: int = 5,
            annotation_background_class: Optional[int] = None,
            add_loc_feats: bool = False,
            **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.
        Args:
            nr_annotation_classes (int): Number of classes in annotation. Used only if setting node labels.
            annotation_background_class (int): Background class label in annotation. Used only if setting node labels.
            add_loc_feats (bool): Flag to include location-based features (ie normalized centroids)
                                  in node feature representation.
                                  Defaults to False.
        """
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        super().__init__(**kwargs)

    def _process(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
    ) -> dgl.DGLGraph:
        """Generates a graph from a given instance_map and features
        Args:
            instance_map (np.array): Instance map depicting tissue components
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Union[None, np.array], optional): Optional node level to include.
                                                          Defaults to None.
        Returns:
            dgl.DGLGraph: The constructed graph
        """
        # add nodes
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add image size as graph data
        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        # get instance centroids
        centroids = self._get_node_centroids(instance_map)

        # add node content
        self._set_node_centroids(centroids, graph)
        self._set_node_features(features, image_size, graph)
        if annotation is not None:
            self._set_node_labels(instance_map, annotation, graph)

        # build edges
        self._build_topology(instance_map, centroids, graph)
        return graph

    def _process_and_save(  # type: ignore[override]
        self,
        instance_map: np.ndarray,
        features: torch.Tensor,
        annotation: Optional[np.ndarray] = None,
        output_name: str = None,
    ) -> dgl.DGLGraph:
        """Process and save in provided directory
        Args:
            output_name (str): Name of output file
            instance_map (np.ndarray): Instance map depicting tissue components
                                       (eg nuclei, tissue superpixels)
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Optional[np.ndarray], optional): Optional node level to include.
                                                         Defaults to None.
        Returns:
            dgl.DGLGraph: [description]
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.bin"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            graphs, _ = load_graphs(str(output_path))
            assert len(graphs) == 1
            graph = graphs[0]
        else:
            graph = self._process(
                instance_map=instance_map,
                features=features,
                annotation=annotation)
            save_graphs(str(output_path), [graph])
        return graph

    def _get_node_centroids(
            self, instance_map: np.ndarray
    ) -> np.ndarray:
        """Get the centroids of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
        Returns:
            centroids (np.ndarray): Node centroids
        """
        regions = regionprops(instance_map)
        centroids = np.empty((len(regions), 2))
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # (y, x)
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            centroids[i, 0] = center_x
            centroids[i, 1] = center_y
        return centroids

    def _set_node_centroids(
            self,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the centroids of the graphs
        Args:
            centroids (np.ndarray): Node centroids
            graph (dgl.DGLGraph): Graph to add the centroids to
        """
        graph.ndata[CENTROID] = torch.FloatTensor(centroids)

    def _set_node_features(
            self,
            features: torch.Tensor,
            image_size: Tuple[int, int],
            graph: dgl.DGLGraph
    ) -> None:
        """Set the provided node features
        Args:
            features (torch.Tensor): Node features
            image_size (Tuple[int,int]): Image dimension (x, y)
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)
        if not self.add_loc_feats:
            graph.ndata[FEATURES] = features
        elif (
                self.add_loc_feats
                and image_size is not None
        ):
            # compute normalized centroid features
            centroids = graph.ndata[CENTROID]

            normalized_centroids = torch.empty_like(centroids)  # (x, y)
            normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids \
                    .unsqueeze(dim=1) \
                    .repeat(1, features.shape[1], 1)
                concat_dim = 2
            elif features.ndim == 2:
                concat_dim = 1

            concat_features = torch.cat(
                (
                    features,
                    normalized_centroids
                ),
                dim=concat_dim,
            )
            graph.ndata[FEATURES] = concat_features
        else:
            raise ValueError(
                "Please provide image size to add the normalized centroid to the node features."
            )

    @abstractmethod
    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Generate the graph topology from the provided instance_map
        Args:
            instance_map (np.array): Instance map depicting tissue components
            centroids (np.array): Node centroids
            graph (dgl.DGLGraph): Graph to add the edges
        """

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information
        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "graphs")


class RAGGraphBuilder(BaseGraphBuilder):
    """
    Super-pixel Graphs class for graph building.
    """

    def __init__(self, kernel_size: int = 3, hops: int = 1, **kwargs) -> None:
        """Create a graph builder that uses a provided kernel size to detect connectivity
        Args:
            kernel_size (int, optional): Size of the kernel to detect connectivity. Defaults to 5.
        """
        logging.debug("*** RAG Graph Builder ***")
        assert hops > 0 and isinstance(
            hops, int
        ), f"Invalid hops {hops} ({type(hops)}). Must be integer >= 0"
        self.kernel_size = kernel_size
        self.hops = hops
        super().__init__(**kwargs)

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation map"""
        assert (
            self.nr_annotation_classes < 256
        ), "Cannot handle that many classes with 8-bits"
        regions = regionprops(instance_map)
        labels = torch.empty(len(regions), dtype=torch.uint8)

        for region_label in np.arange(1, len(regions) + 1):
            histogram = fast_histogram(
                annotation[instance_map == region_label],
                nr_values=self.nr_annotation_classes
            )
            mask = np.ones(len(histogram), np.bool)
            mask[self.annotation_background_class] = 0
            if histogram[mask].sum() == 0:
                assignment = self.annotation_background_class
            else:
                histogram[self.annotation_background_class] = 0
                assignment = np.argmax(histogram)
            labels[region_label - 1] = int(assignment)
        graph.ndata[LABEL] = labels

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Create the graph topology from the instance connectivty in the instance_map"""
        regions = regionprops(instance_map)
        instance_ids = torch.empty(len(regions), dtype=torch.uint8)

        kernel = np.ones((3, 3), np.uint8)
        adjacency = np.zeros(shape=(len(instance_ids), len(instance_ids)))

        for instance_id in np.arange(1, len(instance_ids) + 1):
            mask = (instance_map == instance_id).astype(np.uint8)
#             print("mask:{}".format(mask))
            dilation = cv2.dilate(mask,kernel, iterations=1)
#             print("dilation:{}".format(dilation))
            boundary = dilation - mask
#             print("boundary:{}".format(boundary))
#             print(sum(sum(boundary)))
            idx = pd.unique(instance_map[boundary.astype(bool)])
#             print("idx:{}".format(idx))
#             print(len(idx))
            instance_id -= 1  # because instance_map id starts from 1
            idx -= 1  # because instance_map id starts from 1
#             print("new idx:{}".format(idx))
#             print(type(idx))
            idx = idx.tolist()
#             print(type(idx))
            if -1 in idx:
                idx.remove(-1)
            idx = np.array(idx)
#             print(type(idx))
#             print("new new idx:{}".format(idx))
            if idx.shape[0] != 0:    
                adjacency[instance_id, idx] = 1
#         print(adjacency)

        edge_list = np.nonzero(adjacency)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]))

        for _ in range(self.hops - 1):
            graph = two_hop_neighborhood(graph)

"""This module handles everything related to superpixels"""

import logging
import math
import sys
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import cv2
import h5py
import numpy as np
from skimage.color.colorconv import rgb2hed
from skimage.future import graph
from skimage.segmentation import slic
from skimage.future import graph


class SuperpixelExtractor(PipelineStep):
    """Helper class to extract superpixels from images"""

    def __init__(
        self,
        nr_superpixels: int = None,
        superpixel_size: int = None,
        max_nr_superpixels: Optional[int] = None,
        blur_kernel_size: Optional[float] = 1,
        compactness: Optional[int] = 20,
        max_iterations: Optional[int] = 10,
        threshold: Optional[float] = 0.03,
        connectivity: Optional[int] = 2,
        color_space: Optional[str] = "rgb",
        downsampling_factor: Optional[int] = 1,
        **kwargs,
    ) -> None:
        """Abstract class that extracts superpixels from RGB Images
        Args:
            nr_superpixels (None, int): The number of super pixels before any merging.
            superpixel_size (None, int): The size of super pixels before any merging.
            max_nr_superpixels (int, optional): Upper bound for the number of super pixels.
                                                Useful when providing a superpixel size.
            blur_kernel_size (float, optional): Size of the blur kernel. Defaults to 0.
            compactness (int, optional): Compactness of the superpixels. Defaults to 30.
            max_iterations (int, optional): Number of iterations of the slic algorithm. Defaults to 10.
            threshold (float, optional): Connectivity threshold. Defaults to 0.03.
            connectivity (int, optional): Connectivity for merging graph. Defaults to 2.
            downsampling_factor (int, optional): Downsampling factor from the input image
                                                 resolution. Defaults to 1.
        """
        assert (nr_superpixels is None and superpixel_size is not None) or (
            nr_superpixels is not None and superpixel_size is None
        ), "Provide value for either nr_superpixels or superpixel_size"
        self.nr_superpixels = nr_superpixels
        self.superpixel_size = superpixel_size
        self.max_nr_superpixels = max_nr_superpixels
        self.blur_kernel_size = blur_kernel_size
        self.compactness = compactness
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.connectivity = connectivity
        self.color_space = color_space
        self.downsampling_factor = downsampling_factor
        super().__init__(**kwargs)

    def _process(  # type: ignore[override]
        self, input_image: np.ndarray, tissue_mask: np.ndarray = None
    ) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            input_image (np.array): Input image
            tissue_mask (None, np.array): Input tissue mask
        Returns:
            np.array: Extracted superpixels
        """
        logging.debug("Input size: %s", input_image.shape)
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor != 1:
            input_image = self._downsample(
                input_image, self.downsampling_factor)
            if tissue_mask is not None:
                tissue_mask = self._downsample(
                    tissue_mask, self.downsampling_factor)
            logging.debug("Downsampled to %s", input_image.shape)
        superpixels = self._extract_superpixels(
            image=input_image, tissue_mask=tissue_mask
        )
        if self.downsampling_factor != 1:
            superpixels = self._upsample(
                superpixels, original_height, original_width)
            logging.debug("Upsampled to %s", superpixels.shape)
        return superpixels

    @abstractmethod
    def _extract_superpixels(
        self, image: np.ndarray, tissue_mask: np.ndarray = None
    ) -> np.ndarray:
        """Perform the superpixel extraction
        Args:
            image (np.array): Input tensor
            tissue_mask (np.array): Tissue mask tensor
        Returns:
            np.array: Output tensor
        """

    @staticmethod
    def _downsample(image: np.ndarray, downsampling_factor: int) -> np.ndarray:
        """Downsample an input image with a given downsampling factor
        Args:
            image (np.array): Input tensor
            downsampling_factor (int): Factor to downsample
        Returns:
            np.array: Output tensor
        """
        height, width = image.shape[0], image.shape[1]
        new_height = math.floor(height / downsampling_factor)
        new_width = math.floor(width / downsampling_factor)
        downsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return downsampled_image

    @staticmethod
    def _upsample(
            image: np.ndarray,
            new_height: int,
            new_width: int) -> np.ndarray:
        """Upsample an input image to a speficied new height and width
        Args:
            image (np.array): Input tensor
            new_height (int): Target height
            new_width (int): Target width
        Returns:
            np.array: Output tensor
        """
        upsampled_image = cv2.resize(
            image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
        )
        return upsampled_image

    def precompute(
        self,
        link_path: Union[None, str, Path] = None,
        precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information
        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "superpixels")


class SLICSuperpixelExtractor(SuperpixelExtractor):
    """Use the SLIC algorithm to extract superpixels."""

    def __init__(self, **kwargs) -> None:
        """Extract superpixels with the SLIC algorithm"""
        super().__init__(**kwargs)

    def _get_nr_superpixels(self, image: np.ndarray) -> int:
        """Compute the number of superpixels for initial segmentation
        Args:
            image (np.array): Input tensor
        Returns:
            int: Output number of superpixels
        """
        if self.superpixel_size is not None:
            nr_superpixels = int(
                (image.shape[0] * image.shape[1] / self.superpixel_size)
            )
        elif self.nr_superpixels is not None:
            nr_superpixels = self.nr_superpixels
        if self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_superpixels(
            self,
            image: np.ndarray,
            *args,
            **kwargs) -> np.ndarray:
        """Perform the superpixel extraction
        Args:
            image (np.array): Input tensor
        Returns:
            np.array: Output tensor
        """
        if self.color_space == "hed":
            image = rgb2hed(image)
        nr_superpixels = self._get_nr_superpixels(image)
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=nr_superpixels,
            max_iter=self.max_iterations,
            compactness=self.compactness,
            start_label=1,
        )
        return superpixels


class MergedSuperpixelExtractor(SuperpixelExtractor):
    def __init__(self, **kwargs) -> None:
        """Extract superpixels with the SLIC algorithm"""
        super().__init__(**kwargs)

    def _get_nr_superpixels(self, image: np.ndarray) -> int:
        """Compute the number of superpixels for initial segmentation
        Args:
            image (np.array): Input tensor
        Returns:
            int: Output number of superpixels
        """
        if self.superpixel_size is not None:
            nr_superpixels = int(
                (image.shape[0] * image.shape[1] / self.superpixel_size)
            )
        elif self.nr_superpixels is not None:
            nr_superpixels = self.nr_superpixels
        if self.max_nr_superpixels is not None:
            nr_superpixels = min(nr_superpixels, self.max_nr_superpixels)
        return nr_superpixels

    def _extract_initial_superpixels(self, image: np.ndarray) -> np.ndarray:
        """Extract initial superpixels using SLIC
        Args:
            image (np.array): Input tensor
        Returns:
            np.array: Output tensor
        """
        nr_superpixels = self._get_nr_superpixels(image)
        superpixels = slic(
            image,
            sigma=self.blur_kernel_size,
            n_segments=nr_superpixels,
            compactness=self.compactness,
            max_iter=self.max_iterations,
            start_label=1,
        )
        return superpixels

    def _merge_superpixels(
        self,
        input_image: np.ndarray,
        initial_superpixels: np.ndarray,
        tissue_mask: np.ndarray = None,
    ) -> np.ndarray:
        """Merge the initial superpixels to return merged superpixels
        Args:
            image (np.array): Input image
            initial_superpixels (np.array): Initial superpixels
            tissue_mask (None, np.array): Tissue mask
        Returns:
            np.array: Output merged superpixel tensor
        """
        if tissue_mask is not None:
            # Remove superpixels belonging to background or having < 10% tissue
            # content
            ids_initial = np.unique(initial_superpixels, return_counts=True)
            ids_masked = np.unique(
                tissue_mask * initial_superpixels, return_counts=True
            )

            ctr = 1
            superpixels = np.zeros_like(initial_superpixels)
            for i in range(len(ids_initial[0])):
                id = ids_initial[0][i]
                if id in ids_masked[0]:
                    idx = np.where(id == ids_masked[0])[0]
                    ratio = ids_masked[1][idx] / ids_initial[1][i]
                    if ratio >= 0.1:
                        superpixels[initial_superpixels == id] = ctr
                        ctr += 1

            initial_superpixels = superpixels

        # Merge superpixels within tissue region
        g = graph.rag_mean_color(input_image, initial_superpixels)
        merged_superpixels = graph.merge_hierarchical(
            initial_superpixels,
            g,
            thresh=self.threshold,
            rag_copy=False,
            in_place_merge=True,
            merge_func=self._merging_function,
            weight_func=self._weighting_function,
        )
        merged_superpixels += 1  # Handle regionprops that ignores all values of 0
#         mask = np.zeros_like(initial_superpixels)
#         mask[initial_superpixels != 0] = 1
#         merged_superpixels = merged_superpixels * mask
        return merged_superpixels

    @abstractmethod
    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        """Handle merging of nodes of a region boundary region adjacency graph."""

    @abstractmethod
    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        """Call back called before merging 2 nodes."""

    def _extract_superpixels(
        self, image: np.ndarray, tissue_mask: np.ndarray = None
    ) -> np.ndarray:
        """Perform superpixel extraction
        Args:
            image (np.array): Input tensor
            tissue_mask (np.array, optional): Input tissue mask
        Returns:
            np.array: Extracted merged superpixels.
            np.array: Extracted init superpixels, ie before merging.
        """
        initial_superpixels = self._extract_initial_superpixels(image)
        merged_superpixels = self._merge_superpixels(
            image, initial_superpixels, tissue_mask
        )

        return merged_superpixels, initial_superpixels

    def _process(  # type: ignore[override]
        self, input_image: np.ndarray, tissue_mask=None
    ) -> np.ndarray:
        """Return the superpixels of a given input image
        Args:
            input_image (np.array): Input image.
            tissue_mask (None, np.array): Tissue mask.
        Returns:
            np.array: Extracted merged superpixels.
            np.array: Extracted init superpixels, ie before merging.
        """
        logging.debug("Input size: %s", input_image.shape)
        original_height, original_width, _ = input_image.shape
        if self.downsampling_factor is not None and self.downsampling_factor != 1:
            input_image = self._downsample(
                input_image, self.downsampling_factor)
            if tissue_mask is not None:
                tissue_mask = self._downsample(
                    tissue_mask, self.downsampling_factor)
            logging.debug("Downsampled to %s", input_image.shape)
        merged_superpixels, initial_superpixels = self._extract_superpixels(
            input_image, tissue_mask
        )
        if self.downsampling_factor != 1:
            merged_superpixels = self._upsample(
                merged_superpixels, original_height, original_width
            )
            initial_superpixels = self._upsample(
                initial_superpixels, original_height, original_width
            )
            logging.debug("Upsampled to %s", merged_superpixels.shape)
        return merged_superpixels, initial_superpixels

    def _process_and_save(
            self,
            *args: Any,
            output_name: str,
            **kwargs: Any) -> Any:
        """Process and save in the provided path as as .h5 file
        Args:
            output_name (str): Name of output file
        """
        assert (
            self.save_path is not None
        ), "Can only save intermediate output if base_path was not None when constructing the object"
        superpixel_output_path = self.output_dir / f"{output_name}.h5"
        if superpixel_output_path.exists():
            logging.info(
                f"{self.__class__.__name__}: Output of {output_name} already exists, using it instead of recomputing"
            )
            try:
                with h5py.File(superpixel_output_path, "r") as input_file:
                    merged_superpixels, initial_superpixels = self._get_outputs(
                        input_file=input_file)
            except OSError as e:
                print(
                    f"\n\nCould not read from {superpixel_output_path}!\n\n",
                    file=sys.stderr,
                    flush=True,
                )
                print(
                    f"\n\nCould not read from {superpixel_output_path}!\n\n",
                    flush=True)
                raise e
        else:
            merged_superpixels, initial_superpixels = self._process(
                *args, **kwargs)
            try:
                with h5py.File(superpixel_output_path, "w") as output_file:
                    self._set_outputs(
                        output_file=output_file,
                        outputs=(merged_superpixels, initial_superpixels),
                    )
            except OSError as e:
                print(
                    f"\n\nCould not write to {superpixel_output_path}!\n\n",
                    flush=True)
                raise e
        return merged_superpixels, initial_superpixels


class MyColorMergedSuperpixelExtractor(MergedSuperpixelExtractor):
    def __init__(
            self,
            w_hist: float = 0.5,
            w_mean: float = 0.5,
            **kwargs) -> None:
        """Superpixel merger based on color attibutes taken from the HACT-Net Implementation
        Args:
            w_hist (float, optional): Weight of the histogram features for merging. Defaults to 0.5.
            w_mean (float, optional): Weight of the mean features for merging. Defaults to 0.5.
        """
        self.w_hist = w_hist
        self.w_mean = w_mean
        super().__init__(**kwargs)

    def _color_features_per_channel(self, img_ch: np.ndarray) -> np.ndarray:
        """Extract color histograms from image channel
        Args:
            img_ch (np.ndarray): Image channel
        Returns:
            np.ndarray: Histogram of the image channel
        """
        hist, _ = np.histogram(img_ch, bins=np.arange(0, 257, 64))  # 8 bins
        return hist

    def _weighting_function(
        self, graph: graph.RAG, src: int, dst: int, n: int
    ) -> Dict[str, Any]:
        diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
        diff = np.linalg.norm(diff)
        return {'weight': diff}

    def _merging_function(self, graph: graph.RAG, src: int, dst: int) -> None:
        graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
        graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
        graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] /
                                      graph.nodes[dst]['pixel count'])