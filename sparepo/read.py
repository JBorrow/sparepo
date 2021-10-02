"""
Reader for the hashtable, in combination with the
:class:`SpatialRegion` objects from ``regions.py``.

Use the :class:`SpatialLoader` class to set up and
read from the hashtables.

Note that all large data is actually contained in the
region objects, and the loader class is really just
a convenience object.
"""

from pathlib import Path
from typing import Dict, List

import attr
import h5py
import numpy as np

from sparepo.particle_types import ParticleType
from sparepo.regions import SpatialRegion


@attr.s
class ChunkFileHashtable:
    """
    Hashtable for a single chunk file and particle type.
    """

    filename: Path = attr.ib(converter=Path)
    file_number: int = attr.ib(converter=int)

    hashtable: np.ndarray = attr.ib()


@attr.s
class SpatialLoader:
    """
    Spatially load data from files based on the pre-generated
    hashtable. If you need to create a hashtable, see the
    ``build_hashtable.py``.

    Note that there is no built-in periodic wrapping.

    Parameters
    ----------

    hashtable: Path
        Path to the hashtable hdf5 file.

    snapshot: Path
        Path to the first snapshot (the one including ``.0.hdf5``)
    """

    hashtable: Path = attr.ib(converter=Path)
    snapshot: Path = attr.ib(converter=Path)

    box_size: float
    number_of_chunks: int
    unit: str
    hubble_param: float
    hubble_param_scaling: int

    available_part_types: List[ParticleType]
    centers: np.ndarray
    counts: Dict[ParticleType, np.ndarray]
    cell_size: float
    number_of_cells: int
    cells_per_axis: int

    def __attrs_post_init__(self):
        """
        Loads in metadata from the hashtable.
        """

        with h5py.File(self.hashtable, "r") as handle:
            header_attrs = handle["Header"].attrs

            cell_centers = handle["Cells/Centers"][...]
            cell_counts = {
                ParticleType(int(name[-1])): value[:]
                for name, value in handle["Cells/Counts"].items()
            }

            cell_attrs = handle["Cells"].attrs

            self.box_size = header_attrs["BoxSize"]
            self.number_of_chunks = header_attrs["NumberOfChunks"]
            self.unit = header_attrs["Units"]
            self.hubble_param = header_attrs["HubbleParam"]
            self.hubble_param_scaling = header_attrs["HubbleParamScaling"]

            self.centers = cell_centers
            self.counts = cell_counts
            self.available_part_types = list(cell_counts.keys())
            self.cell_size = cell_attrs["Size"]
            self.number_of_cells = cell_attrs["NumberOfCells"]
            self.cells_per_axis = cell_attrs["CellsPerAxis"]

    def snapshot_filename_for_chunk(self, chunk: int):
        """
        Gets the snapshot filename for a given chunk.
        """

        return self.snapshot.parent / (
            self.snapshot.stem.split(".")[0] + f".{chunk}.hdf5"
        )

    def read_dataset(
        self,
        part_type: ParticleType,
        field_name: str,
        region: SpatialRegion,
    ) -> np.ndarray:
        """
        Reads a dataset in a given spatial region.

        Parameters
        ----------

        part_type: ParticleType
            Particle type to read. Example: ParticleType.Gas

        field_name: str
            Particle field to read. Example: Coordinates

        region: SpatialRegion
            Spatial region to load data within.


        Returns
        -------

        dataset: np.ndarray
            Particle dataset within the specified spatial region.
        """

        if not region.mask_calculated:
            region.set_cell_mask(
                centers=self.centers,
                cell_size=self.cell_size,
            )

        # First, read out the cell data from the hashtable file.
        # This is one contiguous read so doesn't need to be cached,
        # as relative to the particle data reading it is very fast.

        file_mask, file_count = region.get_file_mask(
            hashtable=self.hashtable, part_type=part_type
        )

        particles_to_read = sum(file_count.values())
        dataset_path = f"PartType{part_type.value}/{field_name}"

        with h5py.File(self.snapshot, "r") as handle:
            dataset = handle[dataset_path]

            shape = list(dataset.shape)
            dtype = dataset.dtype

        # Truncate the shape
        shape[0] = particles_to_read

        output = np.empty(shape, dtype=dtype)
        already_read = 0

        for file_number, ranges in file_mask.items():
            with h5py.File(
                self.snapshot_filename_for_chunk(chunk=file_number), "r"
            ) as handle:
                dataset = handle[dataset_path]
                for read_start, read_end in ranges:
                    if read_end == read_start:
                        continue

                    # Because we read inclusively
                    size_of_range = read_end - read_start

                    # Construct selectors so we can use read_direct to prevent creating
                    # copies of data from the hdf5 file.
                    hdf5_read_sel = np.s_[read_start:read_end]
                    output_dest_sel = np.s_[already_read : size_of_range + already_read]

                    dataset.read_direct(
                        output, source_sel=hdf5_read_sel, dest_sel=output_dest_sel
                    )

                    already_read += size_of_range

        return output
