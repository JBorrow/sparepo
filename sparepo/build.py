"""
Builds the spatial hashtable for a set of chunks.

For all of this, we only ever need to actually load in one particle type from
one chunk at a time. How?

1. Load in all co-ordinates that are relevant from that one chunk.
2. Hashtable it up
3. Invert this hashtable for the file.
4. Write the hashtable to PartTypeX/Cell{Y..Z}/FileA.

This is then performed for all of the files, and the global quantities (e.g.
PartTypeTotals) are saved to disk.
"""


import os
from pathlib import Path
from typing import Any

import attr
import h5py
import numpy as np

from sparepo.accelerated import (compute_particle_ids_binned_by_cell,
                                 count_cells)
from sparepo.particle_types import ParticleType

# Do not build the hashtable for the tracers as those
# work differenty.
part_types_to_use = (
    ParticleType.GAS,
    ParticleType.DARK_MATTER,
    ParticleType.STAR,
    ParticleType.BLACK_HOLE,
)


@attr.s
class CellStructure:
    """
    Cell information and methods to use when creating
    hashtables.

    Parameters
    ----------

    box_size: float
        The box-size of the simulation, in the same units
        as the co-ordinates to be read out of the box.

    cells_per_axis: int
        Number of cells per axis.


    Notes
    -----

    Cell indicies are hashed as:
        ``x * (chunks)**2 + y * (chunk) + z``

    """

    box_size: float = attr.ib(converter=float)
    cells_per_axis: int = attr.ib(converter=int)

    @property
    def cell_size(self) -> float:
        return self.box_size / self.cells_per_axis

    def write_metadata_to_hdf5(self, handle=h5py.Group):
        cells = handle.create_group("Cells")

        for key, value in {
            "NumberOfCells": self.cells_per_axis ** 3,
            "Size": self.cell_size,
            "CellsPerAxis": self.cells_per_axis,
        }.items():
            cells.attrs.create(key, value)

        # Thanks to our good friend numpy array ordering, [y, x, z] corresponds to our
        # Hashing.
        x, y, z = np.meshgrid(
            *[(np.arange(self.cells_per_axis) + 0.5) * self.cell_size] * 3
        )
        cell_centers = np.array([y.flatten(), x.flatten(), z.flatten()]).T

        cells.create_dataset("Centers", data=cell_centers)

        return


@attr.s
class FileMetadata:
    """
    Basic metadata about the snapshots.

    Parameters
    ----------

    filename: Path
        Filename to any one of the files in the snapshot.

    """

    filename: Path = attr.ib(converter=Path)

    def read_header_attr(self, name: str) -> Any:
        with h5py.File(self.filename, "r") as handle:
            return handle["Header"].attrs[name]

    @property
    def box_size(self) -> float:
        return float(self.read_header_attr(name="BoxSize"))

    @property
    def number_of_chunks(self) -> int:
        return int(self.read_header_attr(name="NumFilesPerSnapshot"))

    @property
    def hubble_param(self) -> float:
        return float(self.read_header_attr(name="HubbleParam"))

    @property
    def unit(self) -> str:
        return f"{self.read_header_attr(name='UnitLength_in_cm')} cm"

    @property
    def hubble_param_scaling(self) -> int:
        return -1

    @property
    def scale_factor(self) -> float:
        return float(self.read_header_attr(name="Time"))

    @property
    def base_path(self) -> Path:
        """
        Provides the total base path as a string. So if
        ``filename = "test/this/path/file.0.hdf5"``, this would
        return ``test/this/path/file``.
        """
        return self.filename.parent / self.filename.stem.split(".")[0]

    def write_metadata_to_hdf5(self, handle: h5py.Group):
        header = handle.create_group("Header")

        for key, value in {
            "BoxSize": self.box_size,
            "NumberOfChunks": self.number_of_chunks,
            "HubbleParam": self.hubble_param,
            "Units": self.unit,
            "HubbleParamScaling": self.hubble_param_scaling,
            "ScaleFactor": self.scale_factor,
        }.items():
            header.attrs.create(key, value)

        return


def compute_hashtable_for_file(
    filename: Path,
    cell_structure: CellStructure,
):
    """
    Individual hashtable computation for a given file.

    Parameters
    ----------

    filename: Path
        Filename to create the hashtable for.

    cell_structure: CellStructure
        Structure containing cell metadata.

    Returns
    -------

    cell_counts: Dict[ParticleType, np.array]
        Counts for all particle types in the cells in an array
        hashed as described in ``CellStructure``.

    cell_table: Dict[ParticleType, Dict[int, np.ndarray]]
        Hashtable given as [PartType][Cell] = [range].
    """

    cell_counts = {}
    cell_table = {}

    cells_per_axis = cell_structure.cells_per_axis

    for part_type in part_types_to_use:
        with h5py.File(filename, "r") as handle:
            if f"PartType{part_type.value}" not in handle:
                continue

            coordinates = handle[f"/PartType{part_type.value}/Coordinates"][:]

        cells_byte = (coordinates / cell_structure.cell_size).astype(np.int16)
        cells = (
            cells_byte[:, 0] * (cells_per_axis * cells_per_axis)
            + cells_byte[:, 1] * cells_per_axis
            + cells_byte[:, 2]
        )
        counts = count_cells(cells=cells, cells_per_axis=cell_structure.cells_per_axis)

        particle_ids = compute_particle_ids_binned_by_cell(
            cells=cells,
            counts=counts,
        )

        # Remove the cruft.
        clean_particle_ids = {}

        for cell_id, particle_table in enumerate(particle_ids):
            if len(particle_table) > 0:
                clean_particle_ids[cell_id] = particle_table

        cell_counts[part_type] = counts
        cell_table[part_type] = clean_particle_ids

    return cell_counts, cell_table


def create_hashtable(
    snapshot: Path,
    cells_per_axis: int,
    hashtable: Path,
):
    """
    Creates the hashtable file and saves it to disk.
    Note that the file will be opened and closed multiple times
    during the creation process.

    snapshot: Path
        Path to the first snapshot file (``.0.hdf5``)

    cells_per_axis: int
        Number of cells across each spatial axis. The cell size
        will hence be ``BoxSize / cells_per_axis``.

    hashtable: Path
        Filename to write the hashtable to.
    """

    snapshot = Path(snapshot)
    hashtable = Path(hashtable)

    metadata = FileMetadata(filename=snapshot)

    files = {
        file_number: Path(f"{metadata.base_path}.{file_number}.hdf5")
        for file_number in range(metadata.number_of_chunks)
    }

    cell_structure = CellStructure(
        box_size=metadata.box_size,
        cells_per_axis=cells_per_axis,
    )

    counts_by_file = []

    # Remove the output file before we start messing with it, otherwise
    # we're going to get a partial overwrite in the best case and a
    # crash in the worst (or the other way around, if I think about it...)
    if hashtable.exists():
        os.remove(hashtable)

    # Start with metadata
    with h5py.File(hashtable, "w") as handle:
        metadata.write_metadata_to_hdf5(handle=handle)
        cell_structure.write_metadata_to_hdf5(handle=handle)

    for file_number, filename in files.items():
        counts, particle_ids = compute_hashtable_for_file(
            filename=filename, cell_structure=cell_structure
        )

        counts_by_file.append(counts)

        with h5py.File(hashtable, "a") as handle:
            for part_type in part_types_to_use:
                part_type_string = f"PartType{part_type.value}"

                if not part_type_string in handle:
                    part_type_group = handle.create_group(part_type_string)
                else:
                    part_type_group: h5py.Group = handle[part_type_string]

                for cell, table in particle_ids[part_type].items():
                    cell_string = f"Cell{cell}"

                    if not cell_string in part_type_group:
                        cell_group = part_type_group.create_group(cell_string)
                    else:
                        cell_group: h5py.Group = part_type_group[cell_string]

                    file_dataset = cell_group.create_dataset(
                        f"File{file_number}", data=table, compression="gzip"
                    )

                    file_dataset.attrs.create("FileName", filename.parts[-1])
                    file_dataset.attrs.create("FileNumber", file_number)

    # Finish off by creating an array of counts per file
    for part_type in part_types_to_use:
        output_count_array = np.empty(
            (cell_structure.cells_per_axis ** 3, metadata.number_of_chunks),
            dtype=np.int32,
        )

        for file_number, counts in enumerate(counts_by_file):
            output_count_array[:, file_number] = counts[part_type][:]

        with h5py.File(hashtable, "a") as handle:
            if not "Cells/Counts" in handle:
                counts_group = handle.create_group("Cells/Counts")
            else:
                counts_group: h5py.Group = handle["Cells/Counts"]

            counts_group.create_dataset(
                f"PartType{part_type.value}", data=output_count_array
            )
