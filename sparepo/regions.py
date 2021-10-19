"""
Spatial region objects, used to select parts of the volume
to read from snapshots. All inerit from the base
:class:`SpatialRegion`, which should be used for type hints.

When implementing a new spatial region, ensure that you
implement ``set_cell_mask``, which sets the internal
``cell_mask`` and ``cells_to_use``, the cells that
should be read from file to encompass this region.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import attr
import h5py
import numpy as np

from sparepo.particle_types import ParticleType


@attr.s
class SpatialRegion:
    """
    Region to load _at least_ the particles
    inside of. Note that there may be extraneous
    particles loaded _outside_ of the region, but you
    are guaranteed to get all particles _inside_.

    Units should be the same as the internal units of
    ``SpatialLoader``, including the associated h-factors.

    Can only be associated with a single hashtable file per
    instance for performance reasons.
    """

    cell_mask: np.ndarray = attr.ib(init=False)
    cells_to_use: np.ndarray = attr.ib(init=False)
    wrap_table: np.ndarray = attr.ib(init=False)

    file_mask: Dict[ParticleType, Dict[int, np.ndarray]] = attr.ib(init=False)
    file_counts: Dict[ParticleType, Dict[int, int]] = attr.ib(init=False)
    file_wrapper: Dict[ParticleType, Dict[int, np.ndarray]] = attr.ib(init=False)

    mask_calculated: bool = attr.ib(init=False)

    def __attrs_post_init__(self):
        self.file_mask = {}
        self.file_counts = {}
        self.file_wrapper = {}

        # Do this here to ensure that we can have non-default fields.
        self.mask_calculated = False

    def set_cell_mask(self, centers: np.ndarray, cell_size: float):
        return

    def get_file_mask(
        self, hashtable: Path, part_type: ParticleType
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        """
        Computes and gets the file mask for an individual particle
        type. File masks are cached in the object.

        Parameters
        ----------

        hashtable: Path
            Path to the hashtable file.

        part_type: ParticleType
            Particle type to return the file-based table for.


        Returns
        -------

        file_mask: Dict[int, np.ndarray]
            Index masks per file for reading. If you want ranges,
            you must use ``ranges_from_array`` manually now.

        file_counts: Dict[int, int]
            Number of particles to be read from each file.

        file_wrapper: Dict[int, np.ndarray]
            Coordinates-shaped array of integers, from -1, 0, 1, indicating
            whether or not that given co-ordinate should be box-wrapped and
            in what direction.
        """

        if part_type in self.file_mask:
            return (
                self.file_mask[part_type],
                self.file_counts[part_type],
                self.file_wrapper[part_type],
            )

        file_raw_masks: Dict[int, List[np.ndarray]] = {}
        file_raw_wrappers: Dict[int, List[np.ndarray]] = {}
        file_counts: Dict[int, int] = {}

        # Do we need to create a wrapper at all?
        create_wrap_table = (self.wrap_table != 0).any()

        with h5py.File(hashtable, "r") as handle:
            for cell, wrap in zip(self.cells_to_use, self.wrap_table):
                cell_group = handle[f"PartType{part_type.value}/Cell{cell}"]

                for file, raw in cell_group.items():
                    file_number = raw.attrs["FileNumber"]
                    particles_in_cell = len(raw)

                    if create_wrap_table:
                        wrapper = (
                            np.repeat(wrap, repeats=particles_in_cell)
                            .reshape((len(wrap), particles_in_cell))
                            .T
                        )
                    else:
                        wrapper = None

                    if file_number in file_raw_masks:
                        file_raw_masks[file_number].append(raw[:])
                        file_raw_wrappers[file_number].append(wrapper)
                        file_counts[file_number] += particles_in_cell
                    else:
                        file_raw_masks[file_number] = [raw[:]]
                        file_raw_wrappers[file_number] = [wrapper]
                        file_counts[file_number] = particles_in_cell

        # Unpack the raw masks, sort them, and bin them together.
        file_mask = {}
        file_wrapper = {}

        for file in file_raw_masks.keys():
            file_raw_mask = file_raw_masks[file]
            file_raw_wrapper = file_raw_wrappers[file]

            concatenated_mask = np.concatenate(file_raw_mask)
            sort_order = concatenated_mask.argsort()

            file_mask[file] = concatenated_mask[sort_order]

            if create_wrap_table:
                file_wrapper[file] = np.concatenate(file_raw_wrapper)[sort_order]
            else:
                file_wrapper[file] = None

        self.file_mask[part_type] = file_mask
        self.file_counts[part_type] = file_counts
        self.file_wrapper[part_type] = file_wrapper

        return (
            self.file_mask[part_type],
            self.file_counts[part_type],
            self.file_wrapper[part_type],
        )


@attr.s
class CartesianSpatialRegion(SpatialRegion):
    """
    Region to load _at least_ the particles
    inside of. Note that there may be extraneous
    particles loaded _outside_ of the region, but you
    are guaranteed to get all particles _inside_.

    Units should be the same as the internal units of
    ``SpatialLoader``, including the associated h-factors.

    Can only be associated with a single hashtable file per
    instance for performance reasons.

    Loads data in a carteisan cube aligned with the simulation
    grid.

    Parameters
    ----------

    x: Tuple[float]
        Length 2 tuple containing the x limits to read between

    y: Tuple[float]
        Length 2 tuple containing the y limits to read between

    z: Tuple[float]
        Length 2 tuple containing the z limits to read between
    """

    x: Tuple[float] = attr.ib()
    y: Tuple[float] = attr.ib()
    z: Tuple[float] = attr.ib()

    def set_cell_mask(self, centers: np.ndarray, cell_size: float):
        """
        Sets the internal cell mask for the required region.

        Parameters
        ----------

        centers: np.ndarray
            Cell centers.

        cell_size: float
            Cell size in the same units as the centers.

        Notes
        -----

        Sets ``self.cell_mask`` and ``self.cells_to_load`` used by
        ``SpatialLoader``.
        """

        if self.mask_calculated:
            # We've already set ourself!
            return

        boxsize = centers.max() + 0.5 * cell_size

        individual_restrictors = []
        wrap_table = []

        for dimension, restriction in enumerate([self.x, self.y, self.z]):
            lower = restriction[0] - 0.5 * cell_size
            upper = restriction[1] + 0.5 * cell_size

            # Now need to deal with the three wrapping cases:
            if lower < 0.0:
                # Wrap lower -> high
                individual_restrictors.append(
                    np.logical_or(
                        centers[:, dimension] > lower + boxsize,
                        centers[:, dimension] < upper,
                    )
                )

                wrap_table.append(
                    -1 * (centers[:, dimension] > lower + boxsize).astype(int)
                )
            elif upper > boxsize:
                # Wrap high -> lower
                individual_restrictors.append(
                    np.logical_or(
                        centers[:, dimension] > lower,
                        centers[:, dimension] < upper - boxsize,
                    )
                )

                wrap_table.append(
                    1 * (centers[:, dimension] <= upper - boxsize).astype(int)
                )
            else:
                # No wrapping required
                individual_restrictors.append(
                    centers[:, dimension] > lower,
                )

                individual_restrictors.append(
                    centers[:, dimension] < upper,
                )

                wrap_table.append(np.zeros(len(centers), dtype=int))

        wrap_table = np.array(wrap_table).T

        self.cell_mask = np.logical_and.reduce(individual_restrictors)
        self.wrap_table = wrap_table[self.cell_mask]
        self.cells_to_use = np.where(self.cell_mask)[0]

        self.mask_calculated = True

        return


@attr.s
class SphericalSpatialRegion(SpatialRegion):
    """
    Region to load _at least_ the particles
    inside of. Note that there may be extraneous
    particles loaded _outside_ of the region, but you
    are guaranteed to get all particles _inside_.

    Units should be the same as the internal units of
    ``SpatialLoader``, including the associated h-factors.

    Can only be associated with a single hashtable file per
    instance for performance reasons.

    Loads data in a spherical region.

    Parameters
    ----------

    center: Tuple[float]
        The center of the sphere in [x, y, z].

    radius: float
        The radius from the center to load cells within.
    """

    center: Tuple[float] = attr.ib()
    radius: float = attr.ib(converter=float)

    def set_cell_mask(self, centers: np.ndarray, cell_size: float):
        """
        Sets the internal cell mask for the required region.

        Parameters
        ----------

        centers: np.ndarray
            Cell centers.

        cell_size: float
            Cell size in the same units as the centers.

        Notes
        -----

        Sets ``self.cell_mask`` and ``self.cells_to_load`` used by
        ``SpatialLoader``.
        """

        if self.mask_calculated:
            # We've already set ourself!
            return

        boxsize = centers.max() + 0.5 * cell_size

        individual_restrictors = []
        wrap_table = []

        for dimension, restriction in enumerate(
            [[c - self.radius, c + self.radius] for c in self.center]
        ):
            lower = restriction[0] - 0.5 * cell_size
            upper = restriction[1] + 0.5 * cell_size

            # Now need to deal with the three wrapping cases:
            if lower < 0.0:
                # Wrap lower -> high
                individual_restrictors.append(
                    np.logical_or(
                        centers[:, dimension] > lower + boxsize,
                        centers[:, dimension] < upper,
                    )
                )

                wrap_table.append(
                    -1 * (centers[:, dimension] > lower + boxsize).astype(int)
                )
            elif upper > boxsize:
                # Wrap high -> lower
                individual_restrictors.append(
                    np.logical_or(
                        centers[:, dimension] > lower,
                        centers[:, dimension] < upper - boxsize,
                    )
                )

                wrap_table.append(
                    1 * (centers[:, dimension] <= upper - boxsize).astype(int)
                )
            else:
                # No wrapping required
                individual_restrictors.append(
                    centers[:, dimension] > lower,
                )

                individual_restrictors.append(
                    centers[:, dimension] < upper,
                )

                wrap_table.append(np.zeros(len(centers), dtype=int))

        wrap_table = np.array(wrap_table).T

        self.cell_mask = np.logical_and.reduce(individual_restrictors)
        self.wrap_table = wrap_table[self.cell_mask]
        self.cells_to_use = np.where(self.cell_mask)[0]

        self.mask_calculated = True

        return
