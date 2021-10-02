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

from sparepo.accelerated import ranges_from_array
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

    cell_mask: np.ndarray
    cells_to_use: np.ndarray

    file_mask: Dict[ParticleType, Dict[int, np.ndarray]]
    file_counts: Dict[ParticleType, Dict[int, int]]

    mask_calculated: bool

    def __attrs_post_init__(self):
        self.file_mask = {}
        self.file_counts = {}

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
            Range masks per file for reading.

        file_counts: Dict[int, int]
            Number of particles to be read from each file.
        """

        if part_type in self.file_mask:
            return self.file_mask[part_type], self.file_counts[part_type]

        file_raw_masks: Dict[int, List[np.ndarray]] = {}
        file_counts: Dict[int, int] = {}

        with h5py.File(hashtable, "r") as handle:
            for cell in self.cells_to_use:
                print
                cell_group = handle[f"PartType{part_type.value}/Cell{cell}"]

                for file, raw in cell_group.items():
                    file_number = raw.attrs["FileNumber"]

                    if file_number in file_raw_masks:
                        file_raw_masks[file_number].append(raw[:])
                        file_counts[file_number] += len(raw)
                    else:
                        file_raw_masks[file_number] = [raw[:]]
                        file_counts[file_number] = len(raw)

        # Unpack the raw masks, sort them, and bin them together.
        file_mask = {}

        for file, file_raw_mask in file_raw_masks.items():
            file_mask[file] = ranges_from_array(np.sort(np.concatenate(file_raw_mask)))

        self.file_mask[part_type] = file_mask
        self.file_counts[part_type] = file_counts

        return self.file_mask[part_type], self.file_counts[part_type]


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

    mask_calculated = False

    cell_mask: np.ndarray
    cells_to_use: np.ndarray

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

        individual_restrictors = []

        for dimension, restriction in enumerate([self.x, self.y, self.z]):
            lower = restriction[0] - 0.5 * cell_size
            upper = restriction[1] + 0.5 * cell_size

            individual_restrictors.append(
                centers[:, dimension] > lower,
            )

            individual_restrictors.append(
                centers[:, dimension] < upper,
            )

        self.cell_mask = np.logical_and.reduce(individual_restrictors)
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

    mask_calculated = False

    cell_mask: np.ndarray
    cells_to_use: np.ndarray

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

        individual_restrictors = []

        for dimension, restriction in enumerate(
            [[c - self.radius, c + self.radius] for c in self.center]
        ):
            lower = restriction[0] - 0.5 * cell_size
            upper = restriction[1] + 0.5 * cell_size

            individual_restrictors.append(
                centers[:, dimension] > lower,
            )

            individual_restrictors.append(
                centers[:, dimension] < upper,
            )

        self.cell_mask = np.logical_and.reduce(individual_restrictors)
        self.cells_to_use = np.where(self.cell_mask)[0]

        self.mask_calculated = True

        return
