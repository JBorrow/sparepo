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
from typing import Dict, List, Tuple, Union

import attr
import h5py
import numpy as np

from sparepo.accelerated import ranges_from_array
from sparepo.particle_types import ParticleType
from sparepo.regions import SpatialRegion

try:
    import unyt

    unyt_available = True
except (ImportError, ModuleNotFoundError):
    # No unyt.
    unyt_available = False


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
    scale_factor: float

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
            self.scale_factor = header_attrs["ScaleFactor"]

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

    def unit_correction(
        self, part_type: ParticleType, field_name: str
    ) -> Tuple[Union[float, int], float, float, float]:
        """
        Gets the physical correction (i.e. removing a- and h- factors),
        based on the metadata available in the snapshot, as well
        as the scalings of individual units.

        Parameters
        ----------

        part_type: ParticleType
            Particle type to read. Example: ParticleType.Gas

        field_name: str
            Particle field to read. Example: Coordinates


        Returns
        -------

        correction_factor: Union[float, int]
            Appropriate correction factor. If the field is an integer,
            1 is returned as no integer fields should be scaled by
            any cosmological quantities (they are, e.g. particle IDs).

        length_scaling: float
            Unit length^length_scaling factor in the units.

        mass_scaling: float
            Unit mass^mass_scaling in the units.

        velocity_scaling: float
            Unit velocity^velocity_scaling in the units.
        """

        dataset_path = f"PartType{part_type.value}/{field_name}"

        with h5py.File(self.snapshot_filename_for_chunk(chunk=0), "r") as handle:
            dataset = handle[dataset_path]

            a_scale = self.scale_factor ** dataset.attrs["a_scaling"]
            h_scale = self.hubble_param ** dataset.attrs["h_scaling"]

            correction_factor: float = a_scale * h_scale

            length = float(dataset.attrs["length_scaling"])
            mass = float(dataset.attrs["mass_scaling"])
            velocity = float(dataset.attrs["velocity_scaling"])

            if dataset.dtype in (int, np.int64, np.int32, np.uint64, np.uint32):
                correction_factor: int = 1

        return correction_factor, length, mass, velocity

    def unyt_units(
        self, part_type: ParticleType, field_name: str
    ) -> "unyt.unyt_quantity":
        """
        Gets an ``unyt.unyt_quantity`` containing the appropriate units
        for the given field. Requires ``unyt`` be installed.

        Parameters
        ----------

        part_type: ParticleType
            Particle type to read. Example: ParticleType.Gas

        field_name: str
            Particle field to read. Example: Coordinates


        Returns
        -------

        units: unyt.unyt_quantity
            Units assocaited with this dataset.
        """

        if not unyt_available:
            raise ModuleNotFoundError(
                "Unyt not available, please install before using this feature."
            )

        correction, length, mass, velocity = self.unit_correction(
            part_type=part_type, field_name=field_name
        )
        dataset_path = f"PartType{part_type.value}/{field_name}"

        with h5py.File(self.snapshot_filename_for_chunk(chunk=0), "r") as handle:
            header = handle["Header"].attrs
            dtype = handle[dataset_path].dtype

            unit_length = unyt.unyt_quantity(header["UnitLength_in_cm"], "cm").to("kpc")
            unit_mass = unyt.unyt_quantity(header["UnitMass_in_g"], "g").to(
                "Solar_Mass"
            )
            unit_velocity = unyt.unyt_quantity(
                header["UnitVelocity_in_cm_per_s"], "cm/s"
            ).to("km/s")

        if length == 0.0 and mass == 0.0 and velocity == 0.0:
            units = unyt.unyt_quantity(correction * unyt.dimensionless, dtype=dtype)
        else:
            base_units = correction

            for base, power in zip(
                [unit_length, unit_mass, unit_velocity], [length, mass, velocity]
            ):
                if power != 0.0:
                    base_units *= base ** power

            units = unyt.unyt_quantity(base_units, dtype=dtype)

        return units

    def read_dataset(
        self,
        part_type: ParticleType,
        field_name: str,
        region: SpatialRegion,
        brutal: bool = True,
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

        brutal: bool, optional
            Brutal mode reads the whole chunk and then masks
            out the required data afterwards. This is faster on
            fast filesystems, but comes with signifciant memory
            overhead. Default: True, should only be turned off
            when memory is extremely constrained.

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

        for file_number, indices in file_mask.items():
            with h5py.File(
                self.snapshot_filename_for_chunk(chunk=file_number), "r"
            ) as handle:
                to_be_read_from_file = file_count[file_number]
                dataset = handle[dataset_path]

                # Sometimes, though, we do know better than the user. If there are
                # only a tiny number of particles in the file, let's not bother...
                # Yes, the number where brutal becomes worth it or not _really_
                # is this low. Tested by JB on 13 in MBA in Sep 2021.
                if brutal and to_be_read_from_file > 2048:
                    # In brutal mode, we just load the whole chunk and
                    # then index it ourselves... Although, we might as well
                    # only read the part of the array that we _actually_ need.
                    global_start = indices[0]
                    global_stop = indices[-1] + 1
                    size_of_range = to_be_read_from_file

                    output_dest_sel = np.s_[already_read : size_of_range + already_read]

                    full_read = dataset[global_start:global_stop]

                    output[output_dest_sel] = full_read[indices - global_start]

                    already_read += size_of_range

                    continue

                ranges = ranges_from_array(indices)

                # Reading individual particles with a call to `read_direct` is
                # exceptionally slow. So we cache ranges that contain only one
                # particle all together, until we get to a wider range. Then,
                # we read all the 'singles' and the range.
                read_contiguous = []

                for read_start, read_end in ranges:
                    if read_end == read_start:
                        continue

                    if read_end - 1 == read_start:
                        read_contiguous.append(read_start)
                    else:
                        # Need to deal with possible build up of previously contigous items.
                        if len(read_contiguous) > 0:
                            size_of_range = len(read_contiguous)

                            output_dest_sel = np.s_[
                                already_read : size_of_range + already_read
                            ]

                            dataset.read_direct(
                                output,
                                source_sel=read_contiguous,
                                dest_sel=output_dest_sel,
                            )

                            read_contiguous = []

                            already_read += size_of_range

                        # Now actually read the

                        # Because we read inclusively
                        size_of_range = read_end - read_start

                        # Construct selectors so we can use read_direct to prevent creating
                        # copies of data from the hdf5 file.
                        hdf5_read_sel = np.s_[read_start:read_end]
                        output_dest_sel = np.s_[
                            already_read : size_of_range + already_read
                        ]

                        dataset.read_direct(
                            output, source_sel=hdf5_read_sel, dest_sel=output_dest_sel
                        )

                        already_read += size_of_range

                # Any left over contiguous items.
                if len(read_contiguous) > 0:
                    size_of_range = len(read_contiguous)

                    output_dest_sel = np.s_[already_read : size_of_range + already_read]

                    dataset.read_direct(
                        output, source_sel=read_contiguous, dest_sel=output_dest_sel
                    )

                    read_contiguous = []

                    already_read += size_of_range

        assert already_read == particles_to_read

        return output

    def read_dataset_with_units(
        self,
        part_type: ParticleType,
        field_name: str,
        region: SpatialRegion,
        brutal: bool = True,
    ) -> "unyt.unyt_array":
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

        brutal: bool, optional
            Brutal mode reads the whole chunk and then masks
            out the required data afterwards. This is faster on
            fast filesystems, but comes with signifciant memory
            overhead. Default: True, should only be turned off
            when memory is extremely constrained.


        Returns
        -------

        dataset: unyt.unyt_array
            Particle dataset within the specified spatial region,
            with the associated units.


        Notes
        -----

        Requires that `unyt` be installed.
        """

        if not unyt_available:
            raise ModuleNotFoundError(
                "Unyt not available, please install before using this feature."
            )

        return unyt.unyt_array(
            self.read_dataset(
                part_type=part_type, field_name=field_name, region=region, brutal=brutal
            ),
            units=self.unyt_units(part_type=part_type, field_name=field_name),
            name=f"{part_type.name.title()} {field_name} (Physical, h-free)",
        )
