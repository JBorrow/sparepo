SpAREPO
=======

[![PyPI version](https://badge.fury.io/py/sparepo.svg)](https://badge.fury.io/py/sparepo)

Spatial hasthtable building for AREPO. Built to be extremely lightweight
and low-code, with an easily understandable hashtable file.

Purpose
-------

The AREPO snapshots are stored by halo, rather than by spatial region. This
makes it complex to load _spatially_, for instance getting all particles within a
10 Mpc radius of a given point. To do so typically requires loading all data in all
snapshot chunks, and rejecting the vast majority of the data. It also means that
even if you just want some particle property (like e.g. the internal energies), you
still need to load all of the co-ordinates.

`sparepo` solves this by building a 'hashtable' file, which coarse-grains the
particles onto a top-level grid, by interating through the file once. Then, at
any point afterwards, you can request any set of cells to be read out
of the snapshot, already 'knowing' which files (and which positions in file!)
that the particles are in.

`sparepo` provides utilities to build the hashtables, but also provides the
utilities to read said hashtables in python. The file format, described below,
for the hashtable is simple so it can be used in other
implementations if required.

Requirements
------------

It is recommended that you use a recent version of python for `sparepo`,
at least 3.8 (as that is the lowest that we will test on), and no effort is
made to maintain compatibility with any versions of python before this.
`sparepo` requires:

+ `numba` (and hence `llvmlite`)
+ `numpy`
+ `h5py`
+ `attrs`

File formatting is taken care of by `black` and `isort`.

File Format
-----------

The spatial hashtable that is created has the following file
structure:

```
Header/
    Attrs: {
        BoxSize: The box size in given units including h factors.
        NumberOfChunks: M
        Units: Units that length scales are in
        HubbleParam: Hubble parameter
        HubbleParamScaling: For the length units, the exponent of h
    }
Cells/
    Centrers: Nx3 Array of Cell Centers
    Counts/
        PartTypeX : NxM length array of total counts, with M the number of chunks.
    Attrs: {
        Size: 1D size of the cells
        NumberOfCells: N, Total number of cells
        CellsPerAxis: cbrt(N), number of cells per axis.
    }
PartTypeX/
    CellY/
        FileZ: Length O array of indicies.
            Attrs: {
                FileName: Pathless filename of this file.
                FileNumber: Integer file number of this file, helpful
                            for indexing the arrays.
            }
```

with the indexing into the cell array being specified as:

```python
x_cell * (number_of_cells)**2 + y_cell * number_of_cells + z_cell
```


Hashtable Creation
------------------

Creating a hashtable can be done using the `create_hashtable` function,

```python
from sparepo import create_hashtable

create_hashtable(
    snapshot="snapdir_099/snapshot_099.0.hdf5",
    cells_per_axis=14,
    hashtable="snapdir_099/spatial_hashtable_099.hdf5"
)
```
This may take some time, as you might expect. For a `240^3` box, it takes
a few seconds and should _in principle_ scale linearly.


Hashtable Reading
-----------------

Reading from the hashtable is again designed to be simple. Currently
two loading strategies are implemented:

+ `CartesianSpatialRegion(x=[low, high], y=[low, high], z=[low, high])`
+ `SphericalSpatialRegion(center=[x, y, z], radius=r)`

These can then be used with the `SpatialLoader` object to load particles
from file. Note that the majority of the data (the post-processed
hashtables read from file) are stored in the region objects.

```python
from sparepo import SphericalSpatialRegion, SpatialLoader, ParticleType

region = SphericalSpatialRegion(
    center=[16000.0, 16000.0, 16000.0],
    radius=6000.0
)

loader = SpatialLoader(
    hashtable="snapdir_099/spatial_hashtable_099.hdf5",
    snapshot="snapdir_099/snapshot_099.0.hdf5",
)

start = time.time()

x, y, z = loader.read_dataset(
    ParticleType.GAS, field_name="Coordinates", region=region
).T


```

This will load cells containing _at least_ the particles in a sphere
centered on `[16000.0, 16000.0, 16000.0]` with radius `6000.0`. Additional
particles will definitel be loaded, as the loading is cell-by-cell rather
than particle-by-particle for performance reasons. If you require a strict
mask, we encourage you to do that by post-processing the co-ordinates.

The main thing to note here is that particle types are accesed through
the `ParticleType` enum, rather than the usual passing of 'magical'
integers.

The second thing to note is that the first time that `read_dataset` is
called it builds a compressed hashtable and reads the required data
from the hashtable from file (which is then cached), though the time
to do this is typically shorter than the time required to read the
data from file.

Note that the reading performance here is actually limited by the
loop over indicies (and having to call a h5py read for each of them).
Contiguous ranges are read together, which improves performance
significantly, so the read performance is actually entirely limited
by the data locality. More complex reading schemes may be able
to vastly improve the speed of data loading.
