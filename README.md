SpAREPO
========

Spatial hasthtable building for AREPO. Built to be extremely lightweight
and low-code, with an easily understandable hashtable file.

Purpose
-----------

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
for the hashtable is simple, though, so it can be used in other
implementations if required.

Requirements
-------------------

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
---------------

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
        FileZ: Ox2 array of slices.
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
--------------------------

Hashtable Reading
--------------------------

