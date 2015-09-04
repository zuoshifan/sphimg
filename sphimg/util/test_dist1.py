import numpy as np
from mpi4py import MPI
from scalapy import core


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

# if size != 4:
#     raise Exception("Test needs 4 processes.")

# core.initmpi([2, 2], block_shape=[3, 3])

grid_shape = [1, 2]
pc1 = core.ProcessContext(grid_shape, comm=comm) # process context
pc = core.ProcessContext(grid_shape, comm=comm.Dup()) # process context


def test_from_global_array():
    ns = 128 # matrix size
    prank = 0 # process rank

    if rank == prank:
        A = np.random.standard_normal((ns, ns)).astype(np.float64)
        assert not np.isfortran(A) # assert order='C'
        A = np.asfortranarray(A)
    else:
        A = None



    dA = core.DistributedMatrix.from_global_array(A, rank=0, block_shape=[64, 64], context=pc)
    gA = dA.to_global_array()
    if rank == prank:
        assert np.allclose(A, gA)


if __name__ == '__main__':
    test_from_global_array()
