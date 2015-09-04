import numpy as np
from mpi4py import MPI
from scalapy import core


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

core.initmpi([2, 2], block_shape=[3, 3])


def test_from_global_array():
    ns = 5 # matrix size
    prank = 0 # process rank

    if rank == prank:
        A = np.random.standard_normal((ns, ns)).astype(np.float64)
        B = A
        assert not np.isfortran(A) # assert order='C'
        assert not np.isfortran(B) # assert order='C'
    else:
        A = None
        B = None

    A = comm.bcast(A, root=prank) # every process has a copy of A
    assert not np.isfortran(A) # assert order='C'

    dA = core.DistributedMatrix.from_global_array(A, rank=None)
    gA = dA.to_global_array()
    assert not np.isfortran(A) # assert order='C', so A does not changed at all
    assert np.allclose(A, gA)


    dB = core.DistributedMatrix.from_global_array(B, rank=prank)
    gB = dB.to_global_array()
    if rank == prank:
        assert not np.isfortran(B) # assert order='C', so B does not changed at all
        assert np.allclose(B, gB)


    assert np.allclose(dA.local_array, dB.local_array)
    assert np.allclose(gA, gB)


if __name__ == '__main__':
    test_from_global_array()
