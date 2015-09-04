import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")


def test_rand_array():

    ns = 5

    A = np.random.standard_normal((ns, ns)).astype(np.float64)
    A0 = comm.bcast(A, root=0)
    assert np.allclose(A, A0)
    # if rank == 0:
    #     assert np.allclose(A, A0)


if __name__ == '__main__':
    test_rand_array()
