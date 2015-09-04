import numpy as np
import scipy.linalg as la
from mpi4py import MPI
from scalapy import core
import scalapy.routines as rt


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 16:
    raise Exception("Test needs %d processes." % size)

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)

# matrix size
N = 50000

l1 = np.random.uniform(1.0, N, N)
l2 = np.random.uniform(1.0, N, N)

core.initmpi([4, 4], block_shape=[N/size, N/size])
A = core.DistributedMatrix([N, N], dtype=np.complex128)
(g,r,c) = A.local_diagonal_indices()
A.local_array[r,c] += l1[g]
B = core.DistributedMatrix([N, N], dtype=np.complex128)
(g,r,c) = B.local_diagonal_indices()
B.local_array[r,c] += l2[g]

evals, evecsd= rt.eigh(A, B)
if rank == 0:
    # print evals
    print 'Done'


# temp = np.random.standard_normal((N, N)).astype(np.float64)
# temp = temp + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
# temp = temp + temp.T.conj() # Make Hermitian
# r1 = la.eigh(temp)[1]
# assert allclose(la.inv(r1), r1.T.conj()) # assert r1 unitary

# temp = np.random.standard_normal((ns, ns)).astype(np.float64)
# temp = temp + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
# temp = temp + temp.T.conj() # Make Hermitian
# r2 = la.eigh(temp)[1]
# assert allclose(la.inv(r2), r2.T.conj()) # assert r2 unitary

# A = None
# B = None
# rk = 1 if size >=2 else 0
# # rk = 0
# if rank == 0:
#     A = np.dot(np.dot(r1, np.diag(l1)), r1.T.conj())
#     A = np.asfortranarray(A)
#     assert allclose(A, A.T.conj())
# if rank == rk:
#     B = np.dot(np.dot(r2, np.diag(l2)), r2.T.conj())
#     B = np.asfortranarray(B)
#     assert allclose(B, B.T.conj())

# # distribute
# core.initmpi([2, 2], block_shape=[3, 3])
# dA = core.DistributedMatrix.from_global_array(A, rank=0)
# dB = core.DistributedMatrix.from_global_array(B, rank=rk)

# # eigen decomposition
# # evalsd, evecsd = su.eigh_gen(dA, dB)
# evalsd, evecsd = rt.eigh(dA, dB, overwrite_a=False, overwrite_b=False, eigvals=(0, 4))
# print evalsd

# # distribute assert
# LA = rt.dot(rt.dot(evecsd, dA, transA='C', transB='N'), evecsd, transA='N', transB='N')
# LA = LA.to_global_array(rank=0)
# LB = rt.dot(rt.dot(evecsd, dB, transA='C', transB='N'), evecsd, transA='N', transB='N')
# LB = LB.to_global_array(rank=0)
# if rank == 0:
#     # asser equal diagonal elements
#     assert allclose(evalsd, np.diag(LA).real)
#     print np.diag(LB).real
#     assert allclose(np.ones_like(evalsd), np.diag(LB).real)
#     # assert equal matrix
#     assert allclose(np.diag(evalsd).astype(LA.dtype), LA)
#     assert allclose(np.eye(LB.shape[0], dtype=LB.dtype), LB)
