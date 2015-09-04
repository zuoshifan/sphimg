import numpy as np
import scipy.linalg as la
from mpi4py import MPI
from scalapy import core
import scalapy.routines as rt
# import scalapyutil as su


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)

l1 = np.array([2.3, 3.5, 5.2, 7.1, 9.3])
# l1 = np.array([-2.3, 3.5, 5.2, 7.1, 9.3]) # neg
l2 = np.array([1.7, 3.7, 4.1, 8.4, 12.3])
# l2 = np.array([-1.7, 3.7, 4.1, 8.4, 12.3]) # neg
# l2 = np.array([0.0, 3.7, 4.1, 8.4, 12.3]) # neg

# matrix size
ns = 5

temp = np.random.standard_normal((ns, ns)).astype(np.float64)
temp = temp + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
temp = temp + temp.T.conj() # Make Hermitian
r1 = la.eigh(temp)[1]
assert allclose(la.inv(r1), r1.T.conj()) # assert r1 unitary

temp = np.random.standard_normal((ns, ns)).astype(np.float64)
temp = temp + 1.0J * np.random.standard_normal((ns, ns)).astype(np.float64)
temp = temp + temp.T.conj() # Make Hermitian
r2 = la.eigh(temp)[1]
assert allclose(la.inv(r2), r2.T.conj()) # assert r2 unitary

A = None
B = None
rk = 1 if size >=2 else 0
# rk = 0
if rank == 0:
    A = np.dot(np.dot(r1, np.diag(l1)), r1.T.conj())
    A = np.asfortranarray(A)
    assert allclose(A, A.T.conj())
if rank == rk:
    B = np.dot(np.dot(r2, np.diag(l2)), r2.T.conj())
    B = np.asfortranarray(B)
    assert allclose(B, B.T.conj())

# distribute
core.initmpi([2, 2], block_shape=[3, 3])
dA = core.DistributedMatrix.from_global_array(A, rank=0)
dB = core.DistributedMatrix.from_global_array(B, rank=rk)

# eigen decomposition
# evalsd, evecsd = su.eigh_gen(dA, dB)
evalsd, evecsd = rt.eigh(dA, dB, overwrite_a=False, overwrite_b=False, eigvals=(0, 4))
print evalsd

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
