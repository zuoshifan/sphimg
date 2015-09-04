import numpy as np
import scipy.linalg as la
from mpi4py import MPI
from scalapy import core
import scalapy.routines as rt
import scalapyutil as su


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

if size != 4:
    raise Exception("Test needs 4 processes.")

allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)

l1 = np.array([2.3, 3.5, 5.2, 7.1, 9.3])
l2 = np.array([1.7, 3.7, 4.1, 8.4, 12.3])
# l2 = np.array([-1.7, 3.7, 4.1, 8.4, 12.3])
# l2 = np.array([0.0, 3.7, 4.1, 8.4, 12.3])

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

A = np.dot(np.dot(r1, np.diag(l1)), r1.T.conj())
A = np.asfortranarray(A)
A1 = A
B = np.dot(np.dot(r2, np.diag(l2)), r2.T.conj())
B = np.asfortranarray(B)
B1 = B
# assert Hermitian of A and B
assert allclose(A, A.T.conj())
assert allclose(B, B.T.conj())

# distribute
core.initmpi([2, 2], block_shape=[3, 3])
dA = core.DistributedMatrix.from_global_array(A, rank=0)
dB = core.DistributedMatrix.from_global_array(B, rank=0)

# eigen decomposition
evalsd, evecsd = su.eigh_gen(dA, dB, overwrite_a=False, overwrite_b=False)
# evalsd1, evecsd1 = su.eigh_gen(dA, dB, overwrite_a=True, overwrite_b=True)
# # assert same result for overwrite=True/False
# assert allclose(evalsd, evalsd1)
# evecsd = evecsd.to_global_array(rank=0)
# evecsd1 = evecsd1.to_global_array(rank=0)
# if rank == 0:
#     assert allclose(evecsd, evecsd1)


# distribute assert
LA = rt.dot(rt.dot(evecsd, dA, transA='N', transB='N'), evecsd, transA='N', transB='C')
LA = LA.to_global_array(rank=0)
LB = rt.dot(rt.dot(evecsd, dB, transA='N', transB='N'), evecsd, transA='N', transB='C')
LB = LB.to_global_array(rank=0)
if rank == 0:
    # asser equal diagonal elements
    assert allclose(evalsd, np.diag(LA).real)
    assert allclose(np.ones_like(evalsd), np.diag(LB).real)
    # assert equal matrix
    assert allclose(np.diag(evalsd).astype(LA.dtype), LA)
    assert allclose(np.eye(LB.shape[0], dtype=LB.dtype), LB)


# global assert
gevecsd = evecsd.to_global_array(rank=0)
if rank == 0:
    print evalsd
    print la.det(gevecsd)
    print np.abs(la.det(gevecsd))
    LA1 = np.dot(np.dot(gevecsd, A), gevecsd.T.conj())
    assert allclose(evalsd, np.diag(LA).real)
    assert allclose(LA, LA1)
    LB1 = np.dot(np.dot(gevecsd, B), gevecsd.T.conj())
    assert allclose(np.ones_like(evalsd), np.diag(LB).real)
    assert allclose(LB, LB1)

    # assert Ax = lBx
    for i in range(gevecsd.shape[1]):
        assert allclose(np.dot(A, gevecsd.T.conj()[:, i]), evalsd[i] * np.dot(B, gevecsd.T.conj()[:, i]))
        assert allclose(np.dot(gevecsd.T.conj()[:, i].conj(), np.dot(A, gevecsd.T.conj()[:, i])), np.array([evalsd[i]]))
        assert allclose(np.dot(gevecsd.T.conj()[:, i].conj(), np.dot(B, gevecsd.T.conj()[:, i])), np.array([1.0 + 0.0J]))

# scipy.linalg.eigh
evalsn, evecsn = la.eigh(A1, B1, overwrite_a=False, overwrite_b=False)
if rank == 0:
    print la.det(evecsn.T.conj())
    print np.abs(la.det(evecsn.T.conj()))
    # NOTE: evecsn is not unitary
    assert not allclose(np.eye(evecsn.shape[0], dtype=evecsn.dtype), np.dot(evecsn, evecsn.T.conj()))
    assert not allclose(np.eye(evecsn.shape[0], dtype=evecsn.dtype), np.dot(evecsn.T.conj(), evecsn))
    # assert equal diagonal elements
    assert allclose(evalsn, np.diag(np.dot(np.dot(evecsn.T.conj(), A), evecsn)).real)
    assert allclose(np.ones_like(evalsn), np.diag(np.dot(np.dot(evecsn.T.conj(), B), evecsn)).real)
    # assert equal matrix
    assert allclose(np.diag(evalsn).astype(evecsn.dtype), np.dot(np.dot(evecsn.T.conj(), A), evecsn))
    assert allclose(np.eye(evecsn.shape[0], dtype=evecsn.dtype), np.dot(np.dot(evecsn.T.conj(), B), evecsn))

    # assert equal eigenvalues
    assert allclose(evalsn, evalsd)
    # assert cmp_evecs(evecsn, gevecsd.T.conj())

    # further assert
    assert allclose(np.dot(np.dot(evecsn.T.conj(), A), evecsn), np.dot(np.dot(gevecsd, A), gevecsd.T.conj()))
    assert allclose(np.dot(np.dot(evecsn.T.conj(), B), evecsn), np.dot(np.dot(gevecsd, B), gevecsd.T.conj()))

    # assert Ax = lBx
    prod = 1.0
    for i in range(evecsn.shape[1]):
        assert allclose(np.dot(A, evecsn[:, i]), evalsn[i] * np.dot(B, evecsn[:, i]))
        assert allclose(np.dot(evecsn[:, i].conj(), np.dot(A, evecsn[:, i])), np.array([evalsn[i]]))
        assert allclose(np.dot(evecsn[:, i].conj(), np.dot(B, evecsn[:, i])), np.array([1.0 + 0.0J]))
        # print np.dot(evecsn[i, :].conj(), np.dot(A, evecsn[:, i])), evalsn[i]
        # print evalsd[i], evalsn[i]
        # print gevecsd.T.conj()[:, i]
        # print evecsn[:, i]
        # print 'ratio:'
        print gevecsd.T.conj()[:, i] / evecsn[:, i]
        print np.abs(gevecsd.T.conj()[:, i] / evecsn[:, i])


# if __name__ == '__main__':
#     test_eigh_gen()
