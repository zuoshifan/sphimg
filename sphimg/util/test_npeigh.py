import numpy as np
import scipy.linalg as la


allclose = lambda a, b: np.allclose(a, b, rtol=1e-4, atol=1e-6)

l1 = np.array([2.3, 3.5, 5.2, 7.1, 9.3])
l2 = np.array([1.7, 3.7, 4.1, 8.4, 12.3])

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
B = np.dot(np.dot(r2, np.diag(l2)), r2.T.conj())
# assert Hermitian of A and B
assert allclose(A, A.T.conj())
assert allclose(B, B.T.conj())


# scipy.linalg.eigh
evals, evecs = la.eigh(A, B, overwrite_a=False, overwrite_b=False)
evecs = evecs.T.conj() # need Hermitian transpose
print evals
assert not allclose(la.inv(evecs), evecs.T.conj()) # evecs is not unitary
print np.dot(evecs, evecs.T.conj()) # non-diagonal


threshold = 1.0
i_ev = np.searchsorted(evals, threshold)
evals = evals[i_ev:]
evecs = evecs[i_ev:]
print "Modes with S/N > %f: %i of %i" % (threshold, evals.size, ns)
print np.dot(evecs, evecs.T.conj()) # non-diagonal

print np.dot(evecs, np.dot(A, evecs.T.conj())) # diagonal
print np.dot(evecs, np.dot(B, evecs.T.conj())) # identity


import sys
print 'Use flush...'
sys.stdout.flush()
print 'Done.'