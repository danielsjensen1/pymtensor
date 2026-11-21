from numpy import array, cos, dot, empty, nditer, sin, tensordot
from numpy.linalg import norm
from itertools import chain, permutations, product


# TODO: add more error checking for input arrays
def to_tensor(voigt, order='Voigt'):
    """
    Convert a compressed (Voigt) array to full tensor notation.

    Expands a Voigt-compressed array into a fully symmetric tensor by
    filling all permutations of each compressed index pair.

    Supported shapes and their output shapes:

    * ``(3, 3)`` → ``(3, 3)`` (rank-2, returned as-is)
    * ``(3, 6)`` → ``(3, 3, 3)`` (rank-3, e.g. piezoelectric tensor)
    * ``(6, 6)`` → ``(3, 3, 3, 3)`` (rank-4, e.g. elastic stiffness)
    * ``(3, 6, 6)`` → ``(3, 3, 3, 3, 3)`` (rank-5)
    * ``(6, 6, 6)`` → ``(3, 3, 3, 3, 3, 3)`` (rank-6)

    Voigt index convention (1-based):
    ``1→11, 2→22, 3→33, 4→23&32, 5→13&31, 6→12&21``

    VASP index convention (1-based):
    ``1→11, 2→22, 3→33, 4→12&21, 5→23&32, 6→13&31``

    Parameters
    ----------
    voigt : numpy.ndarray
        Compressed tensor array.  Shape must be one of those listed above.
    order : {'Voigt', 'VASP'}, optional
        Index convention for the off-diagonal pairs.  Default is ``'Voigt'``.

    Returns
    -------
    res : numpy.ndarray
        Full tensor with all symmetry-equivalent components filled.

    Notes
    -----
    The function uses ``numpy.nditer`` for iteration and relies on the
    symmetry of the output tensor to fill equivalent components.
    """
    index_map = {0: (0, 0), 1: (1, 1), 2: (2, 2)}
    if order.lower() == 'voigt':
        index_map.update({3: (1, 2), 4: (0, 2), 5: (0, 1)})
    else:
        index_map.update({3: (0, 1), 4: (1, 2), 5: (2, 0)})
    it = nditer(voigt, flags=['multi_index'])
    # TODO: Replace `if` statements using SymPy's `factorint` function.
    if voigt.size == 3*3:
        res = voigt.copy()
    if voigt.size == 3*6:
        res = empty((3,)*3, dtype=voigt.dtype)
        while not it.finished:
            i, jk = it.multi_index
            j, k = index_map[jk]
    #         print(it[0], it.multi_index)
            res[i, j, k] = it[0]
            res[i, k, j] = it[0]
            it.iternext()
    elif voigt.size == 6*6:
        res = empty((3,)*4, dtype=voigt.dtype)
        while not it.finished:
            ij, kl = it.multi_index
            (i, j), (k, l) = [index_map[(ind)] for ind in [ij, kl]]
    #         print(it[0], it.multi_index)
            res[i, j, k, l] = it[0]
            res[i, j, l, k] = it[0]
            res[j, i, k, l] = it[0]
            res[j, i, l, k] = it[0]
            it.iternext()
    elif voigt.size == 3*6*6:
        res = empty((3,)*5, dtype=voigt.dtype)
        while not it.finished:
            i, jk, lm = it.multi_index
            (j, k), (l, m) = [index_map[(ind)] for ind in [jk, lm]]
    #         print(it[0], it.multi_index)
            res[i, j, k, l, m] = it[0]
            res[i, j, k, m, l] = it[0]
            res[i, k, j, l, m] = it[0]
            res[i, k, j, m, l] = it[0]
            it.iternext()
    elif voigt.size == 6*6*6:
        res = empty((3,)*6, dtype=voigt.dtype)
        while not it.finished:
            tensorindices = [index_map[ind] for ind in it.multi_index]
            # Permute each pair of indices based on symmetry arguments and
            # find all possible combinations of indices
            # 3! * 2^3 = 48 possibilities per Voigt index
            for index in product(*(permutations(index) for index in tensorindices)):
                flatindex = tuple(chain.from_iterable(index))
#                 print(flatindex)
                res[flatindex] = it[0]
            it.iternext()
    return res


def to_voigt(tensor, order='Voigt'):
    """
    Convert a full tensor to compressed (Voigt) notation.

    Extracts one representative element per symmetry-equivalent group of
    indices to produce a Voigt-compressed array.

    Supported input shapes and their output shapes:

    * ``(3, 3)`` → ``(3, 3)`` (rank-2, returned as-is)
    * ``(3, 3, 3)`` → ``(3, 6)`` (rank-3, e.g. piezoelectric tensor)
    * ``(3, 3, 3, 3)`` → ``(6, 6)`` (rank-4, e.g. elastic stiffness)
    * ``(3, 3, 3, 3, 3)`` → ``(3, 6, 6)`` (rank-5)
    * ``(3, 3, 3, 3, 3, 3)`` → ``(6, 6, 6)`` (rank-6)

    Parameters
    ----------
    tensor : numpy.ndarray
        Full tensor array whose shape is one of those listed above.
    order : {'Voigt', 'VASP'}, optional
        Index convention for the off-diagonal pairs.  Default is ``'Voigt'``.

    Returns
    -------
    res : numpy.ndarray
        Compressed tensor array.
    """
    index_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2}
#     (0, 1): 3, (1, 0): 3, (1, 2): 4, (2, 1): 4, (2, 0): 5, (0, 2): 5}
    if order.lower() == 'voigt':
        index_map.update({(1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4,
                          (0, 1): 5, (1, 0): 5})
    else:
        index_map.update({(0, 1): 3, (1, 0): 3, (1, 2): 4, (2, 1): 4,
                          (2, 0): 5, (0, 2): 5})
    dims = len(tensor.shape)
    it = nditer(tensor, flags=['multi_index', 'refs_ok'])
    if dims == 2:
        res = tensor.copy()
    if dims == 3:
        res = empty((3, 6), dtype=tensor.dtype)
        while not it.finished:
            i, j, k = it.multi_index
            jk = index_map[(j, k)]
    #         print(it[0], it.multi_index)
            res[i, jk] = it[0].item()
            it.iternext()
    elif dims == 4:
        res = empty((6, 6), dtype=tensor.dtype)
        while not it.finished:
            i, j, k, l = it.multi_index
            ij, kl = [index_map[(ind1, ind2)] for ind1, ind2
                      in [(i, j), (k, l)]]
            res[ij, kl] = it[0].item()
            it.iternext()
    elif dims == 5:
        res = empty((3, 6, 6), dtype=tensor.dtype)
        while not it.finished:
            i, j, k, l, m = it.multi_index
            jk, lm = [index_map[(ind1, ind2)] for ind1, ind2
                      in [(j, k), (l, m)]]
            res[i, jk, lm] = it[0].item()
            it.iternext()
    elif dims == 6:
        res = empty((6, 6, 6), dtype=tensor.dtype)
        while not it.finished:
            I, J, K = [index_map[indices] for indices
                       in [it.multi_index[2*i:2*i+2] for i in range(3)]]
            res[I, J, K] = it[0].item()
            it.iternext()
    return res


def rotu(theta, u):
    """
    Rotation matrix for a rotation by *theta* about an arbitrary axis *u*.

    Uses the Rodrigues rotation formula.  The axis vector is normalised
    internally so it does not need to be a unit vector on input.

    Parameters
    ----------
    theta : float or sympy expression
        Rotation angle in radians.
    u : array-like of length 3
        Rotation axis vector (need not be normalised).

    Returns
    -------
    R : numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    # Normalize u
    ux, uy, uz = u / norm(u)
    c = cos(theta)
    omc = 1 - c
    s = sin(theta)
    return array([[c + ux**2 * omc,        ux * uy * omc - uz * s, ux * uz * omc + uy * s],
                  [uy * ux * omc + uz * s, c + uy**2 * omc,        uy * uz * omc - ux * s],
                  [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz**2 * omc       ]])


def rotx(theta):
    """
    Rotation matrix for a rotation by *theta* about the x-axis.

    Parameters
    ----------
    theta : float or sympy expression
        Rotation angle in radians.

    Returns
    -------
    R : numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    c = cos(theta)
    s = sin(theta)
    return array([[1, 0,  0],
                  [0, c, -s],
                  [0, s,  c]])


def roty(theta):
    """
    Rotation matrix for a rotation by *theta* about the y-axis.

    Parameters
    ----------
    theta : float or sympy expression
        Rotation angle in radians.

    Returns
    -------
    R : numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    c = cos(theta)
    s = sin(theta)
    return array([[ c, 0,  s],
                  [ 0, 1,  0],
                  [-s, 0,  c]])


def rotz(theta):
    """
    Rotation matrix for a rotation by *theta* about the z-axis.

    Parameters
    ----------
    theta : float or sympy expression
        Rotation angle in radians.

    Returns
    -------
    R : numpy.ndarray, shape (3, 3)
        Rotation matrix.
    """
    c = cos(theta)
    s = sin(theta)
    return array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])


def euler_rot(alpha, beta, gamma):
    """
    Rotation matrix from proper (ZXZ) Euler angles.

    Computes ``R = Rz(gamma) @ Rx(beta) @ Rz(alpha)``.

    Parameters
    ----------
    alpha : float
        First rotation angle (about z-axis), in radians.
    beta : float
        Second rotation angle (about x-axis), in radians.
    gamma : float
        Third rotation angle (about z-axis), in radians.

    Returns
    -------
    R : numpy.ndarray, shape (3, 3)
        Combined rotation matrix.
    """
    rot = rotz(alpha)
    rot = dot(rotx(beta), rot)
    rot = dot(rotz(gamma), rot)
    return rot


# Based on this StackOverflow question:
# https://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy/42347571#42347571
def rot_tensor(rotation, tensor):
    """
    Apply a rotation matrix to a tensor of arbitrary rank.

    Contracts the rotation matrix against every index of *tensor* via a
    sequence of ``tensordot`` operations, computing:

    .. math::

        T'_{i_1 i_2 \\ldots i_n} =
            R_{i_1 j_1} R_{i_2 j_2} \\cdots R_{i_n j_n}\\, T_{j_1 j_2 \\ldots j_n}

    Parameters
    ----------
    rotation : numpy.ndarray, shape (3, 3)
        Rotation (or any linear) matrix.  May contain SymPy expressions
        when working in symbolic mode.
    tensor : numpy.ndarray, shape (3,) * n
        Tensor of rank *n*.  May contain SymPy expressions.

    Returns
    -------
    result : numpy.ndarray, shape (3,) * n
        Rotated tensor.

    Examples
    --------
    Rotate a rank-3 tensor ``d_ijk``:

    .. math::

        d'_{ijk} = A_{il} A_{jm} A_{kn}\\, d_{lmn}
    """
    dims = len(tensor.shape)
    res = tensor #  We don't need a deep copy because tensordot returns a new array
    r = rotation
#     r = rotation.T
    for i in range(dims):
        res = tensordot(r, res, axes=((-1),(-1)))
#         res = tensordot(res, r, axes=((0),(0)))
    return res
