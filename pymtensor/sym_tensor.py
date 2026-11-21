from collections import defaultdict
from itertools import chain, combinations_with_replacement, permutations, product
import logging
import time
import psutil

from numpy import (array, empty, empty_like,  nditer, set_printoptions)
# from scipy.linalg import lu, svd
from sympy import Symbol
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys

from pymtensor.rot_tensor import rot_tensor#, to_voigt
from pymtensor.indexing_helpers import expand2full, sort_lists_convert2tuples


class SymbolicTensor(object):
    """
    A symbolic material tensor whose components are SymPy polynomial symbols.

    Constructs a dense tensor of symbolic variables labelled by their
    (possibly Voigt-compressed) indices, then applies crystallographic
    symmetry constraints to determine which components are zero and which
    are related to each other.

    The index string *indices* encodes the rank and symmetry structure of
    the tensor.  See :doc:`../notation` for a full description of the syntax.
    In brief:

    * Each character represents one index group.
    * **Uppercase** letters denote rank-2 (Voigt-compressed) groups;
      **lowercase** letters denote rank-1 groups.
    * An optional integer suffix gives the explicit dimension (e.g. ``'a2'``
      means a rank-2 group labelled ``a``).
    * **Repeated** letters indicate major symmetry — those index groups are
      interchangeable (e.g. ``'AA'`` for a 4th-order fully symmetric tensor).
    * The compact form ``'AbABb'`` is equivalent to the comma-separated form
      ``'a2,b1,a2,B2,b1'``.

    Parameters
    ----------
    indices : str
        Index string encoding the tensor rank and symmetry, e.g.
        ``'AA'`` (4th-order elastic), ``'aB'`` (piezoelectric),
        ``'AAA'`` (3rd-order elastic).
    symbol : str
        Base symbol name.  Each independent component will be named
        ``symbol + index_string``, e.g. ``'c11'``, ``'c12'``, …
    start : int, optional
        Starting value for index labels: ``1`` for 1-based (default),
        ``0`` for 0-based.
    voigt : bool, optional
        If ``True`` (default), label tensor components using compressed
        Voigt indices.  If ``False``, use full (uncompressed) indices.
    reverse : bool, optional
        If ``True`` (default), sort the symbol list in reverse order so
        that the highest-numbered component appears first in the polynomial
        ring.
    create_tensor : bool, optional
        If ``False``, skip building the dense tensor array and polynomial
        ring.  Useful for testing initialisation methods only.

    Attributes
    ----------
    dims : list of int
        Dimension of each index group.
    vm : dict
        Voigt maps keyed by dimension.
    ivm : dict
        Inverse Voigt maps keyed by dimension.
    slices : list of slice
        Index slices into the flat index vector for each group.
    repeats : list of list of int
        Groups of repeated (major-symmetric) index positions.
    tensor : numpy.ndarray of sympy expressions
        Dense symbolic tensor (only present when ``create_tensor=True``).
    names : list of str
        Sorted list of unique component names.
    R : sympy polynomial ring
        The polynomial ring over which the tensor components live.
    ring_vals : tuple
        Ring elements corresponding to each name in *names*.
    tdim : int
        Total tensor rank (sum of all group dimensions).
    """

    def __init__(self, indices, symbol, start=1, voigt=True, reverse=True,
                 create_tensor=True):
        dims, repeats = SymbolicTensor._parse_name(indices)
        # Total uncompressed tensor dimension
        tdim = sum(dims)
        # Initialize all necessary Voigt and inverse-Voigt mappings based on the
        # unique dimensions
        ivm = {}
        vm = {}
        for dim in set(dims):
            ivm[dim], vm[dim] = SymbolicTensor.voigt_map(dim, start)
#         print('vm=', vm)
#         print('ivm=', ivm)
        self.dims = dims
        self.vm = vm
        self.ivm = ivm
        self.slices = SymbolicTensor._create_slices(dims)
        self.repeats = repeats
        # Don't create the tensor if just testing initialization methods
        if create_tensor:
            tensor = empty((3,)*tdim, dtype=object)
            # Create slices for applying the Voigt mappings
            # Find all unique tensor component names
            symbols_dict = {}
            print('start newindex')
            for index in product(range(3), repeat=tdim):
                newindex = self._newindex(index, start, voigt)
                name = ''.join([symbol] + [str(val) for val in newindex])
                print(index, newindex, name)
                symbols_dict[name] = Symbol(name)
            names = list(symbols_dict.keys())
            print('names = {}'.format(names))
            print('Number of unknowns = {}'.format(len(names)))
            sortkey = lambda i: int(i[1:])
            names.sort(reverse=reverse, key=sortkey)
            print('names=', names)
            poly_vals = ring(names, 'QQ<sqrt(3)>')
            poly_dict = dict((name, i) for i, name in enumerate(names))
            R = poly_vals[0]
            ring_vals = poly_vals[1:]
            # Place the ring values in the tensor
            for index in product(range(3), repeat=tdim):
                newindex = self._newindex(index, start, voigt)
                name = ''.join([symbol] + [str(val) for val in newindex])
                tensor[index] = ring_vals[poly_dict[name]]
            self.names = names
            self.R = R
            self.ring_vals = ring_vals
            self.tensor = tensor
        self.tdim = tdim

    @staticmethod
    def _create_slices(dims):
        """
        Build index slices for partitioning the flat index vector by group.

        Parameters
        ----------
        dims : list of int
            Dimension of each index group.

        Returns
        -------
        slices : list of slice
            One slice per group; ``slices[i]`` selects the indices belonging
            to group *i* from a flat index tuple.
        """
        start_slice = 0
        slices = []
        for dim in dims:
            end_slice = start_slice + dim
            slices.append(slice(start_slice, end_slice))
            start_slice = end_slice
        return slices

    @staticmethod
    def _parse_name(name):
        """
        Parse an index string into dimension and repeat lists.

        Accepts two formats:

        * **Compact**: a string of letters, e.g. ``'AbABb'``.  Uppercase
          letters map to rank-2 groups; lowercase to rank-1 groups.
        * **Comma-separated**: explicit ``letter + integer`` tokens,
          e.g. ``'a2,b1,a2,B2,b1'``.

        Parameters
        ----------
        name : str
            Index string in compact or comma-separated format.

        Returns
        -------
        dims : list of int
            Dimension (rank) of each index group.
        repeats : list of list of int
            Each sub-list contains the positions of a repeated (major-symmetric)
            index group.  Groups that appear only once are omitted.
        """
        if ',' in name:
            indices = name.split(',')
            # Check if the last token is empty
            if indices[-1] == '':
                indices.pop()
        else:
            indices = []
            for val in name:
                if val.isupper():
                    indices.append(val + '2')
                else:
                    indices.append(val + '1')
        # TODO: generalize this to allow repeat characters before the
        # integer and do some error checking
        dims = [int(val[1:]) for val in indices]
        repeats = SymbolicTensor._repeated_indices(indices)
        return dims, repeats

    @staticmethod
    def voigt_map(dim, start=0):
        """
        Build Voigt and inverse-Voigt index mappings for a given group dimension.

        The Voigt mapping compresses a symmetric index group of rank *dim*
        into a single integer.  For example, for ``dim=2`` the six unique
        symmetric pairs ``(1,1),(2,2),(3,3),(2,3),(1,3),(1,2)`` map to
        Voigt indices ``0,1,2,3,4,5`` (0-based).

        Currently supports dimensions 1, 2, 3, and 4.

        Parameters
        ----------
        dim : int
            Dimension (rank) of the symmetric index group.  Must be 1–4.
        start : int, optional
            Index offset: ``0`` for 0-based indexing (default),
            ``1`` for 1-based.

        Returns
        -------
        ivm : dict
            Inverse Voigt map: compressed index (int) → full index tuple.
        vm : dict
            Voigt map: full index tuple → compressed index (int).  All
            permutations of a given full index map to the same compressed
            value.
        """
        v = {}
        v[1] = [(1,), (2,), (3,)]
        v[2] = [(1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
        v[3] = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 2, 3), (1, 1, 3),
                (1, 1, 2), (2, 3, 3), (1, 3, 3), (1, 2, 2), (1, 2, 3)]
        v[4] = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 2, 3),
                (1, 1, 1, 3), (1, 1, 1, 2), (2, 2, 3, 3), (1, 1, 3, 3),
                (1, 1, 2, 2), (2, 3, 3, 3), (1, 3, 3, 3), (1, 2, 2, 2),
                (1, 1, 2, 3), (2, 2, 1, 3), (3, 3, 1, 2)]
        vm = {}
        ivm = {}
        ordered_indices = None
        try:
            ordered_indices = v[dim]
        except NotImplementedError:
            print("Voigt map not implemented for dimension {}".format(dim))
        for i, unique_index in enumerate(ordered_indices):
            ishifted = i + start
            # Zero-based indexing: subtract 1
            shifted_index = tuple(val - 1 + start for val in unique_index)
            ivm[ishifted] = shifted_index
            # Every permutation of the index should map to the same Voigt
            # value
            for index in permutations(shifted_index):
                vm[index] = ishifted
        return ivm, vm

    def voigt_map_ascending(self, dim, start=0):
        v = {}

    def _newindex(self, index, start=1, voigt=True):
        """
        Compute the canonical (possibly Voigt-compressed) index for a component.

        Given a raw tensor index tuple, applies the Voigt compression and
        major-symmetry sorting to produce the canonical label used to name
        that component.

        Parameters
        ----------
        index : tuple of int
            Raw (0-based) tensor index, e.g. ``(0, 1, 2, 2)``.
        start : int, optional
            Starting value for index labels (``1`` for 1-based, ``0`` for
            0-based).  Default is ``1``.
        voigt : bool, optional
            If ``True`` (default), return the compressed Voigt index.
            If ``False``, return the full expanded index after sorting.

        Returns
        -------
        newindex : list of int
            Canonical index labels for this tensor component.

        Notes
        -----
        The function is separated from the tensor creation loop to simplify
        unit testing (see ``test_symmetry.py``).
        """
        ivm = self.ivm
        vm = self.vm
        repeats = self.repeats
        slices = self.slices
        dims = self.dims
        # We compress to Voigt notation and then expand after sorting if the
        # user doesn't want Voigt notation
        # Example:
        # index = [0, 1, 0, 2, 1, 1, 1, 0]
        # repeats = [[0, 3]]
        # symmetric=[0, 1, 3]
        index = [val + start for val in index]
        newindex = []
        for dim, slice_ in zip(dims, slices):
            newindex.append(vm[dim][tuple(index[slice_])])
        sortkey = None
        # We can use a custom sort key if using full-tensor notation
        if not voigt:
            sortdict = {1:1, 2:4, 3:6, 4:5, 5:3, 6:2}
            sortkey = lambda i: sortdict[i]
        # Repeated indices need to be sorted
        for irepeat in repeats:
            vals = [newindex[i] for i in irepeat]
            vals.sort(key=sortkey)
            for i, var in zip(irepeat, vals):
                newindex[i] = var
        if not voigt:
            index = newindex
            newindex = []
            for dim, val in zip(dims, index):
                newindex.extend(ivm[dim][val])
        return newindex

    @staticmethod
    def _repeated_indices(name):
        """
        Find groups of repeated (major-symmetric) index tokens.

        Parameters
        ----------
        name : list of str
            List of index tokens such as ``['a2', 'b1', 'a2']``.

        Returns
        -------
        repeats : list of list of int
            Each sub-list contains the positions of tokens that share the
            same letter (case-sensitive).  Tokens appearing only once are
            excluded.

        Examples
        --------
        >>> SymbolicTensor._repeated_indices(['a2', 'b1', 'a2'])
        [[0, 2]]
        """
        # Look for all repeated uppercase letters
        repeats = defaultdict(list)
        for i, val in enumerate(name):
            repeats[val].append(i)
        repeats = [val for val in repeats.values() if len(val) > 1]
        return repeats

    def apply_symmetry(self, symops, timings=True):
        """
        Apply a set of symmetry operations to the tensor and collect constraints.

        For each symmetry operation *R*, forms the equation
        ``rot_tensor(R, T) - T = 0`` and adds the non-trivial polynomial
        equations to a constraint set.  Solves the resulting linear system
        over the polynomial ring to find which tensor components are zero or
        related.

        Parameters
        ----------
        symops : iterable of numpy.ndarray
            Symmetry operation matrices, each of shape ``(3, 3)``.  May be
            generator matrices (see :class:`~pymtensor.symmetry.RedSgSymOps`)
            or the full group orbit
            (see :class:`~pymtensor.symmetry.SgSymOps`).
        timings : bool, optional
            If ``True`` (default), print timing information for each
            symmetry operation and the linear solve.

        Returns
        -------
        sol : dict
            Maps dependent component symbols to their expressions in terms
            of free (independent) components.  An empty dict means the tensor
            is completely unconstrained by the given symmetry.
        """
        tensor = self.tensor; R = self.R; ring_vals = self.ring_vals
        eqs = set()
        if timings: tic = time.perf_counter()
        for symop in symops:
            if timings: symtic = time.perf_counter()
            # Convert the components of the symmetry operation arrays into
            # members of the polynomial ring.
            poly_symop = array([[R(val) for val in row] for row in symop])
            row = (rot_tensor(poly_symop, tensor) - tensor).flatten()
            for eq in row:
                if eq != 0:
                    eqs.add(eq)
            if timings: symtoc = time.perf_counter()
            if timings: print(f"Apply symmetry = {symtoc - symtic:0.4f} seconds")
        if len(eqs) == 0:
            return {}
        print('len(eqs) = {}'.format(len(eqs)))
        print('eqs = ')
        for eq in eqs:
            print(eq.as_expr())
        if timings: toc = time.perf_counter()
        if timings: print(f"Rotate tensor = {toc - tic:0.4f} seconds")

        if timings: tic = time.perf_counter()
        process = psutil.Process()
        print('bytes used = {}'.format(process.memory_info().rss))  # in bytes
        sol = solve_lin_sys(eqs, R, _raw=False)
        process = psutil.Process()
        print('bytes used = {}'.format(process.memory_info().rss))  # in bytes
        if timings: toc = time.perf_counter()
        if timings: print(f"Solve linear system = {toc - tic:0.4f} seconds")
        numfreevars = len(ring_vals) - len(sol)
        print(f"Number of free variables = {numfreevars}")
        logging.info('Number of free variables = {}'.format(numfreevars))
        return sol

    def sol_details(self, symops):
        """
        Apply symmetry and return the full solution dictionary.

        Calls :meth:`apply_symmetry` and then expands the result so that
        every component name (including those that are free variables) maps
        to its value or canonical representative.

        Parameters
        ----------
        symops : iterable of numpy.ndarray
            Symmetry operation matrices (see :meth:`apply_symmetry`).

        Returns
        -------
        fullsol : dict
            Maps every component name string to its SymPy symbol expression.
            Free (independent) components map to themselves; constrained
            components map to the free component they equal (or zero).
        R : sympy polynomial ring
            The polynomial ring used internally.
        """
        R = self.R; names = self.names
        sol = self.apply_symmetry(symops)
        tic = time.perf_counter()
        fullsol = {}
        for name in names:
            symbol = Symbol(name)
            fullsol[name] = sol.get(symbol, symbol)
        toc = time.perf_counter()
        print(f"To full solution = {toc - tic:0.4f} seconds")
        return fullsol, R

    def unique_tensor(self, symops):
        """
        Apply symmetry and return the tensor with equivalent components substituted.

        Calls :meth:`apply_symmetry` and fills every element of the symbolic
        tensor with the canonical representative component (or zero) after
        the symmetry constraints have been applied.

        Parameters
        ----------
        symops : iterable of numpy.ndarray
            Symmetry operation matrices (see :meth:`apply_symmetry`).

        Returns
        -------
        newtensor : numpy.ndarray of sympy expressions
            Tensor of the same shape as :attr:`tensor` with all equivalent
            components replaced by their canonical representative.
        """
        tensor = self.tensor
        sol = self.apply_symmetry(symops)
        tic = time.perf_counter()
        newtensor = empty_like(tensor)
        it = nditer(tensor, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            oldval = it[0].item()
            # Try to retrieve the new value from the solution dictionary.  If
            # the variable doesn't exist then it is unique and we just use the
            # original symbol via the dictionary `get` method
            newval = sol.get(oldval, oldval)
            newtensor[it.multi_index] = newval
            it.iternext()
        toc = time.perf_counter()
        print(f"Format solution = {toc - tic:0.4f} seconds")
        return newtensor


class SparseSymbolicTensor(SymbolicTensor):
    """
    Sparse symbolic tensor exploiting both minor and major symmetries.

    .. note::
        This class is a **work in progress**.  The index-mapping machinery
        (:meth:`_reduced_indices`, :meth:`_major_syms`, etc.) is implemented
        and tested, but the downstream steps that build the sparse polynomial
        system and solve it have not yet been completed.  Do not use this
        class for production computations; use :class:`SymbolicTensor` instead.

    Unlike :class:`SymbolicTensor`, which allocates a dense ``3^N`` symbolic
    array, :class:`SparseSymbolicTensor` is designed to work only with the
    algebraically independent components by exploiting:

    * **Minor symmetries** – Voigt compression of each symmetric index group.
    * **Major symmetries** – permutation equivalence between entire index groups
      (e.g. ``c_{ijkl} = c_{klij}``).

    For a 4th-order elastic tensor this reduces from 81 to 21 independent
    components before any crystal symmetry is applied, giving significant
    memory and time savings for high-rank tensors.

    Parameters
    ----------
    indices : str
        Index string (same syntax as :class:`SymbolicTensor`).
    symbol : str
        Base symbol name.
    start : int, optional
        Starting index value.  Default is ``1``.
    voigt : bool, optional
        Reserved for future use.
    reverse : bool, optional
        Reserved for future use.
    create_tensor : bool, optional
        Reserved for future use.
    """

    def __init__(self, indices, symbol, start=1, voigt=True, reverse=True,
                 create_tensor=True):
        self.start_index = start
        dims, repeats = SymbolicTensor._parse_name(indices)
        # Total uncompressed tensor dimension
        tdim = sum(dims)
#         print(dims, repeats, tdim)
        # Initialize all necessary Voigt and inverse-Voigt mappings based on the
        # unique dimensions
        ivm = {}
        vm = {}
        # First we find the minor symmetries (Voigt mappings)
        for dim in set(dims):
            ivm[dim], vm[dim] = SymbolicTensor.voigt_map(dim, start=0)
        print('vm=', vm)
        print('ivm=', ivm)
        self.vm = vm
        self.ivm = ivm
        # Now we find all of the major symmetries
        red_indices = self._reduced_indices(dims, repeats)
#         for (dim, repeats) in red_indices:
#             print(self._major_syms(len(ivm[dim]), repeats))
        major_syms = [self._major_syms(dim, repeats)
                      for (dim, repeats) in red_indices]
#         major_indices_voigt = [vals[0] for vals in major_syms]
#         degeneracies = [vals[1] for vals in major_syms]
        print('major_syms = ', major_syms)
#         print('degeneracies = ', degeneracies)
#         print('major_indices_voigt = ', major_indices_voigt)
        unique_full_indices = self._unique_full_indices(ivm, major_syms)

    @staticmethod
    def _unique_full_indices(ivm, voigt_indices):
        """
        Convert grouped Voigt-index lists to fully expanded index tuples.

        .. warning::
            This method is incomplete (contains a reference to an undefined
            variable ``foo``) and will raise ``NameError`` at runtime.

        Parameters
        ----------
        ivm : dict
            Inverse Voigt maps keyed by dimension.
        voigt_indices : list
            Grouped Voigt index information as returned by :meth:`_major_syms`.

        Returns
        -------
        bar : tuple
            Partially computed result (implementation not finished).
        """
#         return [ivm[voigt_index] for voigt_index in voigt_indices]
        bar = tuple(chain(*foo) for foo in product(voigt_indices))
        return bar

    @staticmethod
    def _flatten_indices(indices):
        """
        Flatten nested index product groups into a list of full index tuples.

        Takes a list of index groups, where each group is a list of
        full-index tuples for one Voigt slot, and returns the Cartesian
        product of all groups flattened into single tuples.

        Parameters
        ----------
        indices : list of list of tuples
            Each outer element corresponds to one Voigt slot and contains
            the set of full index tuples for that slot.

        Returns
        -------
        flat : list of tuples
            All combinations of full indices, one tuple per combination.

        Examples
        --------
        >>> indices = [([(1, 2), (2, 1)], [(2, 2)]),
        ...            ([(2, 2)], [(1, 2), (2, 1)])]
        >>> SparseSymbolicTensor._flatten_indices(indices)
        [(1, 2, 2, 2), (2, 1, 2, 2), (2, 2, 1, 2), (2, 2, 2, 1)]
        """
        return list(tuple(chain.from_iterable(val))
                    for product_indices in indices
                    for val in product(*product_indices))

    @staticmethod
    def _reduced_indices(dims, repeats):
        """
        Collapse repeated (major-symmetric) index groups into a reduced list.

        Given the dimension and repeat information for a tensor, returns a
        condensed list where each unique index group appears once, annotated
        with its degeneracy (the number of times it is repeated).

        Parameters
        ----------
        dims : list of int
            Dimension of each index group (length = number of groups).
        repeats : list of list of int
            Each sub-list gives the positions of a repeated group.

        Returns
        -------
        red_indices : list of tuple
            Each element is ``(dim, num_repeats)`` — the dimension of the
            representative group and how many times it appears.

        Examples
        --------
        For ``'AA'`` (4th-order elastic tensor with full major symmetry):

        >>> SparseSymbolicTensor._reduced_indices([2, 2], [[0, 1]])
        [(2, 2)]

        For ``'AAa'`` (two symmetric rank-2 groups and one rank-1 group):

        >>> SparseSymbolicTensor._reduced_indices([2, 2, 1], [[0, 1]])
        [(2, 2), (1, 1)]
        """
#         unique_dims = set(dims)
#         print(dims, unique_dims, repeats)
        # (dim, number of repeats, indices to remove)
        red_indices = [(dim, 1) for dim in dims]
        rm_indices = [index for repeat in repeats for index in repeat[1:]]
        rm_indices.sort()
        for repeat in repeats:
            num_repeats = len(repeat)
            first_repeat = repeat[0]
            dim = dims[first_repeat]
            red_indices[first_repeat] = (dim, num_repeats)
        # Go through the list backwards to not mess up the list ordering
        for rm_ind in rm_indices[::-1]:
            red_indices.pop(rm_ind)
        return red_indices

    @staticmethod
    def _expand2full(mapping, iter_indices):
        """
        Expand compressed Voigt indices to all equivalent full index tuples.

        A local copy of :func:`~pymtensor.indexing_helpers.expand2full` that
        returns a list instead of a tuple (retained for backwards
        compatibility within the class).

        Parameters
        ----------
        mapping : dict
            Voigt → full index mapping (see :meth:`~SymbolicTensor.voigt_map`).
        iter_indices : iterable of tuples
            Each element is a tuple of compressed Voigt indices.

        Returns
        -------
        full : list of tuples
            All fully-expanded index tuples.
        """
        full = []
        for indices in iter_indices:
            # print('indices={}'.format(indices))
            expanded = tuple(mapping[index] for index in indices)
            # print(tuple(product(*expanded)))
            # print('expanded={}'.format(expanded))
            term = list(tuple(chain.from_iterable(val))
                        for val in product(*expanded))
            full += term
        return full

#     @staticmethod
    def _major_syms(self, dim_voigt, num_repeats):
        """
        Enumerate all unique Voigt index combinations and their symmetry equivalents.

        For a given Voigt space of size *dim_voigt* (the number of compressed
        indices for a rank-*dim_voigt* group) and a repetition count
        *num_repeats*, finds every canonical ``(representative, equivalents)``
        pair where the representative is the lexicographically smallest
        combination and equivalents is the set of all its permutations.

        Parameters
        ----------
        dim_voigt : int
            The index-group dimension (e.g. ``2`` for a rank-2 Voigt space
            with 6 entries, ``1`` for a rank-1 space with 3 entries).
        num_repeats : int
            How many times this group is repeated (its degeneracy).

        Returns
        -------
        indices_map : list of tuple
            Each element is ``(representative, equiv_set)`` where
            *representative* is the canonical Voigt index tuple
            (``combinations_with_replacement`` order) and *equiv_set* is the
            set of all permutations of that tuple.

        Notes
        -----
        The method currently has ``# @staticmethod`` commented out because
        it accesses ``self.ivm``.  It is an instance method.
        """
        # Representative and remainder indices lists
        ivm = self.ivm
        num_voigt = len(ivm[dim_voigt])
        # We create sets of the permutations in order to pick out the unique
        # values.  We then convert the sets to lists so that they can be sorted.
        voigtmap = {key: sort_lists_convert2tuples(set(permutations(val)))
                    for key, val in ivm[dim_voigt].items()}
        print('voigtmap=', voigtmap)
        voigt_indices = range(num_voigt)
        print('voigt_indices=', voigt_indices)
        unique_indices = tuple(combinations_with_replacement(voigt_indices, num_repeats))
        print('unique_indices=', unique_indices)
        major_indices = tuple(list(set(tuple(permutations(indices)))) for indices in unique_indices)
        print('major_indices=', major_indices)
        print(len(unique_indices), len(major_indices))
        full_indices = tuple(expand2full(voigtmap, indices)
                             for indices in major_indices)
        print('full_indices=', full_indices)
        # for major_index in major_indices:
        #     print(tuple(tuple(chain.from_iterable(map(voigtmap.get, tuple(val)))) for val in major_index))

        # full_indices = [tuple(chain.from_iterable(val)) for val in product(*major_indices)]
        print('full_indices=', full_indices[0])
        indices_map = []
        full_indices_map = []
        for indices in combinations_with_replacement(voigt_indices, num_repeats):
            equiv_indices = set(permutations(indices))
            indices_map.append((indices, equiv_indices))
            print([val for equiv_index in equiv_indices
                   for val in equiv_index])
        print('indices_map=', indices_map)
        return indices_map


def symbolic_piezo(dim, sym, canonical=True, sort=True):
    voigt_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2,
                 (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4,
                 (0, 1): 5, (1, 0): 5}
    tensor = empty((3,)*dim, dtype=object)
    indices = product(range(3), repeat=dim)
    if canonical:
        ls, nc, rs = [''] * 3
        shift = 1
    else:
        ls, nc, rs = ['[', ',', ']']
        shift = 0
    if dim == 2:
        for index in indices:
            i, j = index
            voigt_order = (i, j)
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 3:
        for index in indices:
            # The following statement handles the third-order dielectric constant
            if sort:
                newindex = tuple(sorted(index))
                i = newindex[0]
                J = voigt_map[newindex[1:3]]
            else:
                i = index[0]
                J = voigt_map[index[1:3]]
            voigt_order = (i, J)
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 4:
        for index in indices:
            I = voigt_map[index[0:2]]
            J = voigt_map[index[2:4]]
            if sort:
                voigt_order = sorted([I, J])
            else:
                voigt_order = [I, J]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 5:
        for index in indices:
            i = index[0]
            J, K = [voigt_map[val] for val in [index[1:3], index[3:5]]]
            voigt_order = (i, J, K)
            if sort:
                voigt_order = [i] + sorted([J, K])
            else:
                voigt_order = [i, J, K]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 6:
        for index in indices:
            I, J, K = [voigt_map[val] for val
                       in [index[2*i:2*i+2] for i in range(3)]]
            if sort:
                voigt_order = sorted([I, J, K])
            else:
                voigt_order = [I, J, K]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    raise NotImplementedError("Dimension {} hasn't been programmed yet.".format(dim))
    return tensor


if __name__ == '__main__':
    set_printoptions(linewidth=200)
    sg = SgSymOps()
#     sg.groups['hexagonal']['3']
#     sg.groups['hexagonal']['3d']
    symops = sg('3m')
#     symops = sg('1')
#     symops = sg('32')
#     symops = sg('6')
#     symops = sg('m3d')
#     st = SymbolicTensor("aa", 'd')
    st = SymbolicTensor("abcdef", 'c')
#     st = SymbolicTensor('aB', 'e')
#     st = SymbolicTensor('AAA', 'c')
#     st = SymbolicTensor('ab', 'e')
#     C3p = sg.hexops["SIGd1"]
#     print('st=', to_voigt(st.tensor))
#     ans = to_voigt(rot_tensor(C3p, st.tensor))
#     for i in range(3):
#         for j in range(3):
#             print(i+1, j+1, ans[i, j].expand())
#     print(ans[0, 0].expand())
#     reduced_tensor = st.unique_tensor(symops)
    st.sol_details(symops)
