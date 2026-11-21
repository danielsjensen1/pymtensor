from itertools import chain, product


def apply_mapping(indices, mapping):
    """Apply a set of indices to a mapping.
    
    Parameters
    ----------
    indices : iterable
        Iterable containing the keys used to index into `mapping`
    mapping : dictionary
        Dictionary with objects to index.
    
    Returns
    -------
    objects_tuple: tuple of objects
        A tuple containing the objects corresponding to each index in `indices`.
    """
    return tuple(mapping[index] for index in indices)


def expand2full(mapping, iter_indices):
    """
    Expand compressed (Voigt) indices to the full set of equivalent index tuples.

    For each index tuple in *iter_indices*, each element is looked up in
    *mapping* to obtain the set of equivalent full indices for that slot.
    The Cartesian product over all slots is then formed and flattened so that
    each output tuple is a single, concatenated sequence of raw indices.

    Parameters
    ----------
    mapping : dict
        Maps a compressed (Voigt) index to a tuple of the equivalent full
        index tuples.  For example, the standard rank-2 Voigt map has
        ``mapping[3] = ((1, 2), (2, 1))`` (both orderings of the off-diagonal
        pair).
    iter_indices : iterable of tuples
        Each element is a tuple of compressed indices whose Cartesian product
        of full-index equivalents should be generated.

    Returns
    -------
    full : tuple of tuples
        Every fully-expanded index tuple corresponding to the compressed
        indices in *iter_indices*.

    Examples
    --------
    >>> mapping = {0: ((0, 0),), 3: ((1, 2), (2, 1))}
    >>> expand2full(mapping, [(0, 3)])
    ((0, 0, 1, 2), (0, 0, 2, 1))
    """
    full = tuple(tuple(chain.from_iterable(val))
                 for indices in iter_indices
                 for val in product(*apply_mapping(indices, mapping)))
    # def generate_full():
    #     for indices in iter_indices:
    #         # print('indices={}'.format(indices))
    #         expanded = tuple(mapping[index] for index in indices)
    #         # print(tuple(product(*expanded)))
    #         # print('expanded={}'.format(expanded))
    #         for val in product(*expanded):
    #             yield tuple(chain.from_iterable(val))
    # full = tuple(val for val in generate_full())
    return full


def sort_lists_convert2tuples(iter_of_lists):
    """
    Sort a collection of lists and convert each element to a tuple.

    Parameters
    ----------
    iter_of_lists : iterable of lists
        An iterable whose elements are lists (or any sortable sequences).

    Returns
    -------
    result : tuple of tuples
        A tuple containing the sorted sequences, each converted to a tuple.

    Examples
    --------
    >>> sort_lists_convert2tuples([[3, 1], [1, 2], [1, 1]])
    ((1, 1), (1, 2), (3, 1))
    """
    newiter = list(iter_of_lists)
    newiter.sort()
    return tuple(tuple(val) for val in newiter)