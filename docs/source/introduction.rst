Introduction
============

What is PyMTensor?
------------------

PyMTensor is a Python library for computing the symmetry-reduced form of
material property tensors under any of the 32 crystallographic point groups.
Given a tensor type (specified by its index symmetry structure) and a crystal
symmetry group, PyMTensor determines:

* which tensor components are **identically zero**,
* which non-zero components are **equal to each other** (or related by a
  sign change), and
* how many **independent** components remain.

This is the fundamental first step in characterising any material property
that can be represented as a tensor — elasticity, piezoelectricity,
electro-optics, nonlinear optics, and higher-order generalisations of all of
these.

Physical background
-------------------

Material tensors describe the linear (or nonlinear) response of a crystal to
applied fields.  For example:

* The **elastic stiffness tensor** :math:`c_{ijkl}` relates stress to strain.
* The **piezoelectric tensor** :math:`e_{ijk}` relates electric polarization
  to strain (or stress to electric field).
* The **dielectric susceptibility tensor** :math:`\chi_{ij}` relates
  polarization to the applied electric field.

Crystal symmetry constrains the form of these tensors.  If a crystal is
invariant under a symmetry operation :math:`R`, then the tensor :math:`T`
must satisfy

.. math::

   T_{i_1 i_2 \ldots i_n} =
       R_{i_1 j_1} R_{i_2 j_2} \cdots R_{i_n j_n} \, T_{j_1 j_2 \ldots j_n}

for every operation :math:`R` in the point group.  PyMTensor forms and solves
this linear system symbolically, directly yielding the independent components.

Index symmetry
~~~~~~~~~~~~~~

In addition to crystal symmetry, many tensors possess **intrinsic index
symmetry** — certain indices are interchangeable by thermodynamic or
electromagnetic arguments.  For example:

* The elastic stiffness tensor satisfies :math:`c_{ijkl} = c_{jikl} = c_{ijlk} = c_{klij}`.
* The piezoelectric tensor satisfies :math:`e_{ijk} = e_{ikj}` (symmetry in the
  last two indices).

PyMTensor encodes these intrinsic symmetries directly in the index string
passed to :class:`~pymtensor.sym_tensor.SymbolicTensor`, so they are
automatically accounted for before any crystal symmetry is applied.

Supported symmetry groups
-------------------------

All **32 crystallographic point groups** are supported, covering:

* Triclinic: :math:`1`, :math:`\bar{1}`
* Monoclinic: :math:`2`, :math:`m`, :math:`2/m`
* Orthorhombic: :math:`222`, :math:`mm2`, :math:`mmm`
* Tetragonal: :math:`4`, :math:`\bar{4}`, :math:`4/m`, :math:`422`,
  :math:`4mm`, :math:`\bar{4}2m`, :math:`4/mmm`
* Trigonal: :math:`3`, :math:`\bar{3}`, :math:`32`, :math:`3m`,
  :math:`\bar{3}m`
* Hexagonal: :math:`6`, :math:`\bar{6}`, :math:`6/m`, :math:`622`,
  :math:`6mm`, :math:`\bar{6}m2`, :math:`6/mmm`
* Cubic: :math:`23`, :math:`m\bar{3}`, :math:`432`, :math:`\bar{4}3m`,
  :math:`m\bar{3}m`

Two symmetry-operation classes are provided:

* :class:`~pymtensor.symmetry.RedSgSymOps` — stores only the **generators**
  of each group (recommended; faster for the solver).
* :class:`~pymtensor.symmetry.SgSymOps` — stores the **full orbit** of
  symmetry operations, with both default and IEEE axis conventions.

Dependencies
------------

* **Python** ≥ 3.6
* **NumPy** ≥ 1.6
* **SymPy** ≥ 1.7
* **psutil**
* **gmpy2** *(optional but highly recommended)* — significantly speeds up
  SymPy's polynomial solver for large tensors.

Next steps
----------

* :doc:`tutorial` — step-by-step worked examples.
* :doc:`notation` — full description of the index string syntax.
* :doc:`api/index` — complete API reference.
