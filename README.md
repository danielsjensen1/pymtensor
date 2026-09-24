
# PyMTensor

Material tensor Python package for computing the unique tensor elements for material tensors of arbitrary order and arbitrary crystallographic point group.

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.


# Installation

Install from the repository root using pip:

```bash
pip install .
```

To include the optional `gmpy2` dependency (recommended for speed improvements to SymPy's polynomial solver):

```bash
pip install ".[speed]"
```

For development, install in editable mode:

```bash
pip install -e .
```

PyMTensor requires Python 3.6 or higher, NumPy 1.6 or greater, SymPy 1.7 or greater, and psutil.
The optional dependency gmpy2 for SymPy is highly recommended for speed improvements.

# Quick Start
The following code snippet will compute the unique tensor elements of a 5th-rank tensor with indices 1 and 3 interchangeable.

```python
from pymtensor.symmetry import RedSgSymOps
from pymtensor.sym_tensor import SymbolicTensor

# Choose a symmetry group (e.g. '3m')
sg = RedSgSymOps()
symops = sg('3m')

# Create a 5th-rank symbolic tensor with indices 1 and 3 interchangeable
st = SymbolicTensor("abacd", 'c')

# Solve for the unique tensor elements 
fullsol, polyring = st.sol_details(symops)
print(fullsol)
```

For more examples please refer to the `pymtensor/examples/simple_example.py` file.
