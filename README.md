# PureSet

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)

PureSet is an **immutable, ordered, and homogeneous collection** for Python 3.9+. It combines the best aspects of sequences and sets, ensuring **uniqueness, type consistency, and fast membership testing**.

## Features

- Immutable, ordered collections.
- Ensures all elements are of the same type.
- Supports basic set operations (union, intersection, difference).
- Optimized for performance with hashability checks.

## Installation

Install using PyPI:

```bash
pip install pureset
```

## Usage Examples

```python
from pureset import PureSet

# Create a PureSet instance
ps = PureSet(1, 2, 3, 2)

# Immutability
try:
    ps[0] = 10
except AttributeError:
    print("PureSet is immutable!")

# Set operations
ps1 = PureSet(1, 2, 3)
ps2 = PureSet(2, 3, 4)
print(ps1 | ps2)  # Union: PureSet(1, 2, 3, 4)
print(ps1 & ps2)  # Intersection: PureSet(2, 3)

# Type consistency
try:
    ps = PureSet(1, "string", 3.0)
except TypeError as e:
    print(e)
```

## Tests

Run the following to execute unit tests:

```bash
pytest tests/
```

## Contribution Guidelines

Contributions are welcome! Please submit pull requests or raise issues on the [GitHub repository](https://github.com/gabrielmsilva00/PureSet).

## License

This library is licensed under the [Apache License 2.0](LICENSE).