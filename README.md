# **PureSet**

![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-1.0.250702.0-brightgreen.svg)

**PureSet** is a powerful **immutable**, **ordered**, and **hashable** collection type for Python that ensures **type homogeneity**. It seamlessly combines the benefits of sequences and sets, making it ideal for advanced Python programming in production environments.

## **Why Use PureSet?**

- **Immutability:** Guarantees data integrity by maintaining unchangeable collections.
- **Order Preservation:** Retains the order of elements, making it an excellent substitute for `Enums` or priority lists.
- **Hashable Elements:** Allows storage in mappings like dictionaries or sets.
- **Type Consistency:** Ensures all elements in a `PureSet` are of the same type (and same shape for structs).
- **Efficient Membership Access:** Supports fast membership testing and retrieval.

---

## **Real-World Use Cases**

### 1. **Replacing Enums for Predefined States**

```python
from pureset import PureSet

# Define a set of possible states
OrderStatus = PureSet("Pending", "Processing", "Shipped", "Delivered")

# Membership Testing
status = "Shipped"
if status in OrderStatus:
    print(f"'{status}' is a valid order status.")

# Enforcing Validity
def update_order_status(new_status: str):
    if new_status not in OrderStatus:
        raise ValueError(f"'{new_status}' is an invalid order status.")
    print(f"Order status updated to: {new_status}")

update_order_status("Processing")
```

---

### 2. **Lookup Table for Roles or Access Levels**

```python
# Define Roles
ACCESS_ROLES = PureSet("Admin", "Editor", "Contributor", "Viewer")

# Enforcing Access Control
def check_access(user_role: str):
    if user_role not in ACCESS_ROLES:
        raise PermissionError(f"Unrecognized role: {user_role}")
    print(f"Access granted for role: {user_role}")

check_access("Admin")
```

---

### 3. **Maintaining Unique, Immutable Task Priorities**

```python
# Priority Levels
Priorities = PureSet("High", "Medium", "Low")

# Reversing Priority Levels
print(Priorities.reverse())  # Outputs: PureSet('Low', 'Medium', 'High')
```

---

### 4. **Predictive Membership Matching in Data Pipelines**

```python
def filter_valid_data(data_stream):
    valid_keywords = PureSet("keyword1", "keyword2", "keyword3")
    return [data for data in data_stream if data in valid_keywords]

# Incoming Data Stream
data = ["keyword1", "unknown", "keyword2", "anything"]

# Filter Valid Data
filtered = filter_valid_data(data)
print(filtered)  # Outputs: ['keyword1', 'keyword2']
```

---

## **Key API Highlights**

### **Creation**

```python
ps = PureSet(1, 2, 3, 2)  # Automatically removes duplicates
empty_ps = PureSet()      # Creates an empty PureSet
```

### **Set Operations**

```python
ps1 = PureSet(1, 2, 3)
ps2 = PureSet(2, 3, 4)

print(ps1 | ps2)  # Union: PureSet(1, 2, 3, 4)
print(ps1 & ps2)  # Intersection: PureSet(2, 3)
print(ps1 - ps2)  # Difference: PureSet(1)
print(ps1 ^ ps2)  # Symmetric Difference: PureSet(1, 4)
```

### **Type Safety**

```python
# Raises TypeError if element types differ
PureSet("string", 123)
```

### **Membership Testing**

```python
ps = PureSet("a", "b", "c")

print("a" in ps)  # True
print(ps.get("d", "Not Found"))  # Default Value
```

### **Sorting**

```python
ps = PureSet(3, 1, 4, 2)
sorted_ps = ps.sorted()  # PureSet(1, 2, 3, 4)
```

---

## **Installation**

Install via PyPI:

```bash
pip install pureset
```

---

## **Running Tests**

Run unit tests to verify functionality:

```bash
pytest tests/
```

---

## **Contributions**

We welcome contributions! Please refer to the [GitHub repository](https://github.com/gabrielmsilva00/PureSet) for guidelines. Issues and pull requests are encouraged.

---

## **License**

This project is licensed under the [Apache License 2.0](LICENSE).