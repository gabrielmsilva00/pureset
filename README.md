# **PureSet**

![Python Version](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)
![Version](https://img.shields.io/badge/Version-1.0.250702.0-brightgreen.svg)

**PureSet** is a robust **immutable**, **ordered**, and **hashable** collection type for Python designed specifically for high-performance, professional-level applications. It enforces **type homogeneity** and uniquely combines the functionalities of sets and sequences, making it a cutting-edge solution for advanced use cases.

---

## **Core Features**
- **Ordered and Immutable:** Preserves the sequence of elements while ensuring data integrity through immutability.
- **Fast Membership Testing:** Optimized for quick lookups and membership checks.
- **Hashable and Type-Enforced:** Suitable for use as dictionary keys, ensuring all contained elements share the same data type and shape.
- **High-Performance Set-Like Operations:** Supports union, intersection, and other essential set operations.
---

## **Use Cases**
```python
from pureset import PureSet
```

### **1. Enum Replacement for Constants**
Define ordered constants for application states or configurations.
```python
WORKFLOW_STATES = PureSet("Draft", "Pending", "Approved", "Rejected")

current_state = "Approved"

if current_state in WORKFLOW_STATES:
    print(f"The state '{current_state}' is valid.")
else:
    raise ValueError(f"'{current_state}' is an unrecognized state.")
```
---

### **2. Controlled Lookups for Role-Based Access Control**
Simplify access management for deployment environments or user roles.
```python
AUTHORIZED_ROLES = PureSet("Admin", "Developer", "QA", "Viewer")

def authorize_user(role):
    if role not in AUTHORIZED_ROLES:
        raise PermissionError(f"Role '{role}' is not permitted.")
    print(f"Access granted for role: {role}")

authorize_user("QA")  # Valid example
```
---

### **3. Creating Hashable Lookup Tables**
Function as immutable keys in dynamic caching or dependency injection systems.
```python
cache_store = {}

pipeline_configuration = PureSet("Stage1", "Stage2", "Stage3")
cache_store[pipeline_configuration] = "Pipeline executed successfully."

print(cache_store[pipeline_configuration])  # Retrieve data efficiently
```
---

### **4. Validation in Data Pipelines**
Filter streaming data for valid entries using type consistency.
```python
VALID_KEYWORDS = PureSet("keyword1", "keyword2", "keyword3")

def filter_stream(input_stream):
    return [data for data in input_stream if data in VALID_KEYWORDS]

results = filter_stream(["keyword1", "invalid", "keyword2"])
print(results)  # Output: ['keyword1', 'keyword2']
```
---

### **5. Ordered Priority Queues**
Maintain immutable priority levels for task or job execution.
```python
PRIORITIES = PureSet("Critical", "High", "Medium", "Low")

print(PRIORITIES.reverse())  # Output: PureSet('Low', 'Medium', 'High', 'Critical')
```
---

### **6. Immutable Feature Toggles**
Define immutable and organized feature flags to control application behavior.

```python
FEATURE_FLAGS = PureSet("ENABLE_LOGIN", "ENABLE_SIGNUP", "ENABLE_BILLING")

if "ENABLE_SIGNUP" in FEATURE_FLAGS:
    print("Signup feature is enabled.")
```
---

## **API Overview**
Here is a brief overview of the key capabilities PureSet offers:

### **Immutable and Ordered Construction**
```python
ps = PureSet(3, 1, 2, 2)
print(ps)  # Outputs: PureSet(3, 1, 2)
```

### **Set-Like Operations**
```python
ps1 = PureSet(1, 2, 3)
ps2 = PureSet(3, 4, 5)

print(ps1 | ps2)  # Union: PureSet(1, 2, 3, 4, 5)
print(ps1 & ps2)  # Intersection: PureSet(3)
```

### **Advanced Sorting**
```python
ps = PureSet("banana", "apple", "cherry")
sorted_ps = ps.sorted()
print(sorted_ps)  # Outputs: PureSet('apple', 'banana', 'cherry')
```
---

## **Installation**
Install PureSet via PyPI using the following command:

```bash
pip install pureset
```
---

## **Testing**
Run comprehensive unit tests to verify the robustness of PureSet:

```bash
pytest tests/
```
---

## **License**
This project is licensed under the [Apache License 2.0](LICENSE).