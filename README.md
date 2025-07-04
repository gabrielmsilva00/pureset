# **PureSet**

**PureSet** is an advanced, immutable, ordered, and hashable collection type for Python. It ensures **type homogeneity** across elements, making it a robust replacement for both sets and sequences in numerous real-world production applications. Designed for seasoned Python developers, PureSet offers unparalleled accuracy, predictability, and clarity in managing homogeneous data structures.

---

## **Introduction & Core Features**
1. **Immutability**: Ensures data integrity and prevents accidental modifications.
2. **Ordering**: Retains insertion order, enabling predictable iteration and application as an alternative to `Enums`.
3. **Hashability**: Leverages immutability for seamless integration as dictionary keys or other hash-based collections.
4. **Uniqueness**: Automatically removes duplicates and enforces type consistency across all elements.
5. **Custom Class Support**: Fully supports custom, structured data while ensuring consistent shapes.
---

## **Installation & Requirements**
To install the `PureSet` package, simply use pip:
```bash
pip install pureset
```
- **Python Versions**: Compatible with Python 3.9+.
- **Dependencies**: Pure Python, with no additional dependencies.
---

## **API Overview**
This section provides real-world applications and examples for senior developers to integrate PureSet effectively into their workflows.
```python
from pureset import PureSet
```
---

### **1. Enum Replacement**
Utilize `PureSet` for domain-state modeling or enumerated lookup values.
```python

ORDER_STATES = PureSet("Pending", "Completed", "Shipped", "Cancelled")

# Test State Membership
current_state = "Shipped"
if current_state in ORDER_STATES:
    print(f"Order is in valid state: {current_state}")
```
---

### **2. Validating Homogeneity of Complex Data**
Guarantee same structure across dictionaries or custom data types.
```python
user_profiles = PureSet(
    {"name": "Alice", "age": 25},
    {"name": "Bob", "age": 30}
)

# Adding a mismatched schema raises an error
try:
    user_profiles_with_error = PureSet(
        {"name": "Alice", "age": 25},
        {"username": "Charlie"}  # Error: Different schema
    )
except TypeError as e:
    print(e)
```
---

### **3. Hashable Object for Dependency Injection**
Use PureSet as immutable and hashable keys in immutable lookups.

```python
pipeline_configuration = PureSet("Step1", "Step2", "Step3")

config_cache = {pipeline_configuration: "All steps completed"}
print(config_cache[pipeline_configuration])  # Outputs: All steps completed
```
---

### **4. Data Stream Validation**
Filter valid entries from data pipelines based on predictable signatures.
```python
VALID_EVENTS = PureSet("Click", "Scroll", "Hover")

incoming_events = ["Click", "InvalidEvent", "Hover"]
filtered_events = [event for event in incoming_events if event in VALID_EVENTS]

print(filtered_events)  # Outputs: ['Click', 'Hover']
```
---

### **5. Priority Ordering in Workflow Management**
Simplify priority queues or execution rankings with immutable ordering.
```python
TASK_PRIORITIES = PureSet("High", "Medium", "Low")

# Maintain strict ordering
for priority in TASK_PRIORITIES:
    print(priority)
# Outputs: High, Medium, Low
```
---

### **6. Managing Nested Data**
Validate nested collection types like tuples or lists.
```python
nested = PureSet(
    (1, [2, 3]),
    (4, [5, 6])
)

print(nested)
# Outputs: PureSet((1, [2, 3]), (4, [5, 6]))
```
---

### **7. Function Results Validation**
Guarantee the uniformity of results returned by various processes.
```python
# Ensuring results adhere to expected return signatures
results = PureSet((x, x ** 2) for x in range(3))
print(results)
# Outputs: PureSet((0, 0), (1, 1), (2, 4))
```
---

### **8. Reversible Sets for Dynamic Processes**
Reverse structured outcomes without mutation.
```python
values = PureSet(10, 20, 30, 40)
reversed_values = values.reverse()

print(reversed_values)  # Outputs: PureSet(40, 30, 20, 10)
```

---

## **License**
This project is released under the **Apache License 2.0**. Please review the [LICENSE](LICENSE) file for further details.

---
