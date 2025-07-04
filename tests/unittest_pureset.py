"""Tests for PureSet."""

import unittest
import sys
import os
import timeit
import pickle
import copy
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from pureset import PureSet


class TestTypedSequence(unittest.TestCase):
    """Test cases for PureSet class."""

    def test_args_constructor(self):
        """Test variadic argument constructor."""
        ts = PureSet(1, 2, 3, 2, 4)
        self.assertEqual(ts.to_list(), [1, 2, 3, 4])

        ts = PureSet("hello", "world", "hello")
        self.assertEqual(ts.to_list(), ["hello", "world"])

        ts = PureSet(42)
        self.assertEqual(len(ts), 1)
        self.assertEqual(ts[0], 42)

    def test_empty_handling(self):
        """Test empty sequence handling."""
        ts = PureSet()
        self.assertEqual(len(ts), 0)
        self.assertEqual(ts.to_list(), [])

    def test_type_validation(self):
        """Test type consistency validation."""
        with self.assertRaises(TypeError) as cm:
            PureSet(1, "two", 3)
        self.assertIn("position 2", str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            PureSet(10, 20.0, 30)
        self.assertIn("<class 'int'>", str(cm.exception))

    def test_string_bytes_elements(self):
        """Test strings and bytes as elements."""
        ts = PureSet("hello", "world")
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts[0], "hello")

        ts = PureSet(b"hello", b"world")
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts[0], b"hello")

        with self.assertRaises(TypeError):
            PureSet("hello", b"world")

    def test_slicing(self):
        """Test slice operations."""
        ts = PureSet(1, 2, 3, 4, 5)
        self.assertEqual(ts[1:3].to_list(), [2, 3])
        self.assertEqual(ts[:2].to_list(), [1, 2])
        self.assertEqual(ts[2:].to_list(), [3, 4, 5])
        self.assertEqual(ts[::2].to_list(), [1, 3, 5])
        self.assertEqual(ts[::-1].to_list(), [5, 4, 3, 2, 1])
        self.assertIsInstance(ts[1:3], PureSet)

    def test_slicing_edge_cases(self):
        """Test edge cases in slicing."""
        ts = PureSet(1, 2, 3, 4, 5)

        # Out of bounds slicing
        self.assertEqual(ts[10:20].to_list(), [])
        self.assertEqual(ts[-10:-20].to_list(), [])

        # Negative indices
        self.assertEqual(ts[-3:-1].to_list(), [3, 4])
        self.assertEqual(ts[-1:].to_list(), [5])

        # Step values
        self.assertEqual(ts[::3].to_list(), [1, 4])
        self.assertEqual(ts[1::2].to_list(), [2, 4])
        self.assertEqual(ts[::-2].to_list(), [5, 3, 1])

    def test_index_method(self):
        """Test index() method."""
        ts = PureSet("a", "b", "c", "d")
        self.assertEqual(ts.index("b"), 1)
        self.assertEqual(ts.index("d"), 3)
        self.assertEqual(ts.index("c", 1), 2)
        self.assertEqual(ts.index("b", 0, 3), 1)

        with self.assertRaises(ValueError):
            ts.index("z")

        with self.assertRaises(ValueError):
            ts.index("a", 1, 3)

    def test_index_boundary_conditions(self):
        """Test boundary conditions for index method."""
        ts = PureSet(10, 20, 30, 40, 50)

        # Test with negative start/stop
        self.assertEqual(ts.index(30, -4), 2)
        self.assertEqual(ts.index(40, -3, -1), 3)

        # Test with out of bounds indices
        self.assertEqual(ts.index(10, -100), 0)
        self.assertEqual(ts.index(50, 0, 100), 4)

        # Test edge case where element is at boundary
        self.assertEqual(ts.index(10, 0, 1), 0)

    def test_count_method(self):
        """Test count() method - always returns 0 or 1 due to uniqueness."""
        ts = PureSet(1, 2, 3, 2, 4)
        self.assertEqual(ts.count(2), 1)  # Elements are unique
        self.assertEqual(ts.count(5), 0)
        self.assertEqual(ts.count(1), 1)

    def test_reverse_method(self):
        """Test reverse() method."""
        ts = PureSet(1, 2, 3, 4)
        rev = ts.reverse()
        self.assertEqual(rev.to_list(), [4, 3, 2, 1])
        self.assertIsInstance(rev, PureSet)
        self.assertEqual(ts.to_list(), [1, 2, 3, 4])

    def test_ensure_compatible_public(self):
        """Test ensure_compatible method."""
        ts1 = PureSet(1, 2, 3)
        ts2 = PureSet(4, 5, 6)
        ts_str = PureSet("a", "b", "c")

        self.assertIs(ts1.compatible(ts2), ts2)

        with self.assertRaises(TypeError) as cm:
            ts1.compatible(ts_str)
        self.assertIn("Incompatible element types", str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            ts1.compatible([1, 2, 3])
        self.assertIn("Expected PureSet", str(cm.exception))

    def test_set_operations(self):
        """Test set operations."""
        a = PureSet(1, 2, 3)
        b = PureSet(3, 4, 2)

        self.assertEqual((a | b).to_list(), [1, 2, 3, 4])
        self.assertEqual((a & b).to_list(), [2, 3])
        self.assertEqual((a - b).to_list(), [1])
        self.assertEqual((a ^ b).to_list(), [1, 4])

    def test_hash_and_equality(self):
        """Test hashing and equality."""
        ts1 = PureSet(1, 2, 3)
        ts2 = PureSet(1, 2, 3)
        ts3 = PureSet(3, 2, 1)

        self.assertEqual(ts1, ts2)
        self.assertNotEqual(ts1, ts3)
        self.assertEqual(hash(ts1), hash(ts2))

        d = {ts1: "value"}
        self.assertEqual(d[ts2], "value")

    def test_immutability(self):
        """Test immutability enforcement."""
        ts = PureSet(1, 2, 3)

        with self.assertRaises(AttributeError) as cm:
            ts.new_attr = "value"
        self.assertIn("immutable", str(cm.exception))

    def test_performance(self):
        """Test performance characteristics."""
        setup = "from src.pureset import PureSet; ts = PureSet(*range(1000))"

        int_time = timeit.timeit("ts[500]", setup, number=100000)
        val_time = timeit.timeit("ts[500]", setup, number=100000)

        self.assertLess(int_time, 0.1)
        self.assertLess(val_time, 0.1)

        data = list(range(1000)) * 2
        setup = f"from src.pureset import PureSet; data = {data}"
        creation_time = timeit.timeit("PureSet(*data)", setup, number=100)

        self.assertLess(creation_time, 1.0)

    def test_complex_types(self):
        """Test with custom object types."""

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __eq__(self, other):
                return (
                    isinstance(other, Point) and self.x == other.x and self.y == other.y
                )

            def __hash__(self):
                return hash((self.x, self.y))

        p1 = Point(1, 2)
        p2 = Point(3, 4)
        p3 = Point(1, 2)

        ts = PureSet(p1, p2, p3)
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts[0], p1)
        self.assertEqual(ts[1], p2)

    def test_tuple_heteroginity(self):
        """Tests heterogenous tuple entries."""
        with self.assertRaises(TypeError) as cm:
            PureSet(
                (0, "John"),
                (2.5, "Maria"),
                ("Peter", "Bob"),
                (1, 2, 3),
            )
        self.assertIn("Incompatible", str(cm.exception))

    def test_dict_equality(self):
        """Tests equality between dict entries."""
        ts = PureSet(
            {"id": 0, "name": "John", "age": 15},
            {"id": 1, "name": "Maria", "age": 20},
            {"id": 2, "name": "John", "age": 25},
        )

        self.assertEqual(ts[0] == {"id": 0, "name": "John", "age": 15}, True)
        self.assertEqual(ts[0] == {"id": 1, "name": "Maria", "age": 20}, False)
        self.assertEqual(ts[1], {"id": 1, "name": "Maria", "age": 20})

        ts2 = PureSet(
            {"id": 0, "name": "John", "age": 15},
            {"id": 1, "name": "Maria", "age": 20},
            {"id": 2, "name": "John", "age": 25},
        )
        self.assertEqual(ts, ts2)
        self.assertEqual(ts[0], ts2[0])

        ts3 = PureSet(
            {"id": 0, "name": "John", "age": 25},
            {"id": 1, "name": "Maria", "age": 20},
            {"id": 2, "name": "John", "age": 15},
        )
        self.assertNotEqual(ts, ts3)
        self.assertNotEqual(ts[0], ts3[0])

    def test_dict_heteroginity(self):
        """Tests heterogenous dict entries."""
        with self.assertRaises(TypeError) as cm:
            PureSet(
                {0: "John"},
                {2.5: "Maria"},
                {"Peter": 20},
            )
        self.assertIn("Incompatible", str(cm.exception))

    def test_contains_operator(self):
        """Test __contains__ operator (in)."""
        ts = PureSet(1, 2, 3, 4, 5)

        self.assertIn(3, ts)
        self.assertNotIn(10, ts)
        self.assertIn(1, ts)
        self.assertIn(5, ts)

        ts_str = PureSet("hello", "world", "test")
        self.assertIn("hello", ts_str)
        self.assertNotIn("python", ts_str)

    def test_iteration(self):
        """Test __iter__ method."""
        ts = PureSet(10, 20, 30, 40)

        # Test basic iteration
        result = []
        for item in ts:
            result.append(item)
        self.assertEqual(result, [10, 20, 30, 40])

        # Test list comprehension
        doubled = [x * 2 for x in ts]
        self.assertEqual(doubled, [20, 40, 60, 80])

        # Test with enumerate
        indexed = list(enumerate(ts))
        self.assertEqual(indexed, [(0, 10), (1, 20), (2, 30), (3, 40)])

    def test_repr_and_str(self):
        """Test __repr__ and __str__ methods."""
        ts = PureSet(1, 2, 3)

        # Test repr
        repr_str = repr(ts)
        self.assertIn("PureSet", repr_str)
        self.assertIn("1", repr_str)
        self.assertIn("2", repr_str)
        self.assertIn("3", repr_str)

        # Test str (if different from repr)
        str_str = str(ts)
        self.assertIsInstance(str_str, str)

        # Test with empty sequence
        ts_empty = PureSet()
        repr_empty = repr(ts_empty)
        self.assertIn("PureSet", repr_empty)

    def test_value_lookup_getitem(self):
        """Test special value lookup feature in __getitem__."""
        ts = PureSet("a", "b", "c")

        # Test value lookup
        self.assertEqual(ts["a"], "a")
        self.assertEqual(ts["b"], "b")

        # Test KeyError for non-existent value
        with self.assertRaises(KeyError) as cm:
            ts["z"]
        self.assertIn("not found", str(cm.exception))

    def test_pos_method(self):
        """Test pos() method for positional access."""
        ts = PureSet(10, 20, 30)

        self.assertEqual(ts.pos(0), 10)
        self.assertEqual(ts.pos(1), 20)
        self.assertEqual(ts.pos(2), 30)

        # Test out of bounds
        with self.assertRaises(IndexError) as cm:
            ts.pos(10)
        self.assertIn("out of range", str(cm.exception))

    def test_none_values(self):
        """Test handling of None values."""
        # Test sequence with None values
        ts = PureSet(None, None, None)
        self.assertEqual(len(ts), 1)  # Deduplicated
        self.assertEqual(ts[0], None)

        # Test mixed None with other types should fail
        with self.assertRaises(TypeError):
            PureSet(None, 1, 2)

    def test_float_types(self):
        """Test with float types."""
        ts = PureSet(1.5, 2.5, 3.5, 2.5)
        self.assertEqual(ts.to_list(), [1.5, 2.5, 3.5])
        self.assertEqual(len(ts), 3)

        # Test float operations
        self.assertEqual(ts.count(2.5), 1)
        self.assertEqual(ts.index(3.5), 2)

        # Test float with special values
        ts_special = PureSet(float("inf"), float("-inf"), float("nan"))
        self.assertEqual(len(ts_special), 3)

    def test_boolean_types(self):
        """Test with boolean types."""
        ts = PureSet(True, False, True, False, True)
        self.assertEqual(ts.to_list(), [True, False])
        self.assertEqual(len(ts), 2)

        # Test boolean operations
        self.assertEqual(ts.count(True), 1)
        self.assertEqual(ts.count(False), 1)
        self.assertEqual(ts.index(False), 1)

    def test_pickling(self):
        """Test pickle/unpickle support."""
        ts = PureSet(1, 2, 3, 4, 5)

        # Test pickling
        pickled = pickle.dumps(ts)
        unpickled = pickle.loads(pickled)

        self.assertEqual(ts, unpickled)
        self.assertEqual(ts.to_list(), unpickled.to_list())
        self.assertEqual(hash(ts), hash(unpickled))

        # Test with complex types
        ts_complex = PureSet({"a", "b", "c"}, {"x", "y", "z"}, {"1", "2", "3"})
        pickled_complex = pickle.dumps(ts_complex)
        unpickled_complex = pickle.loads(pickled_complex)

        self.assertEqual(ts_complex, unpickled_complex)

    def test_copy_operations(self):
        """Test copy and deepcopy operations."""
        ts = PureSet([1, 2], [3, 4], [5, 6])

        # Test shallow copy
        ts_copy = copy.copy(ts)
        self.assertEqual(ts, ts_copy)
        self.assertIsNot(ts, ts_copy)

        # Test deepcopy
        ts_deepcopy = copy.deepcopy(ts)
        self.assertEqual(ts, ts_deepcopy)
        self.assertIsNot(ts, ts_deepcopy)

        # Verify deep copy created new objects
        self.assertIsNot(ts[0], ts_deepcopy[0])
        self.assertEqual(ts[0], ts_deepcopy[0])

    def test_large_sequences(self):
        """Test with very large sequences."""
        # Test creation with large sequence
        large_data = list(range(10000))
        ts = PureSet(*large_data)

        # Deduplicated size should be 10000
        self.assertEqual(len(ts), 10000)

        # Test operations on large sequence
        self.assertEqual(ts[5000], 5000)
        self.assertEqual(ts[-1], 9999)
        self.assertIn(7500, ts)

        # Test slicing large sequence
        subset = ts[1000:2000]
        self.assertEqual(len(subset), 1000)
        self.assertEqual(subset[0], 1000)

    def test_frozen_dataclass(self):
        """Test with frozen dataclasses."""

        @dataclass(frozen=True)
        class FrozenPoint:
            x: int
            y: int

        p1 = FrozenPoint(1, 2)
        p2 = FrozenPoint(3, 4)
        p3 = FrozenPoint(1, 2)

        ts = PureSet(p1, p2, p3)
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts[0], p1)
        self.assertEqual(ts[1], p2)

        # Test that frozen objects work in set operations
        ts2 = PureSet(p2, FrozenPoint(5, 6))
        intersection = ts & ts2
        self.assertEqual(len(intersection), 1)
        self.assertEqual(intersection[0], p2)

    def test_nested_typed_sequences(self):
        """Test PureSet containing other TypedSequences."""
        ts1 = PureSet(1, 2, 3)
        ts2 = PureSet(4, 5, 6)
        ts3 = PureSet(1, 2, 3)  # Same as ts1

        # Create nested PureSet
        nested = PureSet(ts1, ts2, ts3)
        self.assertEqual(len(nested), 2)  # ts1 and ts3 are equal
        self.assertEqual(nested[0], ts1)
        self.assertEqual(nested[1], ts2)

        # Test operations on nested sequences
        self.assertIn(ts1, nested)
        self.assertEqual(nested.count(ts1), 1)

    def test_to_methods(self):
        """Test conversion methods."""
        ts = PureSet(3, 1, 4, 1, 5)

        # Test to_list
        lst = ts.to_list()
        self.assertEqual(lst, [3, 1, 4, 5])
        self.assertIsInstance(lst, list)

        # Test to_tuple
        tup = ts.to_tuple()
        self.assertEqual(tup, (3, 1, 4, 5))
        self.assertIsInstance(tup, tuple)

        # Test to_frozenset for hashable
        fs = ts.to_frozenset()
        self.assertEqual(fs, frozenset([3, 1, 4, 5]))
        self.assertIsInstance(fs, frozenset)

        # Test to_frozenset for unhashable
        ts_unhashable = PureSet([1, 2], [3, 4])
        with self.assertRaises(TypeError) as cm:
            ts_unhashable.to_frozenset()
        self.assertIn("unhashable", str(cm.exception))

    def test_thread_safety(self):
        """Test thread safety of PureSet operations."""
        ts = PureSet(*range(1000))
        results = []
        errors = []

        def access_sequence(index):
            try:
                # Perform multiple operations
                val = ts[index]
                contains = index in ts
                count = ts.count(index)
                results.append((val, contains, count))
            except Exception as e:
                errors.append(e)

        # Run multiple threads accessing the sequence
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(access_sequence, i) for i in range(100)]
            for future in futures:
                future.result()

        # Check no errors occurred
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 100)

        # Verify all results are correct
        for i, (val, contains, count) in enumerate(results):
            self.assertEqual(val, i)
            self.assertTrue(contains)
            self.assertEqual(count, 1)

    def test_signature(self):
        """Test .signature property."""
        ts1 = PureSet(
            {"id": 0, "name": "John", "age": 15},
            {"id": 1, "name": "Alice", "age": 20},
            {"id": 2, "name": "Bob", "age": 25},
        )
        sig = ts1.signature
        self.assertEqual(sig, (dict, {"id": int, "name": str, "age": int}))
        
        ts2 = PureSet(
            {"id": 0, "name": "John", "age": 15},
            {"id": 1, "name": "Alice", "age": 20},
            {"id": 2, "name": "Bob", "age": 25},
        )
        self.assertEqual(ts1.signature, ts2.signature)
        self.assertTrue(ts1.compatible(ts2))
        
        data = PureSet(
            [[0.2, 0.4, 0.8, "a", "b", "c"], True],
            [[0.1, 0.2, 0.3, "x", "y", "z"], False],
            [[0.3, 0.4, 0.5, "d", "e", "f"], False],
        )
        self.assertEqual(data.signature, (tuple, (tuple, (float, 3), (str, 3)), bool))


if __name__ == "__main__":
    unittest.main()