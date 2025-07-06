import unittest
import timeit
import pickle
import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any
from collections import namedtuple, UserList, UserDict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from pureset import PureSet

##### === CONSTRUCTION BASICS === #####

class TestPureSetConstruction(unittest.TestCase):
    def test_basic_uniqueness(self):
        self.assertEqual(PureSet(1,2,1,3).to_list(), [1,2,3])
        self.assertEqual(PureSet("x", "y", "x").to_list(), ["x", "y"])
        self.assertEqual(len(PureSet(3,)), 1)

    def test_empty(self):
        self.assertEqual(len(PureSet()), 0)
        self.assertEqual(PureSet().to_list(), [])

    def test_type_homogeneity(self):
        with self.assertRaises(TypeError):
            PureSet(1, "a")
        with self.assertRaises(TypeError):
            PureSet(1, 2.0)

    def test_str_bytes(self):
        self.assertEqual(PureSet("a", "b")[0], "a")
        self.assertEqual(PureSet(b"x", b"y")[0], b"x")
        with self.assertRaises(TypeError):
            PureSet("a", b"a")

    def test_none_values(self):
        self.assertEqual(len(PureSet(None, None)), 1)
        with self.assertRaises(TypeError):
            PureSet(None, 2)

##### === INDEXING AND SLICING === #####

class TestPureSetIndexing(unittest.TestCase):
    def test_indexing(self):
        ts = PureSet("a", "b", "c")
        self.assertEqual(ts[0], "a")
        self.assertEqual(ts[-1], "c")

    def test_slice(self):
        nums = PureSet(0,1,2,3,4)
        self.assertEqual(nums[2:].to_list(), [2,3,4])
        self.assertEqual(nums[:2].to_list(), [0,1])
        self.assertEqual(nums[::-1].to_list(), [4,3,2,1,0])
        self.assertIsInstance(nums[1:3], PureSet)
        self.assertEqual(nums[10:20].to_list(), [])

    def test_index_method(self):
        ts = PureSet("z", "y", "w")
        self.assertEqual(ts.index("y"), 1)
        with self.assertRaises(ValueError):
            ts.index("nope")

    def test_value_lookup_getitem(self):
        ts = PureSet("aa", "bb")
        self.assertEqual(ts["bb"], "bb")
        with self.assertRaises(KeyError):
            ts["cc"]

    def test_pos(self):
        ts = PureSet(10, 20)
        self.assertEqual(ts.pos(1), 20)
        with self.assertRaises(IndexError):
            ts.pos(9)

    def test_count_uniqueness(self):
        ts = PureSet(1,1,2)
        self.assertEqual(ts.count(1), 1)
        self.assertEqual(ts.count(99), 0)


##### === HASHING, EQUALITY, REPR, STR, AND ITER === #####

class TestPureSetObjectBehavior(unittest.TestCase):
    def test_equality_and_hash(self):
        a, b = PureSet(1,2,3), PureSet(1,2,3)
        c = PureSet(3,2,1)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(hash(a), hash(b))
        d = {a: "v"}
        self.assertEqual(d[b], "v")
    
    def test_repr_and_str_empty(self):
        ts = PureSet()
        self.assertTrue("PureSet" in repr(ts))
        self.assertEqual(repr(ts), "PureSet()")
        self.assertTrue(isinstance(str(ts), str))

    def test_basic_iteration(self):
        ts = PureSet(1,2,3,4)
        self.assertEqual(list(ts), [1,2,3,4])

##### === SET AND SEQUENCE OPERATIONS === #####

class TestPureSetOps(unittest.TestCase):
    def test_union_intersection_difference_xor(self):
        a, b = PureSet(1, 2, 3), PureSet(2, 3, 4)
        self.assertEqual((a | b).to_list(), [1,2,3,4])
        self.assertEqual((a & b).to_list(), [2,3])
        self.assertEqual((a - b).to_list(), [1])
        self.assertEqual((a ^ b).to_list(), [1,4])

    def test_add_concat(self):
        ts1, ts2 = PureSet(0,1), PureSet(1,2)
        self.assertEqual((ts1 + ts2).to_list(), [0,1,2])

    def test_reverse_sorted_map_filter(self):
        ts = PureSet(7, 4, 1)
        self.assertEqual(ts.reverse().to_list(), [1,4,7])
        self.assertEqual(ts.sorted().to_list(), [1,4,7])
        self.assertEqual(ts.map(str).to_list(), ["7","4","1"])
        self.assertEqual(ts.filter(lambda x: x%2).to_list(), [7,1])

    def test_copy_and_deepcopy(self):
        ts = PureSet([1,2],[3,4])
        ts_copy = copy.copy(ts)
        ts_deep = copy.deepcopy(ts)
        self.assertEqual(ts, ts_copy)
        self.assertIsNot(ts[0], ts_deep[0])

##### === SIGNATURES AND TYPING === #####

class TestPureSetSignature(unittest.TestCase):
    def test_homogeneous_signature(self):
        self.assertEqual(
            PureSet({'x': 1, 'y': 2}, {'x':2,'y':3}).signature,
            (dict, {'x': int, 'y': int})
        )

    def test_signature_nested(self):
        ts = PureSet([[{"a": 1}], [{"a": 2}]])
        self.assertEqual(ts.signature, (list, ((list, (dict, {'a': int})), 2)))

    def test_type_error_on_incompatible_signature(self):
        with self.assertRaises(TypeError):
            PureSet({'x':1}, {'x':'str'})

##### === TESTS FOR RESTORE AND ROUND-TRIP === #####

class TestPureSetRestore(unittest.TestCase):
    def test_restore_round_trip(self):
        x = [{'a': [1, 2]}, {'a': [3, 4]}]
        ps = PureSet(*x)
        for i, y in enumerate(x):
            self.assertEqual(ps[i], y)
        self.assertEqual(ps.to_list(), x)
        self.assertEqual(PureSet.restore(PureSet.freeze(x)), x)

    def test_nested_pureset_round_trip(self):
        ps1 = PureSet(1, 2)
        ps2 = PureSet(3, 4)
        big = PureSet(ps1, ps2, ps1)
        self.assertEqual(len(big), 2)
        self.assertEqual(big[0], ps1)
        self.assertEqual(big[1], ps2)
        back = PureSet.restore(PureSet.freeze(big))
        self.assertEqual(back, big)

    def test_namedtuple_and_enum(self):
        Pt = namedtuple("Pt", "x y")
        Red = Enum('Red', {'A': 1, 'B': 2})
        pt = Pt(1, 2)
        col = Red.A
        self.assertEqual(PureSet.restore(PureSet.freeze(pt)), pt)
        self.assertEqual(PureSet.restore(PureSet.freeze(col)), col)

    def test_userlist_and_userdict(self):
        ul = UserList([3,4])
        ud = UserDict({'x': 1})
        for obj in (ul, ud):
            frz = PureSet.freeze(obj)
            self.assertEqual(type(PureSet.restore(frz)), type(obj))

    def test_error_on_unfreezable_object(self):
        class CustomUnfreezable:
            def __eq__(self, other): return id(self) == id(other)
            def __hash__(self): return id(self)

        class NonFreezable:
            __hash__ = None

        class F: pass

        with self.assertRaises(TypeError):
            PureSet.freeze(NonFreezable())

        with self.assertRaises(TypeError):
            PureSet.freeze(CustomUnfreezable())

        with self.assertRaises(TypeError):
            PureSet.freeze(F())

##### === ADVANCED CASE TESTS === #####
class TestPureSetEdgeCases(unittest.TestCase):
    def test_slots_and_dict_compatibility(self):
        class S:
            __slots__ = ('foo',)
            def __init__(self, foo): self.foo = foo
        class D:
            def __init__(self, foo): self.foo = foo
        with self.assertRaises(TypeError):
            PureSet(S(1), D(1))
        x = PureSet(S(2), S(3))
        self.assertEqual(x[1].foo, 3)

    def test_class_inheritance_signature(self):
        class Base: pass
        class Child(Base): pass
        with self.assertRaises(TypeError):
            PureSet(Base(), Child())

    def test_dataclass_inheritance_signature(self):
        @dataclass
        class A: x: int
        @dataclass
        class B(A): y: int
        with self.assertRaises(TypeError):
            PureSet(A(1), B(2,3))

    def test_metaclasses(self):
        class MetaA(type): pass
        class MetaB(type): pass
        class A(metaclass=MetaA): pass
        class B(metaclass=MetaB): pass
        with self.assertRaises(TypeError): PureSet(A(), B())

    def test_field_init_in_postinit(self):
        @dataclass
        class C:
            x: int
            def __post_init__(self): self.y = self.x * 2
        ts = PureSet(C(1), C(2))
        self.assertTrue(hasattr(ts[0], "y") and ts[0].y == 2)

    def test_abc_numbers(self):
        import numbers
        class MyInt(numbers.Integral):
            def __init__(self, value): self.value = value
            def __index__(self): return self.value
            def __abs__(self): return MyInt(abs(self.value))
            def __floor__(self): return MyInt(int(self.value))
            def __ceil__(self): return MyInt(int(self.value))
            def __add__(self, other): return MyInt(self.value + int(other))
            def __eq__(self, other): return int(self) == int(other)
            def __le__(self, other): return int(self) <= int(other)
            def __lt__(self, other): return int(self) < int(other)
            def __round__(self): return MyInt(int(self.value))
            def __trunc__(self): return MyInt(int(self.value))
            def __hash__(self): return hash(self.value)
            def __bool__(self): return bool(self.value)
            def __neg__(self): return MyInt(-self.value)
            def __pos__(self): return MyInt(self.value)
            def __int__(self): return self.value
            def __float__(self): return float(self.value)
            def __complex__(self): return complex(self.value)
            def __mul__(self, other): return MyInt(self.value * int(other))
            def __truediv__(self, other): return MyInt(self.value // int(other))
            def __floordiv__(self, other): return MyInt(self.value // int(other))
            def __mod__(self, other): return MyInt(self.value % int(other))
            def __pow__(self, other): return MyInt(self.value ** int(other))
            def __lshift__(self, other): return MyInt(self.value << int(other))
            def __rshift__(self, other): return MyInt(self.value >> int(other))
            def __and__(self, other): return MyInt(self.value & int(other))
            def __or__(self, other): return MyInt(self.value | int(other))
            def __xor__(self, other): return MyInt(self.value ^ int(other))
            def __invert__(self): return MyInt(~self.value)
            def __radd__(self, other): return MyInt(int(other) + self.value)
            def __rsub__(self, other): return MyInt(int(other) - self.value)
            def __rmul__(self, other): return MyInt(int(other) * self.value)
            def __rtruediv__(self, other): return MyInt(int(other) // self.value)
            def __rfloordiv__(self, other): return MyInt(int(other) // self.value)
            def __rmod__(self, other): return MyInt(int(other) % self.value)
            def __rpow__(self, other): return MyInt(int(other) ** self.value)
            def __rlshift__(self, other): return MyInt(int(other) << self.value)
            def __rrshift__(self, other): return MyInt(int(other) >> self.value)
            def __rand__(self, other): return MyInt(int(other) & self.value)
            def __ror__(self, other): return MyInt(int(other) | self.value)
            def __rxor__(self, other): return MyInt(int(other) ^ self.value)
            def __rinvert__(self, other): return MyInt(~int(other))
            def __repr__(self): return f"MyInt({self.value})"
            def denominator(self): return 1
            def numerator(self): return self.value
            def real(self): return self.value
            def imag(self): return 0
            def conjugate(self): return self
            def bit_length(self): return self.value.bit_length()
        a, b = MyInt(1), MyInt(2)
        ts = PureSet(a, b)
        self.assertEqual(len(ts), 2)
        self.assertIsInstance(ts[0], MyInt)

##### === TYPE CONSISTENCY AND UNHASHABLES === #####
class TestTypeConsistency(unittest.TestCase):
    def test_mixed_numeric_types(self):
        with self.assertRaises(TypeError): PureSet(1, 2.0)
        ts1, ts2 = PureSet(1,2,3), PureSet(1.0,2.0,3.0)
        self.assertNotEqual(ts1.signature, ts2.signature)

    def test_dict_incompatibility(self):
        with self.assertRaises(TypeError): PureSet({'a': 1}, {'a': 2.0})
        with self.assertRaises(TypeError): PureSet({'a': 1}, {'b': 2})

    def test_tuple_heteroginity(self):
        with self.assertRaises(TypeError):
            PureSet((0, "John"), (1, 2), ("Peter", "Bob"))
        with self.assertRaises(TypeError):
            PureSet((1, 2), (1, 2, 3))

    def test_incompatible_classvar(self):
        class A:
            x:int
            classvar=42
            def __init__(self, x): self.x=x
        class B: 
            def __init__(self, x): self.y=x
        with self.assertRaises(TypeError): PureSet(A(1), B(1))

##### === PERFORMANCE === #####
import math
n   = min(300, 10**6)
gap = abs(92 - (math.log2(n)))

class TestPureSetPerformance(unittest.TestCase):
    def test_construction_speed_vs_set(self):
        data = [i for i in range(n)] * 2
        t_set = timeit.timeit("set(data)", globals={"data": data}, number=10)
        t_ps = timeit.timeit("PureSet(*data)", globals={"data": data, "PureSet": PureSet}, number=10)
        t_gap = ((100 * t_ps) / t_set)
        t_diff = ((100 * (t_set * gap)) / t_set)
        self.assertLess(
            t_ps,
            t_set * gap,
            f"\nPureSet slower than set by {t_gap:.2f}%\n"
            f"(set: {t_set:.5f}s, PureSet: {t_ps:.5f}s)\n"
            f"Allowed gap: {t_set * gap:.5f}s ({t_diff:.2f}%)\n"
        )

    def test_membership_speed_vs_set(self):
        data = [i for i in range(n)]
        s = set(data)
        ps = PureSet(*data)
        idx = n // 2
        t_set = timeit.timeit("idx in s", globals={"idx": idx, "s": s}, number=100000)
        t_ps  = timeit.timeit("idx in ps", globals={"idx": idx, "ps": ps}, number=100000)
        t_gap = ((100 * t_ps) / t_set)
        t_diff = ((100 * (t_set * gap)) / t_set)
        self.assertLess(
            t_ps,
            t_set * gap,
            f"\nPureSet slower than set by {t_gap:.2f}%\n"
            f"(set: {t_set:.5f}s, PureSet: {t_ps:.5f}s)\n"
            f"Allowed gap: {t_set * gap:.5f}s ({t_diff:.2f}%)\n"
        )

    def test_to_frozenset_speed_vs_set(self):
        data = [i for i in range(n)]
        ps = PureSet(*data)
        t_set = timeit.timeit("frozenset(data)", globals={"data": data}, number=100)
        t_ps  = timeit.timeit("ps.to_frozenset()", globals={"ps": ps}, number=100)
        t_gap = ((100 * t_ps) / t_set)
        t_diff = ((100 * (t_set * gap)) / t_set)
        self.assertLess(
            t_ps,
            t_set * gap,
            f"\nPureSet slower than set by {t_gap:.2f}%\n"
            f"(set: {t_set:.5f}s, PureSet: {t_ps:.5f}s)\n"
            f"Allowed gap: {t_set * gap:.5f}s ({t_diff:.2f}%)\n"
        )

##### === UTILITY/EXTENSION === #####
class TestUtilityAndExtensions(unittest.TestCase):
    def test_to_list_tuple_frozenset(self):
        ts = PureSet("a","b","c")
        self.assertIsInstance(ts.to_list(), list)
        self.assertIsInstance(ts.to_tuple(), tuple)
        self.assertIsInstance(ts.to_frozenset(), frozenset)
        self.assertEqual(set(ts.to_list()), set(ts.to_frozenset()))

    def test_pickle_and_copy_roundtrip(self):
        ts = PureSet(1,2,3,4,5)
        ts2 = pickle.loads(pickle.dumps(ts))
        self.assertEqual(ts, ts2)

    def test_deepcopy_returns_new_objects(self):
        ts = PureSet([7,8],[9,10])
        ts2 = copy.deepcopy(ts)
        self.assertEqual(ts, ts2)
        self.assertIsNot(ts[0], ts2[0])

    def test_value_lookup_and_notfound(self):
        ts = PureSet(11,12,13)
        self.assertEqual(ts.get(12), 12)
        self.assertIsNone(ts.get(99))
        self.assertEqual(ts.get(99, "na"), "na")

##### === EXTREME AND RARE EDGE CASES === #####
class TestPureSetEdgeContainers(unittest.TestCase):
    def test_freeze_memoryview(self):
        data = memoryview(b'abcde')
        ps = PureSet(data)
        self.assertEqual(ps[0].tobytes(), b'abcde')
        frozen = PureSet.freeze(data)
        restored = PureSet.restore(frozen)
        self.assertIsInstance(restored, memoryview)
        self.assertEqual(restored.tobytes(), data.tobytes())

    def test_freeze_range(self):
        r = range(10, 20, 2)
        ps = PureSet(r)
        self.assertEqual(ps[0], r)
        frozen = PureSet.freeze(r)
        restored = PureSet.restore(frozen)
        self.assertEqual(list(restored), list(r))

    def test_freeze_array(self):
        import array
        arr = array.array("i", [1,2,3])
        frozen = PureSet.freeze(arr)
        restored = PureSet.restore(frozen)
        self.assertIsInstance(restored, array.array)
        self.assertEqual(restored.tolist(), arr.tolist())
        arr2 = array.array("f", [1.1, 2.2])
        with self.assertRaises(TypeError):
            PureSet(arr, arr2)

    def test_mix_bytes_bytearray_memoryview(self):
        arr = bytearray(b"abc")
        mem = memoryview(arr)
        with self.assertRaises(TypeError):
            PureSet(arr, mem)

    def test_chainmap_counter_ordereddict_defaultdict(self):
        import collections
        ch = collections.ChainMap({"a": 1}, {"b": 2})
        ct = collections.Counter({'x': 2, 'y': 3})
        od = collections.OrderedDict(a=1, b=2)
        dd = collections.defaultdict(int, foo=42)
        types = [ch, ct, od, dd]
        for container in types:
            frozen = PureSet.freeze(container)
            restored = PureSet.restore(frozen)
            self.assertEqual(type(restored), type(container))
            if hasattr(container, 'items'):
                self.assertEqual(dict(restored.items()), dict(container.items()))
    def test_userstring_userlist_userdict(self):
        from collections import UserString, UserList, UserDict
        us = UserString("test")
        s = "test"
        ul = UserList([1, 2, 3])
        ud = UserDict({'foo': 99})
        with self.assertRaises(TypeError):
            PureSet(us, s)
        self.assertEqual(PureSet(us)[0], us)
        self.assertEqual(PureSet(ul)[0], ul)
        self.assertEqual(PureSet(ud)[0], ud)

class TestPureSetWithNumpy(unittest.TestCase):
    def test_numpy_array_round_trip(self):
        import numpy as np
        arr = np.arange(6).reshape(2, 3)
        ps = PureSet(arr)
        self.assertIsInstance(ps[0], np.ndarray)
        self.assertTrue(np.array_equal(ps[0], arr))
        with self.assertRaises(TypeError):
            PureSet(arr, arr.tolist())
        sc = np.float64(1.23)
        ps = PureSet(sc)
        self.assertAlmostEqual(ps[0], float(sc))
    def test_numpy_string_datetime(self):
        import numpy as np
        arr = np.array(['a', 'b', 'c'], dtype='U')
        ps = PureSet(arr)
        self.assertTrue((ps[0] == arr).all())
        dtarr = np.array(['2023-01-01', '2024-04-05'], dtype='M')
        ps = PureSet(dtarr)
        self.assertTrue((ps[0] == dtarr).all())
    def test_incompatible_signatures(self):
        import numpy as np
        a1 = np.arange(3)
        a2 = np.arange(3).reshape(1,3)
        with self.assertRaises(TypeError):
            PureSet(a1, a2)

class TestPureSetWithPandas(unittest.TestCase):
    def test_dataframe_series_index(self):
        import pandas as pd
        df = pd.DataFrame({'a':[1,2],'b':[3,4]})
        ser = pd.Series([9, 8, 7], index=['x', 'y', 'z'])
        idx = pd.Index([5, 7, 9])
        for obj in (df, ser, idx):
            ps = PureSet(obj)
            self.assertEqual(type(ps[0]), type(obj))
            restored = PureSet.restore(PureSet.freeze(obj))
            self.assertEqual(type(restored), type(obj))
            if hasattr(obj, 'values'):
                self.assertTrue((restored.values == obj.values).all())
    def test_no_mix_list_pandas(self):
        import pandas as pd
        ser = pd.Series([1,2,3])
        with self.assertRaises(TypeError):
            PureSet([1,2,3], ser)
        df = pd.DataFrame({'a':[1,2],'b':[3,4]})
        with self.assertRaises(TypeError):
            PureSet(df, [[1,2], [3,4]])

class TestPureSetWeirdPythonNativeTypes(unittest.TestCase):
    def test_mix_tuple_namedtuple_list(self):
        Pt = namedtuple("Pt", "x y")
        t = (1,2)
        nt = Pt(1,2)
        l = [1,2]
        with self.assertRaises(TypeError):
            PureSet(nt, t)
        with self.assertRaises(TypeError):
            PureSet(t, l)
        with self.assertRaises(TypeError):
            PureSet(nt, l)
    def test_zero_length_types(self):
        self.assertEqual(len(PureSet()), 0)
        self.assertEqual(len(PureSet([])), 1)
        self.assertEqual(len(PureSet(())), 1)
        self.assertEqual(len(PureSet({})), 1)

##### === MAIN ENTRYPOINT === #####
if __name__ == "__main__":
    unittest.main()
