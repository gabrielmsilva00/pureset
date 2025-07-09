"""
perf_pureset_expanded.py: Async & Parallel Performance Benchmarks for PureSet, PureMap vs. Python built-in types.
"""

import sys
import asyncio
import concurrent.futures
import contextlib
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
import pureset
from pureset import PureSet, PureMap

VERSION: str = getattr(pureset, "__version__", "unknown")
LOG_DIR: Path = Path(r'log')
LOG_DIR.mkdir(exist_ok=True)
N: int = 2 ** 16

def make_logfile_name(version: str) -> Path:
  stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
  return LOG_DIR / f'perf_pureset-{version}-{stamp}-perf.log'

def perc_overhead(slower: float, faster: float) -> str:
  if faster == 0.0:
    return "(undefined)"
  if slower >= faster:
    return f"({((slower / faster - 1) * 100):.1f}% overhead)"
  else:
    return f"({((1 - slower / faster) * 100):.1f}% faster)"

def bench_sync(
  stmt: str,
  setup: str,
  globals_: dict[str, Any],
  repeat: int = 4,
  number: int = 4
) -> float:
  from timeit import timeit
  times = [timeit(stmt, setup=setup, globals=globals_, number=number)
           for _ in range(repeat)]
  return sum(times) / (repeat * number) if repeat and number else 0.0

async def bench(
  desc: str,
  stmt_py: str,
  stmt_pu: str,
  setup_py: dict[str, Any],
  setup_pu: dict[str, Any],
  *,
  repeat: int = 4,
  number: int = 4
) -> str:
  loop = asyncio.get_running_loop()
  with concurrent.futures.ThreadPoolExecutor() as pool:
    t_py, t_pu = await asyncio.gather(
      loop.run_in_executor(pool, bench_sync, stmt_py, 'pass', setup_py, repeat, number),
      loop.run_in_executor(pool, bench_sync, stmt_pu, 'pass', setup_pu, repeat, number)
    )
  rel = perc_overhead(t_pu, t_py)
  return f"{desc}\n  Native: {t_py:10.7f} s\n  Pure:   {t_pu:10.7f} s {rel}\n"

async def bench_op(
  desc: str,
  stmt_py: str,
  stmt_pu: str,
  setup_py: dict[str, Any],
  setup_pu: dict[str, Any],
  *,
  repeat: int = 4,
  number: int = 4
) -> str:
  try:
    return await bench(desc, stmt_py, stmt_pu, setup_py, setup_pu, repeat=repeat, number=number)
  except Exception as exc:
    return f"{desc}\n  [skipped] {exc}\n"

async def bench_map_method(
  meth: str,
  py_dict: dict[str, int],
  pu_map: 'PureMap',
  *,
  repeat: int = 4,
  number: int = 4
) -> str:
  desc: str = f"Map Method '{meth}'"
  py_attr = getattr(py_dict, meth, None)
  pu_attr = getattr(pu_map, meth, None)
  if py_attr is None or pu_attr is None:
    return f"{desc}\n  Not supported by one or both implementations.\n"
  try:
    py_result = py_attr() if callable(py_attr) else py_attr
    pu_result = pu_attr() if callable(pu_attr) else pu_attr
    list(py_result)
    list(pu_result)
  except Exception as exc:
    return f"{desc}\n  Not supported ({exc})\n"
  py_stmt = f"list(obj.{meth}())" if callable(py_attr) else f"list(obj.{meth})"
  pu_stmt = f"list(obj.{meth}())" if callable(pu_attr) else f"list(obj.{meth})"
  return await bench_op(
    desc, py_stmt, pu_stmt, {'obj': py_dict}, {'obj': pu_map}, repeat=repeat, number=number
  )

async def bench_eq_cmp(
  op: str,
  py_l: Any, py_r: Any,
  pu_l: Any, pu_r: Any,
  label: str,
  *,
  repeat: int = 4,
  number: int = 4
) -> str:
  stmt_py = f"l {op} r"
  stmt_pu = f"l {op} r"
  return await bench_op(
    f"{label} ({op})", stmt_py, stmt_pu,
    {'l': py_l, 'r': py_r}, {'l': pu_l, 'r': pu_r},
    repeat=repeat, number=number
  )

async def bench_mutating(
  method_name: str,
  py_obj_factory: Callable[[], Any],
  pu_obj_factory: Callable[[], Any],
  arg: Any,
  supports_pure: bool = True,
  *,
  repeat: int = 4,
  number: int = 4
) -> str:
  desc = f"Mutating Method '{method_name}'"
  if not supports_pure:
    return f"{desc}\n  Not supported by Pure impl.\n"
  py_stmt = f"obj.{method}({('arg',)[0]})" if arg is not None else f"obj.{method}()"
  pu_stmt = py_stmt
  def safe_py(): obj = py_obj_factory(); getattr(obj, method_name)(arg)
  def safe_pu(): obj = pu_obj_factory(); getattr(obj, method_name)(arg)
  return await bench_op(
    desc,
    f"fn()",
    f"fn()",
    {'fn': safe_py}, {'fn': safe_pu},
    repeat=repeat, number=number
  )

def _safe_lookup(d: Mapping[str, int], k: str) -> Optional[int]:
  try: return d[k]
  except KeyError: return None

def _profile_memory(obj: Any) -> int:
  try:
    import tracemalloc
    tracemalloc.start()
    # Perform simple op to force allocation
    _ = list(obj) if hasattr(obj, "__iter__") else obj
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak
  except ImportError:
    return -1

async def run_suite_async() -> str:
  keys: list[str] = [str(i) for i in range(N)]
  vals: list[int] = list(range(N))
  pairs: list[tuple[str, int]] = list(zip(keys, vals))
  keysB: list[str] = [str(i) for i in range(N // 2, N + N // 2)]
  pairsB: list[tuple[str, int]] = list(zip(keysB, vals[:len(keysB)]))
  psetA, psetB = PureSet(keys), PureSet(keysB)
  py_setA, py_setB = set(keys), set(keysB)
  py_frozensetA, py_frozensetB = frozenset(keys), frozenset(keysB)
  py_tupleA, py_tupleB = tuple(keys), tuple(keysB)
  pmapA, pmapB = PureMap(pairs), PureMap(pairsB)
  py_dictA, py_dictB = dict(pairs), dict(pairsB)
  ix_hit: str = keys[N // 4]
  ix_miss: str = "not-present"
  ix_slice: tuple[int, int] = (N // 4, N // 4 + 8)

  constr_tasks = [
    bench_op(
      "Set Construction", "ctor()", "ctor()",
      {'ctor': lambda: set(keys)}, {'ctor': lambda: PureSet(keys)}),
    bench_op(
      "FrozenSet Construction", "ctor()", "ctor()",
      {'ctor': lambda: frozenset(keys)}, {'ctor': lambda: PureSet(keys)}),
    bench_op(
      "Tuple Construction", "ctor()", "ctor()",
      {'ctor': lambda: tuple(keys)}, {'ctor': lambda: PureSet(keys)}),
    bench_op(
      "Map Construction", "ctor()", "ctor()",
      {'ctor': lambda: dict(pairs)}, {'ctor': lambda: PureMap(pairs)}),
    bench_op(
      "Large Set Construction (2x)", "ctor()", "ctor()",
      {'ctor': lambda: set(keys + keysB)}, {'ctor': lambda: PureSet(keys + keysB)}
    ),
    bench_op(
      "Large Map Construction (2x)", "ctor()", "ctor()",
      {'ctor': lambda: dict(pairs + pairsB)}, {'ctor': lambda: PureMap(pairs + pairsB)}
    )
  ]
  memb_tasks = [
    bench_op(
      "Set Membership (hit)", "item in obj", "item in obj",
      {'obj': py_setA, 'item': ix_hit}, {'obj': psetA, 'item': ix_hit}),
    bench_op(
      "Set Membership (miss)", "item in obj", "item in obj",
      {'obj': py_setA, 'item': ix_miss}, {'obj': psetA, 'item': ix_miss}),
    bench_op(
      "FrozenSet Membership (hit)", "item in obj", "item in obj",
      {'obj': py_frozensetA, 'item': ix_hit}, {'obj': psetA, 'item': ix_hit}),
    bench_op(
      "FrozenSet Membership (miss)", "item in obj", "item in obj",
      {'obj': py_frozensetA, 'item': ix_miss}, {'obj': psetA, 'item': ix_miss}),
    bench_op(
      "Tuple Membership (hit)", "item in obj", "item in obj",
      {'obj': py_tupleA, 'item': ix_hit}, {'obj': psetA, 'item': ix_hit}),
    bench_op(
      "Tuple Membership (miss)", "item in obj", "item in obj",
      {'obj': py_tupleA, 'item': ix_miss}, {'obj': psetA, 'item': ix_miss}),
  ]
  map_tasks = [
    bench_op(
      "Map Lookup (hit)", "structure[key]", "structure[key]",
      {'structure': py_dictA, 'key': ix_hit}, {'structure': pmapA, 'key': ix_hit}),
    bench_op(
      "Map Lookup (miss)", "safe_lookup(structure, key)", "safe_lookup(structure, key)",
      {'structure': py_dictA, 'key': ix_miss, 'safe_lookup': _safe_lookup},
      {'structure': pmapA, 'key': ix_miss, 'safe_lookup': _safe_lookup})
  ]
  slice_tasks = [
    bench_op(
      "Tuple Slicing", "obj[start:stop]", "obj[start:stop]",
      {'obj': py_tupleA, 'start': ix_slice[0], 'stop': ix_slice[1]},
      {'obj': psetA, 'start': ix_slice[0], 'stop': ix_slice[1]})
  ]
  conv_tasks = [
    bench_op(
      "Set to List", "list(obj)", "list(obj)",
      {'obj': py_setA}, {'obj': psetA}),
    bench_op(
      "Map to Items List", "list(obj.items())", "list(obj.items)",
      {'obj': py_dictA}, {'obj': pmapA}),
  ]
  op_cases = [
    ("Set Union", "|", py_setA, py_setB, psetA, psetB),
    ("Set Intersection", "&", py_setA, py_setB, psetA, psetB),
    ("Set Difference", "-", py_setA, py_setB, psetA, psetB),
    ("Set SymDiff", "^", py_setA, py_setB, psetA, psetB),
    ("FrozenSet Union", "|", py_frozensetA, py_frozensetB, psetA, psetB),
    ("FrozenSet Intersection", "&", py_frozensetA, py_frozensetB, psetA, psetB)
  ]
  op_tasks = [
    bench_op(label, f"l {op} r", f"l {op} r", {'l': py_l, 'r': py_r}, {'l': pu_l, 'r': pu_r})
    for label, op, py_l, py_r, pu_l, pu_r in op_cases
  ]
  eqcmp_ops = [
    ("==", "Set Equality", py_setA, py_setB, psetA, psetB),
    ("!=", "Set Inequality", py_setA, py_setB, psetA, psetB),
    ("<", "Set Less-Than", py_setA, py_setB, psetA, psetB),
    (">", "Set Greater-Than", py_setA, py_setB, psetA, psetB),
    ("<=", "Set LTE", py_setA, py_setB, psetA, psetB),
    (">=", "Set GTE", py_setA, py_setB, psetA, psetB),
    ("==", "Map Equality", py_dictA, py_dictB, pmapA, pmapB),
    ("!=", "Map Inequality", py_dictA, py_dictB, pmapA, pmapB)
  ]
  eqcmp_tasks = [
    bench_eq_cmp(op, py_l, py_r, pu_l, pu_r, label)
    for op, label, py_l, py_r, pu_l, pu_r in eqcmp_ops
  ]
  map_method_tasks = [
    bench_map_method(meth, py_dictA, pmapA)
    for meth in ("keys", "values", "items")
  ]
  edge_tasks = [
    bench_op(
      "Empty Set Construction", "ctor()", "ctor()",
      {'ctor': lambda: set([])}, {'ctor': lambda: PureSet([])}),
    bench_op(
      "Singleton Set Construction", "ctor()", "ctor()",
      {'ctor': lambda: set(['x'])}, {'ctor': lambda: PureSet(['x'])}),
    bench_op(
      "Empty Map Construction", "ctor()", "ctor()",
      {'ctor': lambda: dict([])}, {'ctor': lambda: PureMap([])}),
    bench_op(
      "Singleton Map Construction", "ctor()", "ctor()",
      {'ctor': lambda: dict([('x', 1)])}, {'ctor': lambda: PureMap([('x', 1)])})
  ]
  memory_tasks = [
    bench_op(
      "Set Memory Profile", "mem_prof(obj)", "mem_prof(obj)",
      {'mem_prof': _profile_memory, 'obj': set(keys)},
      {'mem_prof': _profile_memory, 'obj': PureSet(keys)}
    ),
    bench_op(
      "Map Memory Profile", "mem_prof(obj)", "mem_prof(obj)",
      {'mem_prof': _profile_memory, 'obj': dict(pairs)},
      {'mem_prof': _profile_memory, 'obj': PureMap(pairs)}
    )
  ]
  parallel_tasks = [
    bench_op(
      "Parallel Set Construction",
      "list(executor.map(ctor, [keys]*4))",
      "list(executor.map(ctor, [keys]*4))",
      {'executor': concurrent.futures.ThreadPoolExecutor(), 'ctor': lambda x: set(x), 'keys': keys},
      {'executor': concurrent.futures.ThreadPoolExecutor(), 'ctor': lambda x: PureSet(x), 'keys': keys}
    )
  ]
  all_sections = [
    ("-- Construction --", constr_tasks),
    ("\n-- Membership --", memb_tasks),
    ("\n-- Mapping Lookup --", map_tasks),
    ("\n-- Slicing --", slice_tasks),
    ("\n-- Conversion --", conv_tasks),
    ("\n-- Set Operations --", op_tasks),
    ("\n-- Equality/Comparisons --", eqcmp_tasks),
    ("\n-- Map Methods --", map_method_tasks),
    ("\n-- Edge Cases --", edge_tasks),
    ("\n-- Memory Profile --", memory_tasks),
    ("\n-- Parallel Construction --", parallel_tasks)
  ]

  progress_total = sum(len(tgrp) for _, tgrp in all_sections)
  progress_done = 0

  async def run_and_collect(section_title: str, task_list: list[asyncio.Task]) -> str:
    nonlocal progress_done
    lines = [section_title]
    res = await asyncio.gather(*task_list)
    for chunk in res:
      lines.append(chunk.rstrip())
      progress_done += 1
      print(f"[{progress_done}/{progress_total}] Completed: {chunk.splitlines()[0]}")
    return "\n".join(lines)

  t0 = time.perf_counter()
  section_results = await asyncio.gather(*[run_and_collect(section, tasks) for section, tasks in all_sections])
  duration = time.perf_counter() - t0
  lines = []
  lines.append(
    f'perf_pureset.py PERFORMANCE SUITE {datetime.now().isoformat()}\nVersion: {VERSION} | N={N}\n'
  )
  lines.extend(section_results)
  lines.append(f'\n--- END OF SUITE IN {duration:.2f}s ---\n')
  return "\n".join(lines)

def log_perf_atomic(result: str, version: str) -> None:
  with open(make_logfile_name(version), 'w', encoding='utf-8') as f:
    f.write(result)

def main() -> None:
  import asyncio
  print(f"perf_pureset.py PERFORMANCE SUITE {datetime.now().isoformat()}\nVersion: {VERSION} | N={N}\n")
  print("--- START OF SUITE ---")
  res = asyncio.run(run_suite_async())
  print(res)
  log_perf_atomic(res, VERSION)

if __name__ == "__main__":
  main()