"""Event columns from the native core, convertible to whatever table library you have."""

from __future__ import annotations

# torch's TraceEntry actions by number, then our driver-side kinds
ACTIONS = {
    0: "alloc",
    1: "free_requested",
    2: "free_completed",
    3: "segment_alloc",
    4: "segment_free",
    5: "segment_map",
    6: "segment_unmap",
    7: "snapshot",
    8: "oom",
    9: "annotate",
    100: "driver_alloc",
    101: "driver_free",
    102: "driver_create",
    103: "driver_release",
    104: "module_load",
}


class Timeline:
    def __init__(self, columns: dict[str, list]):
        self.columns = dict(columns)
        self.columns["action"] = [ACTIONS.get(a, str(a)) for a in self.columns.get("action", [])]

    def __len__(self) -> int:
        return len(self.columns.get("ts", []))

    def to_dict(self) -> dict[str, list]:
        return self.columns

    def to_polars(self):
        import polars as pl

        return pl.DataFrame(self.columns)

    def to_arrow(self):
        import pyarrow as pa

        return pa.table(self.columns)

    def to_pandas(self):
        import pandas as pd

        return pd.DataFrame(self.columns)
