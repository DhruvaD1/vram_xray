from vramxray.frag import explain, holes_in, if_freed
from vramxray.snapshot import ACTIVE, INACTIVE, MiB, from_dict


def snap(segments, traces=(), device=0):
    """segments: list of (total_size, stream, [(size, state, frames)...]) laid out contiguously."""
    raw = {"segments": [], "device_traces": [list(traces)]}
    addr = 0x1000_0000
    for total, stream, blocks in segments:
        seg = {
            "address": addr,
            "total_size": total,
            "stream": stream,
            "device": device,
            "segment_type": "large",
            "blocks": [],
        }
        b_addr = addr
        for size, state, frames in blocks:
            seg["blocks"].append(
                {
                    "address": b_addr,
                    "size": size,
                    "requested_size": size,
                    "state": state,
                    "frames": [{"filename": f, "line": 1, "name": "f"} for f in frames],
                }
            )
            b_addr += size
        raw["segments"].append(seg)
        addr += total + 0x1000_0000
    return from_dict(raw)


def test_split_remainder_layout():
    s = snap(
        [
            (
                512 * MiB,
                0,
                [(500 * MiB, INACTIVE, []), (4 * MiB, ACTIVE, ["pin.py"]), (8 * MiB, INACTIVE, [])],
            )
        ]
    )
    holes = holes_in(s.segments[0])
    assert [h.size for h in holes] == [500 * MiB, 8 * MiB]
    assert holes[0].left is None and holes[0].right.size == 4 * MiB
    ex = explain(s, request=505 * MiB, stream=0)
    assert ex.verdict == "fragmentation"
    assert (
        ex.free_in_segments == 508 * MiB and ex.largest_hole == 500 * MiB and ex.trapped == 8 * MiB
    )
    assert (
        len(ex.pins) == 1
        and ex.pins[0].wasted == 508 * MiB
        and ex.pins[0].frames[0].filename == "pin.py"
    )
    assert if_freed(ex.holes, [ex.pins[0].block]) == 512 * MiB


def test_request_fits_is_not_fragmentation():
    s = snap([(512 * MiB, 0, [(500 * MiB, INACTIVE, []), (12 * MiB, ACTIVE, [])])])
    assert explain(s, request=100 * MiB, stream=0).verdict == "unknown"


def test_exhaustion_when_nothing_would_fit():
    s = snap([(64 * MiB, 0, [(60 * MiB, ACTIVE, []), (4 * MiB, INACTIVE, [])])])
    ex = explain(s, request=1024 * MiB, stream=0, device_free=10 * MiB)
    assert ex.verdict == "exhaustion"


def test_other_stream_free_is_reported():
    s = snap(
        [(1024 * MiB, 7, [(1024 * MiB, INACTIVE, [])]), (64 * MiB, 0, [(64 * MiB, ACTIVE, [])])]
    )
    ex = explain(s, request=512 * MiB, stream=0, device_free=0)
    assert ex.verdict == "stream"


def test_small_pool_segments_ignored_for_large_requests():
    raw_small = snap([(2 * MiB, 0, [(2 * MiB, INACTIVE, [])])])
    raw_small.segments[0].segment_type = "small"
    ex = explain(raw_small, request=4 * MiB, stream=0, device_free=0)
    assert ex.free_in_segments == 0 and ex.verdict == "exhaustion"


def test_oom_entry_drives_request_and_age():
    traces = [
        {"action": "alloc", "addr": 0x1000_0000, "size": 500 * MiB, "stream": 0, "time_us": 1},
        {
            "action": "alloc",
            "addr": 0x1000_0000 + 500 * MiB,
            "size": 4 * MiB,
            "stream": 0,
            "time_us": 2,
        },
        {"action": "alloc", "addr": 0x9999_0000, "size": 1 * MiB, "stream": 0, "time_us": 3},
        {"action": "oom", "addr": 3 * MiB, "size": 505 * MiB, "stream": 0, "time_us": 4},
    ]
    s = snap(
        [
            (
                512 * MiB,
                0,
                [(500 * MiB, INACTIVE, []), (4 * MiB, ACTIVE, ["pin.py"]), (8 * MiB, INACTIVE, [])],
            )
        ],
        traces,
    )
    ex = explain(s)
    assert ex.request == 505 * MiB and ex.device_free == 3 * MiB and ex.verdict == "fragmentation"
    assert ex.pins[0].age == 1
