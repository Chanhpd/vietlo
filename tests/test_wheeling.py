"""
Unit tests for Combinatorial Wheeling System.
"""

from src.wheeling import CombinatorialWheeler


def test_combinatorial_wheeler():
    wheeler = CombinatorialWheeler(pick_count=6)
    pool = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]  # 10 numbers
    res = wheeler.generate_abbreviated_wheel(pool, target_guarantee=3, condition_drawn=4, max_tickets=6)

    assert res["pool_size"] == 10
    assert 1 <= res["ticket_count"] <= 6
    assert res["guarantee_coverage_pct"] >= 95.0

    for ticket in res["tickets"]:
        assert len(ticket) == 6
        assert len(set(ticket)) == 6
        assert all(n in pool for n in ticket)

