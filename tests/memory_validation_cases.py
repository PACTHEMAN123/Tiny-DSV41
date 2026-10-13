"""Shared fixture dimensions and topology cases for distributed acceptance."""


def fixture_dimensions(world):
    if world == 8:
        return 8, 16
    return 6, 12


def validation_cases(world, all_topologies=False):
    experts, length = fixture_dimensions(world)
    if all_topologies:
        return [(cp, ep, "direct_shard", "cpu", "async", True, False)
                for cp in range(1, world + 1) for ep in range(1, world + 1)
                if world % cp == world % ep == 0 and length % cp == experts % ep == 0]
    if world == 8:
        return [(1, 1, "layerwise", "cuda", "sync", False, False),
                (1, 8, "direct_shard", "cuda", "sync", True, False),
                (2, 4, "direct_shard", "cpu", "sync", False, True),
                (4, 2, "direct_shard", "cpu", "async", True, True),
                (8, 1, "layerwise", "cpu", "async", True, False),
                (8, 8, "direct_shard", "cpu", "async", True, True)]
    if world == 6:
        return [(2, 3, "direct_shard", "cpu", "async", True, True),
                (3, 2, "layerwise", "cuda", "sync", False, False)]
    return [(1, 1, "layerwise", "cuda", "sync", False, False),
            (1, 2, "direct_shard", "cuda", "sync", True, False),
            (2, 1, "direct_shard", "cpu", "sync", False, True),
            (2, 2, "direct_shard", "cpu", "async", True, False),
            (1, 2, "layerwise", "cpu", "async", True, True)]
