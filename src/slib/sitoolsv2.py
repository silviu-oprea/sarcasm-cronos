from collections import defaultdict


def groupby(key_extractor, it):
    groups = defaultdict(list)
    for val in it:
        groups[key_extractor(val)].append(val)
    for key, val in groups.items():
        yield key, val
