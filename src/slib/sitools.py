from collections import OrderedDict, defaultdict
from itertools import count, islice, repeat

from slib import stypes
from slib import sfunctions


def chunks(csize, seq):
    for i in range(0, len(seq), csize):
        yield seq[i: i + csize]


def icount(f, iterable):
    count_ = 0
    for item in iterable:
        if f(item):
            count_ += 1
    return count_


def dedupe(key_extractor,
           deduplicator,
           iterable):
    cache = OrderedDict()
    for item in iterable:
        key = key_extractor(item)
        if key in cache:
            cache[key] = deduplicator(cache[key], item)
        else:
            cache[key] = item
    return cache.values()


def idiff(pos, neg):
    for item in pos:
        if item not in neg:
            yield item


def filter_not(f, iterable):
    return filter(lambda x: not f(x), iterable)


def fold_left(f, init, iterable):
    res = init
    for item in iterable:
        res = f(res, item)
    return res


def flatten(iterable):
    for item in iterable:
        if stypes.isiterable(item):
            yield from flatten(item)
        else:
            yield item


def islice_unique(iterable, n, key_extractor):
    return islice(distinct(iterable, key_extractor), n)


def zip_with_index(iterable):
    return zip(iterable, count())


def reduce_left(f, iterable):
    it = iter(iterable)
    res = next(it)
    for item in it:
        res = f(res, item)
    return res


def replicate(tpe, n):
    for _ in repeat(None, n):
        yield tpe()


class distinct:
    def __init__(self, iterable, key_extractor = sfunctions.id):
        self._it = iter(iterable)
        self._key_extractor = key_extractor
        self._visited = set()

    def __next__(self):
        item = next(self._it)
        key = self._key_extractor(item)
        while key in self._visited:
            item = next(self._it)
            key = self._key_extractor(item)
        self._visited.add(key)
        return item



class group_by:
    def __init__(self, keyx, iterable):
        groups = defaultdict(list)
        for item in iterable:
            groups[keyx(item)].append(item)
        self._groups = groups

    def __getitem__(self, k):
        return self._groups[k]

    def __len__(self):
        return len(self._groups)

    def __iter__(self):
        return iter(self._groups.items())


def group_indices_by(key_extractor, iterable):
    groups = defaultdict(list)
    for idx, item in enumerate(iterable):
        groups[key_extractor(item)].append(idx)
    return groups


def split(f, iterable):
    leftc = list()
    rightc = list()
    for item in iterable:
        left, right = f(item)
        leftc.append(left)
        rightc.append(right)
    return leftc, rightc


def get_all(idxs, seq):
    return [seq[idx] for idx in idxs]


if __name__ == '__main__':
    l1 = [('a', 4), ('b', 4), ('a', 7), ('c', 10)]
    d = dedupe(lambda t: t[0], lambda t1, t2: t1 if t1[1] > t2[1] else t2, l1)
    print(d)
