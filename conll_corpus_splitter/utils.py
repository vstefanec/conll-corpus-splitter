from collections import OrderedDict, namedtuple

from enum import IntEnum


class Dataset(IntEnum):
    TRAIN = 0
    DEV = 1
    TEST = 2

    def __repr__(self):
        return self.name


class RotatingList(list):
    def __getitem__(self, index):
        if isinstance(index, int):
            index %= len(self)
            return super().__getitem__(index)

        elif isinstance(index, slice):
            start = index.start
            stop = index.stop
            step = index.step

            if start is None and stop is None:
                return super().__getitem__(index)

            if start is None:
                start = 0
            start %= len(self)

            if stop is None:
                stop = len(self)

            if step is None:
                step = 1

            stop %= len(self)

            if step > 0 and stop < start:
                return super().__getitem__(slice(start, len(self), step)) + super().__getitem__(slice(0, stop, step))

            elif step < 0 and stop > start:
                return super().__getitem__(slice(stop, 0, step)) + super().__getitem__(slice(len(self), start, step))

            return super().__getitem__(index)

        else:
            return super().__getitem__(index)


class MetadataValue(namedtuple('MetadataValue', ['value', 'text', 'line_no'])):
    pass


class MetadataDiffDict(OrderedDict):
    def diff_and_update(self, updating_dict):
        diff_items = sorted(set(updating_dict.items()) - set(self.items()), key=lambda v: v[1].line_no)
        diff = self.__class__(diff_items)
        self.update(updating_dict)
        return diff
