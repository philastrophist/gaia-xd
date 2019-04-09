import warnings
from collections import OrderedDict, namedtuple
from functools import partial
from itertools import cycle
import numpy as np


__all__ = ['Backend']

class BackendBase(object):
    def __init__(self, name):
        self.name = name
        self.arrays = {}
        self._iteration = 0

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value

    @property
    def varnames(self):
        return list(self.arrays.keys())

    @property
    def is_setup(self):
        return len(self.arrays) > 0

    @property
    def _length(self):
        return self.arrays[list(self.arrays.keys())[0]].shape[0]


    def setup(self, length, **varvalues):
        assert not self.is_setup, "Backend already setup"
        for k, v in varvalues.items():
            v = np.asarray(v)
            self.arrays[k] = np.zeros((length, ) + v.shape, dtype=v.dtype)


    def grow(self, length):
        space_left = self._length - self.iteration
        space_needed = length - space_left
        if space_needed > 0:
            for k, array in self.arrays.items():
                self.arrays[k] = np.append(array, np.zeros((space_needed,) + array.shape[1:], dtype=array.dtype), axis=0)


    def crop(self):
        for k, array in self.arrays.items():
            self.arrays[k] = self.arrays[k][:self.iteration]


    def save(self, **varvalues):
        if self.iteration >= self._length:
            self.grow(1)
            warnings.warn("Backend is full, now expanding backend 1 step at a time. "
                          "This is really inefficient. Use backend.grow(x) to make more space")
        for k, v in varvalues.items():
            self.arrays[k][self.iteration] = np.asarray(v)
        self.iteration += 1


    def __getitem__(self, item):
        if isinstance(item, int):
            return {v: self.get_values(v, item) for v in self.varnames}

        if isinstance(item, str):
            return self.get_values(item)

        if isinstance(item, tuple):
            return self.get_values(item[0], item[1:])

        if isinstance(item, slice):
            return {v: self.get_values(v, item) for v in self.varnames}


    def get_values(self, varname, index=None):
        variable = self.arrays[varname]
        if index is None:
            index = slice(0, self.iteration)
        if isinstance(index, slice):
            if index.stop > self.iteration:
                raise IndexError("Cannot get slices {} above maximum iteration {}".format(index, self.iteration))
        if isinstance(index, tuple):
            if index[0] < 0:
                index = (self.iteration + index[0],) + index[1:]
        elif isinstance(index, int):
            if index < 0:
                index += self.iteration
        return variable[index]


    def __len__(self):
        return self.iteration


    def __repr__(self):
        return "<XDGMM Backend - {} iterations ({})>".format(self.iteration, self.varnames)


    def __getattr__(self, item):
        if item == 'is_setup':
            return self.is_setup
        return self.arrays[item][:self.iteration]


class HDFBackendBase(BackendBase):
    def __init__(self, name, fname):
        self.fname = fname
        super().__init__(name)
        del self._iteration

    @property
    def iteration(self):
        try:
            with self.open(mode='r') as f:
                return f[self.name].attrs['iteration']
        except (OSError, KeyError) as e:
            return 0

    @iteration.setter
    def iteration(self, value):
        with self.open(mode='a') as f:
            f[self.name].attrs['iteration'] = int(value)

    @property
    def varnames(self):
        with self.open(mode='r') as f:
            return list(f[self.name].keys())

    @property
    def is_setup(self):
        try:
            return len(self.varnames) > 0
        except (OSError, KeyError) as e:
            return False

    @property
    def _length(self):
        with self.open(mode='r') as f:
            group = f[self.name]
            return group[list(group.keys())[0]].shape[0]

    def open(self, **kwargs):
        import h5py
        return h5py.File(self.fname, **kwargs)


    def setup(self, length, **varvalues):
        assert not self.is_setup, "Backend already setup"
        with self.open(mode='a')  as f:
            group = f.require_group(self.name)
            for k, v in varvalues.items():
                v = np.asarray(v)
                group.create_dataset(k, shape=(length, )+v.shape, dtype=v.dtype, maxshape=(None,)+v.shape)


    def grow(self, length):
        space_left = self._length - self.iteration
        space_needed = length - space_left
        if space_needed > 0:
            with self.open(mode='r+') as f:
                group = f[self.name]
                for k, ds in group.items():
                    ds.resize(ds.shape[0] + space_needed, axis=0)

    def crop(self):
        with self.open(mode='r+') as f:
            group = f[self.name]
            for k, ds in group.items():
                ds.resize(self.iteration, axis=0)


    def save(self, **varvalues):
        if self.iteration >= self._length:
            self.grow(1)
            warnings.warn("Backend is full, now expanding backend 1 step at a time. "
                          "This is really inefficient. Use backend.grow(x) to make more space")
        for k, v in varvalues.items():
            with self.open(mode='r+') as f:
                f[self.name][k][self.iteration] = np.asarray(v)
        self.iteration += 1


    def get_values(self, varname, index=None):
        with self.open(mode='r') as f:
            self.arrays = f[self.name]
            return super(HDFBackendBase, self).get_values(varname, index=index)

    def __getattr__(self, item):
        if item == 'is_setup':
            return self.is_setup
        with self.open(mode='r') as f:
            self.arrays = f[self.name]
            return self.arrays[item][:self.iteration]

    def __getitem__(self, item):
        with self.open(mode='r') as f:
            self.arrays = f[self.name]
            return super(HDFBackendBase, self).__getitem__(item)

Event = namedtuple('Event', ['event', 'origin', 'origin_iteration', 'into', 'into_iteration'])

class MultiBackend(object):
    def __init__(self, name, backend_type):
        self.name = name
        self.backend_type = backend_type
        self.store = OrderedDict(master=backend_type('master'))
        self._events = []
        self.current_name = 'master'

    @property
    def varnames(self):
        return self.store[list(self.store.keys())[0]].varnames

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, value):
        pass

    def print_events(self):
        for event in self.events:
            print("{}[{}] -{}-> {}[{}]".format(event.origin, event.origin_iteration, event.event, event.into, event.into_iteration))

    def draw_tree(self):
        import matplotlib.pyplot as plt
        import networkx as nx
        ypos = {k: -i for i, k in enumerate(self.store.keys())}

        G = nx.MultiDiGraph()
        G.add_node('master\n[0]', pos=[0, 0])
        for event in self.events:
            if event.event != 'switch':
                old = "{}\n[{}]".format(event.origin, event.origin_iteration)
                new = "{}\n[{}]".format(event.into, event.into_iteration)

                if old not in G:
                    chain = [int(i.split('[')[1][:-1]) for i in G.nodes.keys() if event.origin in i]
                    previous = "{}\n[{}]".format(event.origin, max(chain))
                    G.add_edge(previous, old)
                    xpos = G.nodes[previous]['pos'][0] + 1
                    G.nodes[old]['pos'] = [xpos, ypos[event.origin]]
                G.add_edge(old, new)
                xpos = G.nodes[old]['pos'][0] + 1
                G.nodes[new]['pos'] = [xpos, ypos[event.into]]
            else:
                chain = [int(i.split('[')[1][:-1]) for i in G.nodes.keys() if event.into in i]
                previous = "{}\n[{}]".format(event.into, max(chain))
                new = "{}\n[{}]".format(event.into, event.into_iteration)
                G.add_edge(previous, new)
                xpos = G.nodes[previous]['pos'][0] + 1
                G.nodes[new]['pos'] = [xpos, ypos[event.into]]

        pos = {k: v['pos'] for k, v in G.nodes.items()}
        nx.draw(G, pos, with_labels=True)

    @property
    def current(self):
        return self.store[self.current_name]

    @property
    def master(self):
        return self.store['master']

    def grow(self, length):
        return self.current.grow(length)

    def crop(self):
        return self.current.crop()

    def get_values(self, varname, index=None):
        return self.current.get_values(varname, index)

    def __getattr__(self, item):
        return self.current.__getattr__(item)

    def __getitem__(self, item):
        return self.current.__getitem__(item)

    def setup(self, length, **varvalues):
        self.master.setup(length, **varvalues)

    def save(self, **varvalues):
        self.current.save(**varvalues)

    def branch_chain(self, length, name):
        assert name not in self.store
        new = self.backend_type(name)
        new.setup(length, **self.master[0])
        self.store[name] = new
        self.events.append(Event('branch', self.current_name, self.store[self.current_name].iteration, name, 0))
        self.switch_chain(name)

    def merge_chain(self, into):
        self.store[into].grow(len(self.current))
        new_length = len(self.store[into])
        for i in range(len(self.current)):
            self.store[into].save(**self.current[i])
        self.events.append(Event('merge', self.current_name, self.store[self.current_name].iteration, into, new_length))
        self.switch_chain(into)

    def switch_chain(self, name):
        self.events.append(Event('switch', self.current_name, self.store[self.current_name].iteration, name, self.store[name].iteration))
        self.current_name = name

    @property
    def iteration(self):
        return self.current.iteration

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return "<GMMBackend - {} iterations - {} chains (current={}) ({})>".format(self.iteration, len(self.store), self.current_name, self.varnames)

    @property
    def is_setup(self):
        if len(self.store):
            return all(chain.is_setup for chain in self.store)
        return False


def Backend(name, fname=None):
    if fname is None:
        return MultiBackend(name, BackendBase)
    return MultiBackend(name, partial(HDFBackendBase, fname=fname))


if __name__ == '__main__':
    a = np.ones(3)
    b = np.eye(5)
    c = 1

    store = Backend('backend', 'test.h5')
    store.setup(10, a=a, b=b, c=c)
    store.save(a=a, b=b, c=c)
    store.save(a=a, b=b, c=c)

    store.branch_chain(2, 'event-1')
    store.save(a=a, b=b, c=c)
    store.save(a=a*2, b=b*2, c=c*2)
    store.merge_chain('master')

    store.branch_chain(1, 'event-2')
    store.save(a=a, b=b, c=c)

    store.switch_chain('master')
    store.grow(1)
    store.save(a=a, b=b, c=c)

    import matplotlib.pyplot as plt
    store.draw_tree()
    store.print_events()
    plt.show()