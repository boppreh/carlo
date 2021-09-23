import multiprocessing
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from collections import namedtuple
import itertools
import queue

Snapshot = namedtuple('Snapshot', 'n min max mean bins_width pdf')

class Digest:
    def __init__(self, n_bins, seed_value=0):
        self.results = []
        self.min_found, self.max_found = float('inf'), float('-inf')
        self.mean = None
        self.n = 0
        self.bins_start = seed_value
        self.bins_end = seed_value * 1.1
        self.n_bins = n_bins
        self.bins = [1] + [0] * (self.n_bins - 1)

    @property
    def bins_width(self):
        return (self.bins_end - self.bins_start) / len(self.bins)
    
    def update(self, value):
        self.n += 1
        self.min_found = min(self.min_found, value)
        self.max_found = max(self.max_found, value)
        self.mean = value if self.mean is None else (self.mean * self.n + value) / (self.n + 1)

        while value > self.bins_end:
            self.bins_end = self.bins_end + (self.bins_end - self.bins_start)
            self.bins = [self.bins[i] + self.bins[i+1] for i in range(0, len(self.bins), 2)] + len(self.bins)//2 * [0]
        while value < self.bins_start:
            self.bins_start = self.bins_start - (self.bins_end - self.bins_start)
            self.bins = len(self.bins)//2 * [0] + [self.bins[i] + self.bins[i+1] for i in range(0, len(self.bins), 2)]
        bin_i = int((value - self.bins_start) / self.bins_width)
        assert bin_i >= 0
        self.bins[bin_i] += 1

    def get_snapshot(self):
        return Snapshot(self.n, self.min_found, self.max_found, self.mean, self.bins_width, {self.bins_start + self.bins_width * i: bin_value / sum(self.bins) for i, bin_value in enumerate(self.bins) if bin_value})

def _run_plot(snapshots_queue):
    def redraw():
        try:
            snapshot = snapshots_queue.get(False)
        except queue.Empty:
            return

        plt.clf()
        plt.bar(snapshot.pdf.keys(), snapshot.pdf.values(), width=snapshot.bins_width)
        format_number = lambda n: f'{float(f"{n:.4g}"):g}'
        plt.xlabel('Result')
        plt.ylabel('Probability')
        #plt.ylim([0, 1])
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        mode = max(snapshot.pdf.keys(), key=snapshot.pdf.__getitem__)
        plt.title(f'Samples: {format_number(snapshot.n)} - Min: {format_number(snapshot.min)} - Mean: {format_number(snapshot.mean)} - Mode: {format_number(mode)}±{format_number(snapshot.bins_width/2)} - Max: {format_number(snapshot.max)}')
        plt.draw()

    fig = plt.figure()
    redraw()
    plt.show(block=False)
    while True:
        redraw()
        try:
            fig.canvas.flush_events()
        except:
            break

def plot(sequence, n=float('inf'), n_bins=100):
    if callable(sequence):
        fn = sequence
        sequence = (fn() for _ in itertools.count())
    else:
        sequence = iter(sequence)

    snapshots_queue = multiprocessing.Queue()

    plot_process = multiprocessing.Process(target=_run_plot, args=(snapshots_queue,))
    plot_process.start()


    digest = Digest(n_bins=n_bins, seed_value=next(sequence))
    i = 0
    while i < n-1:
        i += 1
        digest.update(next(sequence))
        if snapshots_queue.empty():
            snapshots_queue.put(digest.get_snapshot())

    last_snapshot = digest.get_snapshot()
    snapshots_queue.put(last_snapshot)
    return last_snapshot

from random import randint
def d(n):
    return randint(1, n)

if __name__ == '__main__':
    import sys
    import re
    from random import *

    if len(sys.argv) > 1:
        sequence = lambda: eval(' '.join(sys.argv[1:]))
    else:
        def stdin_numbers():
            for line in sys.stdin:
                yield from re.findall(r'\d+\.?\d*', line)
        sequence = stdin_numbers()

    print(plot(sequence))
