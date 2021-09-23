import multiprocessing
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from collections import namedtuple
import itertools
import math

Snapshot = namedtuple('Snapshot', 'n min max mean bins_width bins')

class Digest:
    def __init__(self, n_bins, seed_value=0):
        assert n_bins % 2 == 0, 'Number of bins must be even.'
        self.results = []
        self.min_found, self.max_found = float('inf'), float('-inf')
        self.mean = None
        self.n = 0
        self.bins_start = seed_value
        self.bins_end = seed_value
        self.n_bins = n_bins
        self.bins = [1]
        self.is_int = isinstance(seed_value, int)
        self.bins_width = 0
    
    def update(self, value, count=1):
        self.n += 1
        self.min_found = min(self.min_found, value)
        self.max_found = max(self.max_found, value)
        self.mean = value if self.mean is None else (self.mean * self.n + value) / (self.n + 1)

        if len(self.bins) == 1:
            if value == self.bins_start:
                self.bins[0] += 1
            else:
                old_value = self.bins_start
                old_count = self.bins[0]
                full_width = abs(self.bins_start - value)
                self.bins_width = full_width / self.n_bins
                midway = self.bins_start + (full_width) / 2
                if self.is_int:
                    midway = round(midway)
                    self.bins_width = max(1, math.ceil(self.bins_width))
                self.bins_start = midway - self.n_bins//2 * self.bins_width
                self.bins_end = midway + self.n_bins//2 * self.bins_width
                self.bins = [0] * self.n_bins
                self.update(old_value, old_count)

        while True:
            value_bin_i = round((value - self.bins_start) / self.bins_width)
            if 0 <= value_bin_i < self.n_bins:
                try:
                    self.bins[value_bin_i] += count
                except IndexError:
                    breakpoint()
                break

            non_empty_bin_indexes = [i for i, count in enumerate(self.bins) if count]
            min_bin_i = non_empty_bin_indexes[0]
            max_bin_i = non_empty_bin_indexes[-1]
            n_non_empty_bins = self.n_bins - (1 + max_bin_i - min_bin_i)
            if value_bin_i > max_bin_i and value_bin_i - max_bin_i <= n_non_empty_bins:
                # We can remove empty bins from the left to fit the value.
                self.bins = self.bins[n_non_empty_bins:] + [0] * n_non_empty_bins
                self.bins_start += n_non_empty_bins * self.bins_width
                self.bins_end += n_non_empty_bins * self.bins_width
            elif value_bin_i < min_bin_i and min_bin_i - value_bin_i <= n_non_empty_bins:
                # We can remove empty bins from the right to fit the value.
                self.bins = [0] * n_non_empty_bins + self.bins[:-n_non_empty_bins]
                self.bins_start -= n_non_empty_bins * self.bins_width
                self.bins_end -= n_non_empty_bins * self.bins_width
            else:
                # Value is too far away, shrink all bins by 2 and try again.
                self.bins = [self.bins[i] + self.bins[i+1] for i in range(0, self.n_bins, 2)] + self.n_bins//2 * [0]
                self.bins_width *= 2

    def get_snapshot(self):
        return Snapshot(self.n, self.min_found, self.max_found, self.mean, self.bins_width, {self.bins_start + self.bins_width * i: bin_value for i, bin_value in enumerate(self.bins) if bin_value})

def _run_plot(receiver_pipe):

    def draw(snapshot):
        plt.clf()
        #print(snapshot.bins)
        plt.bar(snapshot.bins.keys(), [value / snapshot.n for value in snapshot.bins.values()], width=snapshot.bins_width)
        format_number = lambda n: f'{float(f"{n:.4g}"):g}'
        plt.xlabel('Result')
        plt.ylabel('Probability')
        #plt.ylim([0, 1])
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        mode = max(snapshot.bins.keys(), key=snapshot.bins.__getitem__)
        plt.title(f'Samples: {format_number(snapshot.n)} - Min: {format_number(snapshot.min)} - Mean: {format_number(snapshot.mean)} - Mode: {format_number(mode)}±{format_number(snapshot.bins_width/2)} - Max: {format_number(snapshot.max)}')
        plt.draw()

    snapshot = receiver_pipe.recv()
    fig = plt.figure()
    draw(snapshot)
    plt.show(block=False)
    while True:
        if receiver_pipe.poll():
            snapshot = receiver_pipe.recv()
        draw(snapshot)
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

    receiver_pipe, sender_pipe = multiprocessing.Pipe(duplex=False)

    plot_process = multiprocessing.Process(target=_run_plot, args=(receiver_pipe,))
    plot_process.start()


    digest = Digest(n_bins=n_bins, seed_value=next(sequence))
    i = 0
    while i < n-1 and plot_process.is_alive():
        i += 1
        digest.update(next(sequence))
        if not receiver_pipe.poll():
            sender_pipe.send(digest.get_snapshot())

    last_snapshot = digest.get_snapshot()
    sender_pipe.send(last_snapshot)
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
