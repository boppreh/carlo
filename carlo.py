import multiprocessing
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick
from collections import namedtuple
import itertools
import math

class Digest:
    """
    Computes a running histogram and general statistics about the incoming data.
    Automatically adjusts bin sizes and locations as new data comes in.

    The secret to maintain correctness is to never make changes that require
    "splitting" bins. This leaves us with moving bins left and right (if there
    are empty bins at the ends), or merging bins by doubling their size.

    Since new values will never make the histogram range shrink, those three
    operations are sufficient to always cover incoming values, at the cost
    of having up to 50% of bins empty.
    """
    def __init__(self, n_bins, seed_value, is_int=None):
        """
        Initializes the running digest.
        - `n_bins`: the number of bins to divide the histogram in. Note that
        the histogram starts with only one bin, until a second unique value is
        seen, then the full number of bins is created.
        - `seed_value`: a value sampled from the running sequence, to be used
        for initial alignment of bins and min/max value seen.
        - `is_int`: is True, forces the start and end values of bins to be
        integers. If None, automatically decides this based on the seed value.
        """
        if n_bins % 2 == 1:
            raise ValueError(f'Number of bins must be even, got {n_bins}.')
        self.results = []
        self.min = self.max = seed_value
        self.mean = None
        self.n = 1
        self.bins_start = seed_value
        self.bins_end = seed_value
        self.n_bins = n_bins
        # Unless overridden, assume that if the seed value is an integer, all
        # following values will be integers too, and force the bin attributes to
        # be integers too.
        self.is_int = is_int if is_int is not None else isinstance(seed_value, int)
        self.bins_width = 0
        # The `bins` attribute is an array for efficiency reasons, but converted
        # to a more useful dict {bin_middle_point: count} when taking a snapshot.
        self.bins = [1]
    
    def update(self, value, count=1):
        """
        Updates this instances statistics to include the new value.
        """
        # Count (`n`), maximum and minimum values, and mean are computed
        # separately from the histogram to increase precision.
        self.n += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        # TODO: improve numeric stability.
        self.mean = value if self.mean is None else (self.mean * self.n + value) / (self.n + 1)

        if self.bins_width == 0:
            # We have only seen one unique value so far, and cannot compute bins yet.

            if value == self.bins_start:
                # Another duplicated value, increment and skip everything else.
                self.bins[0] += 1
                return

            # We'll need to restore the current count after creating the new bins.
            old_value = self.bins_start
            old_count = self.bins[0]

            full_width = abs(self.bins_start - value)
            self.bins_width = full_width / self.n_bins
            midway = self.bins_start + (full_width) / 2
            if self.is_int:
                midway = round(midway)
                self.bins_width = max(1, math.ceil(self.bins_width))
            # Place old and new value at the extremes of the histogram.
            self.bins_start = midway - self.n_bins//2 * self.bins_width
            self.bins_end = midway + self.n_bins//2 * self.bins_width
            self.bins = [0] * self.n_bins
            # Restore old count.
            old_bin_i = round((old_value - self.bins_start) / self.bins_width)
            self.bins[old_bin_i] = old_count

        while True:
            # Compute bin index for value.
            value_bin_i = round((value - self.bins_start) / self.bins_width)
            if 0 <= value_bin_i < self.n_bins:
                # Bin index is valid, increment and stop.
                self.bins[value_bin_i] += count
                break

            # The new value does not fit into our current bin division. We have
            # three options: move the bins right, left, or shrink them. Which
            # one we do depends on how many empty bins at ends we have for
            # moving.

            # Find which bins we are actually using, i.e. non-empty.
            non_empty_bin_indexes = [i for i, count in enumerate(self.bins) if count]
            min_bin_i = non_empty_bin_indexes[0]
            max_bin_i = non_empty_bin_indexes[-1]
            n_non_empty_bins = self.n_bins - (1 + max_bin_i - min_bin_i)
            if value_bin_i > max_bin_i and value_bin_i - max_bin_i <= n_non_empty_bins:
                # We can remove empty bins from the left to fit the new value.
                self.bins = self.bins[n_non_empty_bins:] + [0] * n_non_empty_bins
                # We move the histogram "window" all the way to the right.
                self.bins_start += n_non_empty_bins * self.bins_width
                self.bins_end += n_non_empty_bins * self.bins_width
            elif value_bin_i < min_bin_i and min_bin_i - value_bin_i <= n_non_empty_bins:
                # We can remove empty bins from the right to fit the new value.
                self.bins = [0] * n_non_empty_bins + self.bins[:-n_non_empty_bins]
                # We move the histogram "window" all the way to the left.
                self.bins_start -= n_non_empty_bins * self.bins_width
                self.bins_end -= n_non_empty_bins * self.bins_width
            else:
                # Value is too far away, shrink all bins by 2 and try again.
                # Create new empty bins at the end. May not be right, but
                # the loop will fix it if needed.
                self.bins = [self.bins[i] + self.bins[i+1] for i in range(0, self.n_bins, 2)] + self.n_bins//2 * [0]
                self.bins_width *= 2

    def get_snapshot(self):
        """
        Returns an immutable snapshot of the statistics so far.
        """
        bins_dict = {self.bins_start + self.bins_width * i: bin_value for i, bin_value in enumerate(self.bins) if bin_value}
        return Snapshot(self.n, self.min, self.max, self.mean, self.bins_width, bins_dict)

# A snapshot of the running digest, necessary to maintain thread-/process- safety.
Snapshot = namedtuple('Snapshot', 'n min max mean bins_width bins')

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

def plot(sequence, n=float('inf'), n_bins=100, is_int=None):
    if callable(sequence):
        fn = sequence
        sequence = (fn() for _ in itertools.count())
    else:
        sequence = iter(sequence)

    receiver_pipe, sender_pipe = multiprocessing.Pipe(duplex=False)

    plot_process = multiprocessing.Process(target=_run_plot, args=(receiver_pipe,))
    plot_process.start()


    digest = Digest(n_bins=n_bins, seed_value=next(sequence), is_int=None)
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
