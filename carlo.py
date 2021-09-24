import math
import multiprocessing
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt
from collections import namedtuple

CONFIDENCE = 0.95

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
        self.mean = seed_value
        self.msq = 0
        self.n = 1
        self.bins_start = seed_value
        self.bins_end = seed_value
        self.n_bins = n_bins
        # Unless overridden, assume that if the seed value is an integer, all
        # following values will be integers too, and force the bin attributes to
        # be integers too.
        self.is_int = is_int if is_int is not None else isinstance(seed_value, int)
        self.bins_width = 1
        # The `bins` attribute is an array for efficiency reasons, but converted
        # to a more useful dict {bin_middle_point: count} when taking a snapshot.
        self.bins = [1]
    
    def update(self, value, count=1):
        """
        Updates this instances statistics to include the new value.
        """
        # Count (`n`), maximum and minimum values, and mean are computed
        # separately from the histogram to increase precision.
        self.min = min(self.min, value)
        self.max = max(self.max, value)

        # Online mean and variance algorithm, adapted from Welford (1962), described here:
        # https://stats.stackexchange.com/questions/235129/online-estimation-of-variance-with-limited-memory
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        # Note that this can't be replaced by delta ** 2 because the `mean`
        # was updated after delta was calculated.
        self.msq += delta * (value - self.mean)

        if len(self.bins) == 1:
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
        variance = float('nan') if self.n == 1 else self.msq / (self.n - 1)
        return Snapshot(self.n, self.min, self.max, self.mean, variance, self.bins_width, bins_dict, self.is_int)

# A snapshot of the running digest, necessary to maintain thread-/process- safety.
Snapshot = namedtuple('Snapshot', 'n min max mean variance bins_width bins is_int')

def _run_plot(receiver_pipe):
    """
    Continually pulls snapshots from the given pipe and displays them
    interactively with matplotlib until the user closes the window.
    """
    # Black magic helper function to format a number to 4 significant places.
    format_number = lambda n: f'{float(f"{n:.4g}"):g}'

    def draw(snapshots):
        """  Updates the rendering with the given snapshot. """
        plt.clf()
        plt.xlabel('Result')
        plt.ylabel('Probability')
        # Formats Y axis with 0% to 100% instead of 0 to 1.
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        # We have computed the bins and bar heights already, so use `.bar()`
        # instead of `.hist()`.
        for snapshot in snapshots:
            mode = max(snapshot.bins.keys(), key=snapshot.bins.__getitem__)
            mode_error = '' if snapshot.is_int and snapshot.bins_width <= 1 else f'±{format_number(snapshot.bins_width/2)}'
            if math.isnan(snapshot.variance):
                mean_error_str = ''
            else:
                stdev = math.sqrt(snapshot.variance)
                mean_error = stdev / math.sqrt(snapshot.n)
                mean_error_str = f'±{format_number(mean_error)}'
            label = f'Samples: {format_number(snapshot.n)} - Min: {format_number(snapshot.min)} - Mean: {format_number(snapshot.mean)}{mean_error_str} - Mode: {format_number(mode)}{mode_error} - Max: {format_number(snapshot.max)}'
            plt.bar(snapshot.bins.keys(), [value / snapshot.n for value in snapshot.bins.values()], width=snapshot.bins_width, label=label)
        plt.legend()
        plt.draw()

    fig = plt.figure()
    # Blocks until the first snapshot is provided to avoid showing incorrect data.
    snapshots = receiver_pipe.recv()
    draw(snapshots)
    plt.show(block=False)
    while True:
        # The receiver pipe will contain at most one snapshot, and by receiving
        # it we signal to the sender that they should prepare a new one. If a
        # snapshot is not available, render the previous one again.
        try:
            if receiver_pipe.poll():
                snapshots = receiver_pipe.recv()
        except BrokenPipeError:
            pass
        draw(snapshots)
        try:
            fig.canvas.flush_events()
        except:
            # When the window closes this method throws "_tkinter.TclError", but
            # that's too unrelated from this code to import. Just catch everything.
            break

def plot(*sequences_or_fns, n=float('inf'), n_bins=100, is_int=None):
    """
    Plots a sequence of values, or the results of repeatedly calling the given
    function. The statistics of the values is continually updated and shown in
    an interactive window to the user.

    - `sequence_or_fn`: if a list, tuple, or generator of numbers is given,
    compute streaming statistics on all values. If a function is given,
    repeatedly call it with no arguments and plot the returned numbers.
    - `n`: plot at most this many values. Defaults to infinite.
    - `n_bins`: maximum number of bins to spit the values into. To deal with the
    streaming values, the actual number of bins will be between n_bins/2 and n_bins.
    - `is_int`: is True, forces the start and end values of bins to be
    integers. If None, automatically decides this based on the seed value.
    """
    iterators = []
    for sequence_or_fn in sequences_or_fns:
        if callable(sequence_or_fn):
            def loop(fn=sequence_or_fn):
                # Could be replaced with a generator and itertools.count, but I
                # had some inexplicable aliasing problems that resulted in the
                # the same function being used in every iterator.
                while True:
                    yield fn()
            iterators.append(loop())
        else:
            iterators.append(iter(sequence_or_fn))

    receiver_pipe, sender_pipe = multiprocessing.Pipe(duplex=False)

    # Run plotting window in a separate process to minimize the performance impact
    # of plotting on the computation of the sequence, and to avoid the computation
    # from lagging the plotting window.
    plot_process = multiprocessing.Process(target=_run_plot, args=(receiver_pipe,))
    plot_process.start()

    digests = [Digest(n_bins=n_bins, seed_value=next(iterator), is_int=None) for iterator in iterators]
    # Manual counting with a while loop to accommodate `n=float('inf')`.
    i = 0
    # The plot_process will die when the user closes the window, and we should
    # stop the computation too.
    try:
        while i < n-1 and plot_process.is_alive():
            i += 1
            for iterator, digest in zip(iterators, digests):
                digest.update(next(iterator))

            if not receiver_pipe.poll():
                # The plotting window has just consumed a snapshot, give it another one.
                sender_pipe.send([digest.get_snapshot() for digest in digests])
    except StopIteration:
        # An iterator has finished.
        pass

    # Ensure that the final state of the sequence is shown.
    last_snapshots = [digest.get_snapshot() for digest in digests]
    sender_pipe.send(last_snapshots)

    # Probably nobody will use this, but it's easy to return the last snapshot.
    return last_snapshots[0] if len(last_snapshots) == 1 else last_snapshots

from random import randint
def d(n):
    """ Generates a random number by rolling a `n`-sided dice, e.g. d(6). """
    return randint(1, n)

if __name__ == '__main__':
    import sys
    import re
    from random import *

    if len(sys.argv) > 1:
        # Usage:
        #     $ python -m carlo 'd(6)+d(12)'
        sequences_or_fns = [(lambda arg=arg: eval(arg)) for arg in sys.argv[1:]]
    else:
        # TODO: fix broken example
        # Usage:
        #     $ echo "1 2 3" | python -m carlo
        sequences_or_fns = [(number for line in sys.stdin for number in map(float, re.findall(r'\d+\.?\d*', line)))]

    print(plot(*sequences_or_fns))
