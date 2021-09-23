import time, math

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
        #print(f'{value=}, {self.bins=}, {bin_i=}, {self.bins_width=}, {self.bins_start=}, {self.bins_end=}')
        self.bins[bin_i] += 1

    def describe(self):
        return self.min_found, self.max_found, self.mean, {self.bins_start + self.bins_width * i: bin_value / sum(self.bins) for i, bin_value in enumerate(self.bins)}

def plot_sample(fn, n):
    from threading import Thread, Lock
    stop = False
    digest = Digest(20, fn())

    def run_samples():
        nonlocal stop
        for i in range(n-1):
            if stop:
                return
            digest.update(fn())
        stop = True

    Thread(target=run_samples).start()

    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    fig = plt.figure()
    subplot = fig.add_subplot()
    fig.canvas.draw()
    plt.show(block=False)
    while True:
        if not stop:
            subplot.clear()
            min_found, max_found, mean, pdf = digest.describe()
            subplot.bar(pdf.keys(), pdf.values(), width=digest.bins_width)
            format_number = lambda n: f'{float(f"{n:.4g}"):g}'
            plt.xlabel('Result')
            plt.ylabel('Probability')
            plt.ylim([0, 1])
            #plt.xlim([min_found, max_found])
            subplot.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            mode = max(pdf.keys(), key=pdf.__getitem__)
            subplot.title.set_text(f'Min: {format_number(min_found)} - Mean: {format_number(mean)} - Mode: {format_number(mode)}±{format_number(digest.bins_width/2)} - Max: {format_number(max_found)}')
        fig.canvas.draw_idle() # Super important, otherwise the window freezes.
        try:
            fig.canvas.flush_events()
        except:
            break

    stop = True
    return digest

if __name__ == '__main__':
    import random
    def flip_or_bust():
        money = 1
        while True:
            if random.random() > 0.5:
                money *= 2
            else:
                return money
    results = []
    for i in range(1000):
        results.append(max(0.5, random.random()))
    #plot_sample(lambda: time.sleep(0.01) or flip_or_bust(), n=10000)
    plot_sample(lambda: (time.sleep(0.01), max(0.5, random.random()))[1], n=1000)
    #plot_sample(lambda: time.sleep(0.01) or random.choice([1, 2]), n=1000)