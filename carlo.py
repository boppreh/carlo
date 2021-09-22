import time, math

class Digest:
    def __init__(self, n_bins=100):
        self.results = []
        self.min_found, self.max_found = float('inf'), float('-inf')
        self.mean = None
        self.n = 0

    def update(self, value):
        self.n += 1
        self.results.append(value)
        self.min_found = min(self.min_found, value)
        self.max_found = max(self.max_found, value)
        self.mean = value if self.mean is None else (self.mean * self.n + value) / (self.n + 1)

    def describe(self, n_bins):
        if not self.results:
            return {}

        s = sorted(self.results)
        prev_bin = 0
        bin_width = (s[-1] - s[0]) / n_bins
        bins = [0 for i in range(n_bins+1)]
        for n in s:
            bins[int((n - s[0]) / bin_width)] += 1 
        return self.min_found, self.max_found, self.mean, {s[0] + bin_width * i: bins[i] / len(s) for i in range(n_bins)}

def plot_sample(fn, n):
    from threading import Thread, Lock
    stop = False
    digest = Digest()

    def run_samples():
        for i in range(n-1):
            if stop:
                return
            digest.update(fn())

    Thread(target=run_samples).start()

    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    fig = plt.figure()
    subplot = fig.add_subplot()
    fig.canvas.draw()
    plt.show(block=False)
    while True:
        subplot.clear()
        min_found, max_found, mean, pdf = digest.describe(100)
        subplot.plot(pdf.keys(), pdf.values())
        format_number = lambda n: f'{float(f"{n:.4g}"):g}'
        plt.xlabel('Result')
        plt.ylabel('Probability')
        plt.ylim([0, 1])
        #plt.xlim([min_found, max_found])
        subplot.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        subplot.title.set_text(f'Min: {format_number(min_found)} - Mean: {format_number(mean)} - Max: {format_number(max_found)}')
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
    plot_sample(lambda: time.sleep(0.01) or flip_or_bust(), n=10000)
    #plot_sample(lambda: (time.sleep(0.001), max(0.5, random.random()))[1], n=1000)
    #plot_sample(lambda: time.sleep(0.01) or random.choice([1, 2]), n=1000)