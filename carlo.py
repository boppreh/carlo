import time, math

def plot_sample(fn, n):
    from threading import Thread, Lock
    from tdigest import TDigest
    digest = TDigest()
    min_found = max_found = fn()
    stop = False
    digest_lock = Lock()

    def run_samples():
        nonlocal min_found, max_found

        last_time = time.time()
        
        for i in range(n-1):
            if stop:
                return
            result = fn()
            with digest_lock:
                digest.update(result)
                min_found, _, max_found = sorted([min_found, result, max_found])

    Thread(target=run_samples).start()

    from matplotlib import pyplot as plt
    import matplotlib.ticker as mtick
    fig = plt.figure()
    subplot = fig.add_subplot()
    fig.canvas.draw()
    plt.show(block=False)
    while True:
        subplot.clear()
        n_values = 100
        with digest_lock:
            values = [min_found + (max_found - min_found) / n_values * i for i in range(n_values)]
            # Sorted because it often returns percentiles that are decreasing.
            #percentiles = sorted([min_found] + [digest.percentile(i) for i in range(1, 100)] + [max_found])
            cdf = [digest.cdf(value) for value in values]
        print(values[0], cdf[0])
        pdf = [cdf[0]] + [cdf[i] - cdf[i-1] for i in range(1, len(cdf))]
        #print(values)
        #print(pdf)
        #print(cdf)
        subplot.plot(values, pdf)
        format_number = lambda n: f'{float(f"{n:.4g}"):g}'
        plt.xlabel('Return')
        plt.ylabel('Probability')
        plt.ylim([0, 1])
        plt.xlim([min_found, max_found])
        subplot.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        subplot.title.set_text(f'Min: {format_number(min_found)} - Mean: {format_number(digest.percentile(50))} - Max: {format_number(max_found)}')
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
    #plot_sample(lambda: time.sleep(0.01) or flip_or_bust(), n=1000)
    plot_sample(lambda: max(0.5, random.random()), n=1000)
    #plot_sample(lambda: time.sleep(0.01) or random.choice([1, 2]), n=1000)