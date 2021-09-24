# carlo

Interactively plot streaming sequences of numbers.

Accepts lists, generators, numbers from stdin, or a function to be repeatdly evaluated. Displays data as histograms, with bin size and location automatically adjusted, and extra statistics in legends.

## Example 1

Compare samples from one 20-sided dice vs three 6-sided dices.

As standalone module (where `d(n)` simulates the roll of a `n`-sided dice):

    $ python -m carlo 'd(20)' 'd(6)+d(6)+d(6)'

Or imported:

    from carlo import plot, d
    plot(lambda: d(20), lambda: d(6)+d(6)+d(6))
    
![example screenshot showing two histograms superimposed](./screenshot1.png)

## Example 2

Sample values from `max(0.5, random()**0.2)`.

As standalone module (all functions from the `random` module are automatically available).

    $ python -m carlo 'max(0.5, random()**0.2)'

Or imported:

    from carlo import plot
    from random import random
    plot(lambda: max(0.5, random()**0.2))
    
![example screenshot showing a skewed-looking histogram](./screenshot2.png)
