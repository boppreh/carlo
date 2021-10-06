from carlo import d, plot

import sys
import re
from random import *

def main():
    if len(sys.argv) > 1:
        # Usage:
        #     $ python -m carlo 'd(6)+d(12)'
        compiled_fns = [compile(arg, f'<argv_function_{i}>', 'eval') for i, arg in enumerate(sys.argv[1:])]
        sequences_or_fns = [lambda compiled_fn=compiled_fn: eval(compiled_fn) for compiled_fn in compiled_fns]
        labels = sys.argv[1:]
    else:
        # Usage:
        #     $ echo "1 2 3" | python -m carlo
        sequences_or_fns = [(number for line in sys.stdin for number in map(float, re.findall(r'\d+\.?\d*', line)))]
        labels = ()

    print(plot(*sequences_or_fns, labels=labels))

if __name__ == '__main__':
    main()