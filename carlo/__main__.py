from carlo import d, plot

import sys
import re
from random import *

def main():
    if len(sys.argv) > 1:
        # Usage:
        #     $ carlo "d(6)+d(12)"

        # Fix issue where Windows users typing single quotes would cause
        # quoted strings to be passed as argument.
        args = [arg.strip('\'"') for arg in sys.argv[1:]]
        compiled_fns = [compile(arg, f'<argv_function_{i}>', 'eval') for i, arg in enumerate(args)]
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