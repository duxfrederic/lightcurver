import string
import itertools


def generate_star_names(n):
    def all_strings():
        size = 1
        while True:
            for s in itertools.product(string.ascii_lowercase, repeat=size):
                yield "".join(s)
            size += 1

    return [name for _, name in zip(range(n), all_strings())]

