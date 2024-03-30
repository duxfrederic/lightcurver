import string
import itertools


def generate_star_names(n):
    """
    just so we have a nicer way (than gaia ID in a given combined footprint / selection of stars) to refer to our stars
    (assign each a letter / a combination of letters).
    Args:
        n: int, number of labels to generate

    Returns:
        list of strings, e.g. ['a', 'b', 'c', 'd', 'e', 'f', ...]
    """
    def all_strings():
        size = 1
        while True:
            for s in itertools.product(string.ascii_lowercase, repeat=size):
                yield "".join(s)
            size += 1

    return [name for _, name in zip(range(n), all_strings())]

