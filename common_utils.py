from itertools import takewhile


def filename_filter(filename, args):
    g_type, size, seed, optimal, p, m = parse_filename(filename)
    if g_type == args.graph_type and size == args.node and p == args.p and m == args.m:
        return True


def filter_filenames(filenames, args):
    return list(filter(lambda file: filename_filter(file, args), filenames))


def parse_filename(filename):
    # parameters:  type_size_seed_optimal_p_m
    g_type = "".join(takewhile(lambda x: not x.isnumeric(), filename)).rstrip("_")

    filename = filename.replace(g_type, "").replace(".adjlist", "").lstrip("_")
    params = filename.split("_")

    p = None
    m = None
    size, seed, optimal, *rest = params

    pindex = rest.index("p")
    mindex = rest.index("m")
    if rest[pindex + 1] != "":
        p = float(rest[pindex + 1])
    if rest[mindex + 1] != "":
        m = float(rest[mindex + 1])

    size = int(size)
    seed = int(seed)
    optimal = float(optimal)

    return g_type, size, seed, optimal, p, m
