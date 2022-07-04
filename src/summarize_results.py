import numpy as np
import os, sys

def print_usage():
    print(
        f"Usage: {sys.argv[0]} <result file> <mode>"
        f""
        f"  mode = mpeg | ae"
    )
    exit(1)

def read_log_file(resultfile):
    f = open(resultfile)
    lines = f.readlines()[1:]
    data = np.loadtxt(lines, delimiter=',')
    return data

def add_key_val(key_to_vals, key, val):
    vals = key_to_vals[key] if key in key_to_vals else []
    vals.append(val)
    key_to_vals[key] = vals

if __name__ == "__main__":
    resultfile = sys.argv[1]
    mode = sys.argv[2]

    data = read_log_file(resultfile)
    
    level_to_sizes = {}
    level_to_quals = {}
    levels = set()
    for vec in data:
        if mode == "mpeg":
            size = vec[1]
            qual = vec[2]
            qp = vec[3]
            if qp <= 0:
                continue
            add_key_val(level_to_sizes, qp, size)
            add_key_val(level_to_quals, qp, qual)
            levels.add(qp)
        if mode == "ae":
            size = vec[1]
            qual = vec[2]
            loss = vec[3]
            qp = vec[4]
            if qp <= 0:
                continue
            level = str(loss)+"-"+str(qp)
            add_key_val(level_to_sizes, level, size)
            add_key_val(level_to_quals, level, qual)
            levels.add(level)

    levels = sorted(levels)
    for level in levels:
        print(np.mean(level_to_sizes[level]), np.mean(level_to_quals[level]))
