# transform a boolean vector to bit vector
def bool_to_bit_vec(bool_vec):
    return [1 if x else 0 for x in bool_vec]