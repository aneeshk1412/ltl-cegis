# helping functions are defined here


# return Cartesian product of given lists
from enum import Flag


def product(*args):
    if not args:
        return iter(((),))  # yield tuple()
    return (items + (item,)
            for items in product(*args[:-1]) for item in args[-1])


# return Cartesian product of given lists capped by a given number
def product(lst, cap, dim, repeat=True):
    assert dim in {1, 2, 3} # for now we hard code it for these two cases
    res = []
    iter = 0
    flag = False


    if dim == 1:
        ls = (list(map(lambda x:[x], lst)))
        return ls[:cap]

    if dim == 2:
        for e1 in lst:
            for e2 in lst:
                if not(repeat) and e1==e2:
                    continue
                iter += 1
                res.append((e1, e2))
                if iter >= cap:
                    flag = True
                    break
            if flag:
                break

    elif dim == 3:
        for e1 in lst:
            for e2 in lst:
                if not(repeat) and e1==e2:
                    continue
                for e3 in lst:
                    if not(repeat) and (e1==e3 or e2==e3):
                        continue
                    iter += 1
                    res.append((e1, e2, e3))
                    if iter >= cap:
                        flag = True
                        break
                if flag:
                    break
            if flag:
                break
    else:
        raise Exception('the requested dimensionality is not supported')
    return res



class bcolors:
    HEADER = '\u001b[36m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'