import matplotlib.pyplot as plt

def avg(l):
    assert(len(l) > 0)
    return sum(l) / len(l)

def num_demos_vs_num_cegis_loops():
    X = [1, 2, 3, 5, 10]
    Y = [[7, 16, 29, 103, 44], [10, 3, 18, 4, 49], [9, 64, 2, 1, 21], [0, 1, 1, 2, 2], [0, 1, 0, 2, 3]]
    Y = [avg(l) for l in Y]
    return X, Y

if __name__ == '__main__':
    X, Y = num_demos_vs_num_cegis_loops()

    plt_title="Number of CEGIS loops -vs- Number of Ground Truth Demos"
    plt_xlabel="Number of Demonstrations"
    plt_ylabel="Number of CEGIS loops to SAT"

    plt.plot(X, Y, label='MiniGrid-DoorKey-16x16-v0', marker='o')
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.title(plt_title)
    plt.legend()

    plt.show()
