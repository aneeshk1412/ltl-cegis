import matplotlib.pyplot as plt

def avg(l):
    if len(l):
        return sum(l) / len(l)
    return 0

def num_demos_vs_num_cegis_loops():
    seeds = [100, 300, 400]
    num_demos = [1, 2, 3, 5, 8, 10]
    num_cegis_loops = [[11, 13, 4], [11, 13, 4], [11, 13, 4], [11, 13, 4], [], []]
    num_cegis_loops = [avg(l) for l in num_cegis_loops]
    return num_demos, num_cegis_loops

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
