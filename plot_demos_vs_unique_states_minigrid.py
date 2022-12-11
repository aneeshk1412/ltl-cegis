import matplotlib.pyplot as plt
from demos_gen_minigrid import generate_demonstrations
from asp_minigrid import action_selection_policy_DoorKey_ground_truth

def generate_new_random_demos(iters=10, seed=None):
    X, Y = [0], [0]
    for i in range(iters):
        demo = generate_demonstrations("MiniGrid-DoorKey-16x16-v0", action_selection_policy_DoorKey_ground_truth, seed=seed, num_demos=i, timeout=100)
        demo_set=set(demo)
        X.append(i)
        Y.append(len(demo_set))
    return X, Y

def accumulate_demos(iters=10, seed=None):
    X, Y = [0], [0]
    demo_set = set()
    for i in range(iters):
        demo = generate_demonstrations("MiniGrid-DoorKey-16x16-v0", action_selection_policy_DoorKey_ground_truth, seed=seed, num_demos=1, timeout=100)
        demo_set.update(demo)
        X.append(i)
        Y.append(len(demo_set))
    return X, Y

if __name__ == '__main__':
    X, Y = generate_new_random_demos(iters=50)

    plt_title="Variation of Unique (bool s, a) pairs with Number of Ground Truth Demos"
    plt_xlabel="Number of Demonstrations"
    plt_ylabel="Number of Unique (boolean state, action) pairs"

    plt.plot(X, Y, label='MiniGrid-DoorKey-16x16-v0', marker='o')
    plt.xlabel(plt_xlabel)
    plt.ylabel(plt_ylabel)
    plt.title(plt_title)
    plt.legend()

    plt.show()
