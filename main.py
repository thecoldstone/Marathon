import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# settings for seaborn plotting style
sns.set(color_codes=True)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(5,5)})


def add_matrices(mx_1, mx_2, index):
    mx_1[:, :, index] = mx_1[:, :, index] + mx_2


def plot_graphic(runners, p):

    y_pos = np.arange(len(runners))

    plt.bar(y_pos, p, align='center', alpha=0.5)
    plt.xticks(y_pos, runners)
    plt.ylabel('Victories')
    plt.title('Runner')

    plt.show()


def individual(r_1, r_2, r_3, r_4, stats):

    add_matrices(stats, r_1, 0)
    add_matrices(stats, r_2, 1)
    add_matrices(stats, r_3, 2)
    add_matrices(stats, r_4, 3)

    results = stats[:, n - 1, :]
    _, performance = np.unique(np.argmax(results, axis=1), return_counts=True)

    plot_graphic(('Winnie', 'Piglet', 'Rabbit', 'Eeyore'), performance)


def group(r_1, r_2, r_3, r_4, stats):

    g_1 = r_1 + r_2
    g_2 = r_3 + r_4

    add_matrices(stats, g_1, 0)
    add_matrices(stats, g_2, 1)

    results = stats[:, n - 1, :]
    _, performance = np.unique(np.argmax(results, axis=1), return_counts=True)

    plot_graphic(('Winnie + Piglet', 'Rabbit + Eeyore'), performance)


if __name__ == "__main__":

    np.random.seed(70)

    n = int(input("For how long each marathon will last: "))
    m = input("Mode I/G: ")
    players = 4

    if m is "G":
        players = 2

    """ Initialize 2d matrices for runners
        1 axe: number of marathon
        2 axe: samples from the parameterized distributions.
    """
    winnie = np.random.exponential(size=(pow(10, 4) * n)).reshape(pow(10, 4), n)
    piglet = np.random.normal(size=(pow(10, 4) * n), loc=1, scale=1).reshape(pow(10, 4), n)
    rabbit = np.random.poisson(size=(pow(10, 4) * n)).reshape(pow(10, 4), n)
    eeyore = np.random.binomial(n, p=0.5, size=(pow(10, 4) * n)).reshape(pow(10, 4), n)

    """ Initialize 3d matrix for marathon statistics
        1 axe: number of marathon
        2 axe: samples from the parameterized distributions
        3 axe: number of runner
    """
    s = np.zeros((pow(10, 4), n, players))

    if m is "I":
        individual(winnie, piglet, rabbit, eeyore, s)
    elif m is "G":
        group(winnie, piglet, rabbit, eeyore, s)
