import scipy.signal as sig


# Discounting function used to calculate discounted returns.
# From: Arthur Juliani's blog on RL algorithms on github.com/awjuliani
def discount(x, gamma):
    """
    Filters the input series such that a discounted accumulated series is returned (by discount factor gamma).
    Example:
    input=[100, 1, 3]
    output=[100+g1+gg3, 1+g3, 3, ...] (where g=gamma and gg=gamma^2)

    Args:
        x (ndarray): The 1D input sequence.
        gamma (float): The discount factor.

    Returns: The discounted-accumulated return series.
    """

    return sig.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


if __name__ == "__main__":
    rs = [3, 4, 1, 2]
    # should return: 3+g4+gg1+ggg2, 4+g1+gg2, 1+g2, 2
    returns = discount(rs, 1.0)

    print(returns)
