import numpy as np
import matplotlib.pyplot as plt
import sigma_clip as sc

def test_sigma_clip():
    """
    Simulate data drawn from a gaussian with outliers. Assert that the correct
    data are removed.
    """

    np.random.seed(123)

    N, Noutliers= 100, 10
    x = np.linspace(0, 10, N)
    y = np.random.randn(N)
    inds = range(100)
    outlier_inds = np.random.choice(inds, size=Noutliers)
    y[outlier_inds] = np.random.randn(Noutliers) * 20

    plt.plot(x, y, ".")

    _, mask = sc.sigma_clip(y)

    plt.plot(x[outlier_inds], y[outlier_inds], "*")
    plt.plot(x[~mask], y[~mask], ".")
    plt.savefig("test")


def test_running_sigma_clip():
    """
    Simulate data drawn from a gaussian with outliers and a sinusoidal mean
    function. Assert that the correct data are removed.
    """

    np.random.seed(123)

    # Generate data
    N, Noutliers= 1000, 10
    x = np.linspace(0, 100, N)
    y = 10*np.sin(2*np.pi*(1./10)*x)
    y += np.random.randn(N)
    inds = range(N)
    outlier_inds = np.random.choice(inds, size=Noutliers)
    y[outlier_inds] = np.random.randn(Noutliers) * 20

    # plot data
    plt.figure(figsize=(16, 9))
    plt.plot(x, y, ".", label="all data")

    # mask, spl = sc.spline_sigma_clip(x, y, len(x)*10, sigma=3)
    # plt.plot(x[~mask], y[~mask], ".", label="clipped")

    # Simple sigma clip
    _, mask = sc.sigma_clip(y)
    plt.plot(x[~mask], y[~mask], "co", ms=20, zorder=0, alpha=.5,
             label="non-running outliers")

    # Running sigma clip
    mask = sc.running_sigma_clip(10, x, y)
    plt.plot(x[~mask], y[~mask], "k^",
             label="running outliers", zorder=2)
    print(N - sum(mask), "running outliers found")

    # # Interval sigma clip
    # _, mask = sc.interval_sigma_clip(2, x, y)
    # print(N - sum(mask), "interval outliers found")

    # plt.plot(x[outlier_inds], y[outlier_inds], "*", ms=10,
    #          label="true outliers", zorder=1)
    # plt.plot(x[~mask], y[~mask], ".", label="interval outliers", zorder=3)

    plt.legend()
    plt.savefig("test")

if __name__ == "__main__":

    # test_sigma_clip()
    test_running_sigma_clip()
