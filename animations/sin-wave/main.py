import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import interpolate
import seaborn as sns


def get_wave_with_variable_frequency(time, freq_array):
    """Returns an array sin(f(t) * t)"""
    # full_like uses time to determine the shape and data type
    # of the returned array
    # time[1] - time[0] is the value to fill it with
    # dt = np.full_like(time, time[1] - time[0])
    dt = time[1] - time[0]  # Time step
    phases = (freq_array * 2 * np.pi * dt).cumsum()
    return np.sin(phases), ((phases + np.pi) % (2 * np.pi) - np.pi)


def setup_time_axes(t_start, t_end):
    fig, ax = plt.subplots(1, 1, figsize=(20, 4), dpi=300)
    fig.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(t_start - 0.1, t_end + 0.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(color="white", linewidth=0.4, alpha=0.3, zorder=0)
    return fig, ax


def angle2color(angle, cmap):
    # return phase_cmap((angle % (2 * np.pi)) / (2 * np.pi))
    return cmap((angle % (2 * np.pi)) / (2 * np.pi))


# Animation Function
def animate_wave(t_current, time, lc):
    # Create an array of alpha values where each segment's visibility
    # is determined by the condition
    alphas = np.where(
        time[:-1] <= t_current, 1, 0
    )  # Ends of each segment determine visibility
    lc.set_alpha(alphas)  # Apply alpha values to the line collection
    return (lc,)


if __name__ == "__main__":

    N_points = 5000
    t_start = 0
    t_end = 5

    # Array of time points
    time = np.linspace(t_start, t_end, N_points)

    # Generating a random frequency modulation pattern
    generator = np.random.default_rng(seed=322)  # A way to generate random numbers
    x_samples = np.linspace(t_start, t_end, 10)
    freq_samples = generator.random(x_samples.shape) * 6

    # For interp1d, x and y (x_sample and freq_samples) are used to approximate
    # some function. It returns a function whose call method uses interpolation
    # to find the value of new points.
    interpolation = interpolate.interp1d(x_samples, freq_samples, kind="quadratic")

    # Array of frequencies
    freq = interpolation(time)  # time == the new points to interpolate

    wave, phase = get_wave_with_variable_frequency(time, freq)

    # Periodic colormap to turn angles into colors
    phase_cmap = sns.color_palette("hls", as_cmap=True)

    fig, ax = setup_time_axes(t_start, t_end)

    # With reshape, the -1 tells numpy that we don't know what that dimension
    # is supposed to be and numpy will work it out for us.
    points = np.array([time, wave]).T.reshape(-1, 1, 2)

    segments = np.concatenate(
        [points[:-1], points[1:]], axis=1
    )  # Segments for LineCollection

    lc = matplotlib.collections.LineCollection(segments, linewidth=6)
    lc.set_colors(angle2color(phase[:-1], phase_cmap))
    lc.set_capstyle("round")
    ax.add_collection(lc)

    # # Animation class via FuncAnimation
    anim = FuncAnimation(
        fig,
        animate_wave,
        fargs=(time, lc),
        frames=np.linspace(t_start, t_end, 5000),
        interval=30,
    )
    anim.save("videos/sin_wave.mp4")
