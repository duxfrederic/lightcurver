import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from astropy.time import Time

SEASON_PAD = 20.0  # days
COLOR_CYCLE = ["royalblue", "crimson", "darkorange", "forestgreen", "purple"]


def find_sources(df):
    """
    Return the list of point source identifiers (like 'A', 'B', 'C'...)
    that have *_mag and *_d_mag_down, *_d_mag_up columns in the CSV.
    """
    # Look for any column that ends with "_mag"
    mag_cols = [col for col in df.columns if col.endswith("_mag")]
    sources = []
    for col in mag_cols:
        # Strip off the "_mag" part to get the source name
        source_candidate = col.replace("_mag", "")
        # Check that the needed error columns exist
        d_down = f"{source_candidate}_d_mag_down"
        d_up = f"{source_candidate}_d_mag_up"
        if d_down in df.columns and d_up in df.columns:
            sources.append(source_candidate)
    return sources


def measure_scatter(mag_series):
    """
    Return the scatter (90th - 10th percentile) of a magnitude series.
    """
    q90 = np.nanpercentile(mag_series, 90)
    q10 = np.nanpercentile(mag_series, 10)
    return q90 - q10


def compute_offsets(df, sources):
    """
    Compute offsets so that the brightest source is the reference (offset=0),
    and each subsequent source is moved to avoid overlap, using a simplistic
    approach:
       - Sort sources by median brightness (lowest mag is brightest)
       - The first source has offset 0
       - For the next sources, shift them relative to the previous source
         by the difference in medians plus an extra separation factor based on scatter.
    Return a dict: {source_name: offset_value}
    """
    # get median magnitudes
    source_medians = {}
    for s in sources:
        mags = df[f"{s}_mag"].dropna()
        if len(mags) > 0:
            source_medians[s] = np.median(mags)
        else:
            source_medians[s] = np.inf  # if no data, push it to the end

    # sort sources by brightness
    sorted_sources = sorted(sources, key=lambda source: source_medians[source])

    offsets = {sorted_sources[0]: 0.0}
    for i in range(1, len(sorted_sources)):
        prev_s = sorted_sources[i-1]
        curr_s = sorted_sources[i]
        prev_median = source_medians[prev_s]
        curr_median = source_medians[curr_s]
        # Define a separation factor using scatter from both
        sep_prev = measure_scatter(df[f"{prev_s}_mag"].dropna())
        sep_curr = measure_scatter(df[f"{curr_s}_mag"].dropna())
        # shift so that the median of curr_s lines up above prev_s by about 30% the sum of scatters
        offsets[curr_s] = (prev_median - curr_median) + 0.3*(sep_prev + sep_curr)
        # ah, and cumulate the previous offset:
        offsets[curr_s] += offsets[prev_s]

    return offsets


def find_segments(df, gap_threshold):
    """
    Split MJD range into segments separated by 'gap_threshold' days.
    Return a list of (start_mjd, end_mjd) for each segment.
    """
    mjd_sorted = np.sort(df['mjd'].unique())
    gaps = np.where(np.diff(mjd_sorted) > gap_threshold)[0]
    segments = []
    start = 0
    for gap in gaps:
        end = gap
        segments.append((mjd_sorted[start], mjd_sorted[end]))
        start = end + 1
    segments.append((mjd_sorted[start], mjd_sorted[-1]))
    return segments


def add_break_indicator(ax, width_ratio, left=True, right=True):
    """
    Cosmetic function, adds break indicator to the plot segments.
    Args:
        ax: matplotlib axis object
        width_ratio: ratio of the width of the axis to that of full plot
        left: add left indicators? Pass False if first segment
        right: add right indicators? Pass False if last segment.

    Returns:
        None
    """
    d = .008  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    dw = 0.4 * d / width_ratio
    kwargs['color'] = 'gray'
    if right:
        ax.plot((1 - dw, 1 + dw), (-d, +d), **kwargs)  # top-left diagonal
        ax.plot((1 - dw, 1 + dw), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    if left:
        ax.plot((-dw, dw), (-d, +d), **kwargs)  # top-right diagonal
        ax.plot((-dw, dw), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


def plot_photometry(csv_file, output_pdf, gap_threshold=70.0, sources=None, figsize=(12, 5),
                    plot_title=None):
    """
    function that:
      - reads the pipeline CSV
      - identifies sources
      - computes offsets
      - splits data into segments (seasons)
      - makes a multi-panel figure, one panel per segment


    Args:
        csv_file: CSV file containing the photometry, output of the automatic modelling
        output_pdf: where to write the plot, PDF but matplotlib will handle other containers
        gap_threshold: gaps defining seasons; gaps larger than this will be eliminated from the plot
                       to save space.
        sources: None or list of strings. If None, plots all the sources in the CSV file.
        figsize: passed to plt.figure(). if needed, can make it larger. default (12, 5)
        plot_title: If None, uses ROI name. Pass a string for custom name ('' for nothing).
    """
    df = pd.read_csv(csv_file)
    if sources is None:
        sources = find_sources(df)

    # if no sources by now, just quit
    if not sources:
        print("No point sources found in CSV.")
        return

    # compute some reasonable separation between each curve
    offsets = compute_offsets(df, sources)

    # separate seasons
    segments = find_segments(df, gap_threshold=gap_threshold)
    num_segments = len(segments)
    segment_durations = [end - start for start, end in segments]
    legend_loc = np.argmax(segment_durations)
    total_duration = sum(segment_durations) if segment_durations else 1.0
    # width of the subplots proportional to lengths of seasons
    width_ratios = [(dur / total_duration) for dur in segment_durations]

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, num_segments, width_ratios=width_ratios, figure=fig)

    # build the structure of the plot
    axs = []
    ax0 = fig.add_subplot(gs[0])
    axs.append(ax0)
    for i in range(1, len(segments)):
        ax = fig.add_subplot(gs[i], sharey=ax0)  # share y-axis with the first subplot
        axs.append(ax)
        plt.setp(ax.get_yticklabels(), visible=False)  # hide y-axis labels for shared axes

    # sort sources by brightness (lowest median => brightest)
    sorted_sources = sorted(sources, key=lambda ssource: np.median(df[f"{ssource}_mag"].dropna()))

    # we'll keep track of all the plotted values for setting the scale at the end:
    all_mags = []
    for i, ((start_mjd, end_mjd), width_ratio, ax) in enumerate(zip(segments, width_ratios, axs)):

        mask = (df['mjd'] >= start_mjd) & (df['mjd'] <= end_mjd)
        segment_data = df[mask]

        for j, s in enumerate(sorted_sources):
            color = COLOR_CYCLE[j % len(COLOR_CYCLE)]
            mag = segment_data[f"{s}_mag"]
            mag_up = segment_data[f"{s}_d_mag_up"]
            mag_down = segment_data[f"{s}_d_mag_down"]
            # average with scatter if available:
            if (up_scatter := f"{s}_scatter_mag_up") in segment_data.columns:
                mag_up += segment_data[up_scatter]
                mag_up *= 0.5
            if (down_scatter := f"{s}_scatter_mag_down") in segment_data.columns:
                mag_down += segment_data[down_scatter]
                mag_down *= 0.5
            mjd_vals = segment_data["mjd"]

            # shift by offset
            y_shifted = mag + offsets[s]
            all_mags.extend(y_shifted.tolist())

            ax.errorbar(
                mjd_vals, y_shifted,
                yerr=[mag_down, mag_up],
                fmt='o', ms=3, color=color, ecolor=color, alpha=0.7,
                elinewidth=0.2
            )

        # formatting the axis:
        ax.tick_params(direction='in', which='both', top=True)
        if num_segments == 1:
            ax.tick_params(right=True)
            ax.set_ylabel('magnitude')

        if num_segments > 1:
            ax.yaxis.set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:  # first subplot
                ax.spines['left'].set_visible(True)
                ax.tick_params(axis='y', which='both', left=True)
                ax.yaxis.set_visible(True)
                ax.set_ylabel('mag')
                if num_segments > 1:
                    add_break_indicator(ax, width_ratio, left=False, right=True)
            elif i == len(axs) - 1:  # last subplot
                add_break_indicator(ax, width_ratio, left=True, right=False)
                ax.spines['right'].set_visible(True)
                ax.tick_params(axis='y', which='both', right=True, labelright=False, left=False)
                ax.yaxis.set_visible(True)
            else:  # bulk subplot
                add_break_indicator(ax, width_ratio, left=True, right=True)

            if width_ratio < 0.4:
                xloc = plt.MaxNLocator(2)
                ax.xaxis.set_major_locator(xloc)
        ax.set_xlabel("MJD")
        SEASONPAD = 20
        ax.set_xlim([start_mjd - SEASONPAD, end_mjd + SEASONPAD])
        if i == 0:
            # first subplot only, y-axis label
            ax.set_ylabel("Magnitude")
        ax.tick_params(axis='x', rotation=-30)
        ax.set_xlabel('')

        dates = Time((start_mjd, end_mjd), format='mjd').to_datetime()
        start_str = dates[0].strftime("%Y.%m")
        end_str = dates[1].strftime("%Y.%m")
        if width_ratio < 0.08:
            date_str = start_str
        elif width_ratio < 0.15 and start_str[:4] == end_str[:4]:
            date_str = start_str + r'$\,\to\,$' + end_str[-2:]
        elif width_ratio < 0.15:
            date_str = start_str
        else:
            date_str = start_str + r'$\,\to\,$' + end_str
        uptext = date_str

        # now we're changing the top spine of the x-axis to show dates instead of MJDs.
        ax.text(0.5, 1.005, uptext, transform=ax.transAxes, ha='center', va='bottom', fontsize=8)

    legend_handles = []
    for j, label in enumerate(sorted_sources):
        color = COLOR_CYCLE[j % len(COLOR_CYCLE)]
        if (magoff := offsets[label]) > 0:
            ll = f"{label} (+{magoff:>3.01f} mag)"
        elif (magoff := offsets[label]) < 0:
            ll = f"{label} ({magoff:>5.01f} mag)"
        else:
            ll = label
        legend_handles.append(Patch(facecolor=color, label=ll))
    legend = axs[legend_loc].legend(handles=legend_handles,
                                    loc='upper right',
                                    frameon=False,
                                    title=plot_title)
    plt.setp(legend.get_title())

    # setting the scale:
    y_low_mag = np.nanpercentile(all_mags, 1)
    y_high_mag = np.nanpercentile(all_mags, 100.0)
    # add 15% at the top for legend
    mag_range = y_high_mag - y_low_mag
    margin = 0.15 * mag_range
    ax0.set_ylim((y_high_mag, y_low_mag - margin))

    # final adjustments
    plt.subplots_adjust(wspace=0.1, bottom=0.00)
    fig.text(0.51, -0.1, 'MJD', ha='center', va='center')
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    # hopefully it'll look okay.
