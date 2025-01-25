from importlib import resources


def generate_lightcurve_html(df, output_file="lightcurves.html"):
    """
    Generates an interactive light curve visualization HTML file.

    Args:
        df: pandas.DataFrame containing light curve data
        output_file: Path for output HTML file
    """

    csv_data = df.to_csv(index=False, float_format="%.6f")

    template = resources.read_text("lightcurver.plotting", "plot_curves_template.html")
    # inject light curves
    html = template.replace(
        '// CSV_DATA_PLACEHOLDER',
        f'const csvData = `{csv_data}`;'
    )

    with open(output_file, 'w') as f:
        f.write(html)
