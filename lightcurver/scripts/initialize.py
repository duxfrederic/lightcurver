import os
import shutil
import argparse
from ruamel.yaml import YAML
from importlib import resources
from pathlib import Path


def copy_template(template_path, target_path):
    shutil.copy(template_path, target_path)


def update_config(config_path):
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(config_path, 'r') as file:
        config = yaml.load(file)

    # Prompt the user for new values
    new_value = input("Enter new value for 'example_key': ")
    config['example_key'] = new_value

    with open(config_path, 'w') as file:
        yaml.dump(config, file)


def initialize():
    parser = argparse.ArgumentParser(description="Initialize the basic configuration for lightcurver.")
    parser.add_argument('--workdir', type=str, help="The path to the desired working directory.",
                        default=".")
    parser.add_argument('--roi_name', type=str, help="Name of the region of interest.",
                        default=None)
    parser.add_argument('--roi_ra', type=float, help="R.A. in degrees of the region of interest.",
                        default=None)
    parser.add_argument('--roi_dec', type=float, help="Dec. in degrees of the region of interest.",
                        default=None)
    parser.add_argument('--photom_band', type=str, help="Photometric filter of the observations.",
                        default=None)
    args = parser.parse_args()
    workdir = Path(args.workdir).absolute()
    workdir.mkdir(exist_ok=True)
    print(f'Initializing working directory at {workdir}')
    # template config file in the installed package
    with resources.open_text('lightcurver.pipeline.example_config_file', 'config.yaml') as ff:
        config_path = workdir / 'config.yaml'
        with open(config_path, 'w') as new_file:
            new_file.write(ff.read())

    # header parser directory
    parser_dir = workdir / 'header_parser'
    parser_dir.mkdir(exist_ok=True)
    parser_file = parser_dir / 'parse_header.py'
    with open(parser_file, 'w') as ff:
        ff.write(f"""
def parse_header(header):
    raise RuntimeError('Adjust the header parser function at {parser_file}')
    # example:
    from dateutil import parser
    from astropy.time import Time
    exptime = header['exptime']
    gain = header['gain']
    time = Time(parser.parse(header['obstart']))
    return {{'exptime': exptime, 'gain': gain, 'mjd': time.mjd}}

""")

    # adjusting config file
    yaml = YAML()
    yaml.preserve_quotes = True

    with open(config_path, 'r') as file:
        config = yaml.load(file)
    config['workdir'] = str(workdir)
    if args.roi_name is None:
        args.roi_name = input("Name of the target? ").strip()
    if args.roi_ra is None:
        args.roi_ra = float(input("Right ascension of the target? "))
    if args.roi_dec is None:
        args.roi_dec = float(input("Declination of the target? "))
    config["ROI"] = {args.roi_name: {'coordinates': [args.roi_ra, args.roi_dec]}}

    if args.photom_band is None:
        config['photometric_band'] = input('Photometric band of the observations? ').strip()

    with open(config_path, 'w') as file:
        yaml.dump(config, file)
    print(f"Adapt the header parser at {parser_file}.")
    print(f"Prepared rough configuration at {config_path} -- go through it and refine it.")

