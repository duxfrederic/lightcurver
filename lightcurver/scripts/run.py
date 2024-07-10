import argparse
import os

from lightcurver.pipeline.workflow_manager import WorkflowManager


def run():
    parser = argparse.ArgumentParser(description="Run the lightcurver pipeline.")
    parser.add_argument('config_file', type=str, help="The path to the config.yaml configuration file..")
    args = parser.parse_args()

    os.environ['LIGHTCURVER_CONFIG'] = args.config_file
    wf_manager = WorkflowManager()
    wf_manager.run()
