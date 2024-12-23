import argparse
import os
import yaml
from importlib import resources

from lightcurver.pipeline.workflow_manager import WorkflowManager


def run():
    # loading the possible tasks from the yaml file defining the pipeline.
    with resources.open_text('lightcurver.pipeline', 'pipeline_dependency_graph.yaml') as file:
        pipe_config = yaml.safe_load(file)
    task_list = [task['name'] for task in pipe_config['tasks']]

    docstring = f"""
    Run the lightcurver pipeline.

    The pipeline can be run in entirety or from a specific start point to a stop point.

    Arguments:
    - `--start`: The name of the step to begin execution from.
    - `--stop`: The name of the step to stop execution at.

    Examples:
    1. Run the entire pipeline:
       `python run.py config.yaml`

    2. Run from a specific step:
       `python run.py config.yaml --start plate_solving`

    3. Run up to a specific step:
       `python run.py config.yaml --stop star_photometry`

    4. Run from a start step to a stop step:
       `python run.py config.yaml --start plate_solving --stop star_photometry`

    List of tasks:
    {', '.join(task_list)}
    """

    parser = argparse.ArgumentParser(description=docstring, formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('config_file', type=str,
                        help="The path to the config.yaml configuration file.")

    parser.add_argument('--start', type=str,
                        help="Name of the step to start the pipeline from. Default: start of pipeline.", default=None)

    parser.add_argument('--stop', type=str,
                        help="Name of the step to stop the pipeline at. Default: end of pipeline.",
                        default=None)

    args = parser.parse_args()

    os.environ['LIGHTCURVER_CONFIG'] = args.config_file
    wf_manager = WorkflowManager()
    wf_manager.run(start_step=args.start, stop_step=args.stop)
