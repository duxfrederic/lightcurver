import yaml
from importlib import resources
from collections import deque
import logging
from datetime import datetime


from ..structure.user_config import get_user_config
from ..structure.database import initialize_database
from ..processes.cutout_making import extract_all_stamps
from ..processes.star_querying import query_gaia_stars
from ..processes.psf_modelling import model_all_psfs
from ..processes.star_photometry import do_star_photometry
from ..processes.normalization_calculation import calculate_coefficient
from ..processes.roi_deconv_file_preparation import prepare_roi_deconv_file
from ..processes.roi_modelling import do_deconvolution_of_roi
from ..processes.alternate_plate_solving_with_gaia import alternate_plate_solve
from ..utilities.zeropoint_from_gaia import calculate_zeropoints
from ..structure.exceptions import TaskWasNotSuccessful
from .task_wrappers import (read_convert_skysub_character_catalog,
                            plate_solve_all_frames, calc_common_and_total_footprint_and_save)
from .state_checkers import check_plate_solving


def setup_base_logger():
    time_now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    user_config = get_user_config()
    log_dir = user_config['workdir'] / 'logs'
    log_file_path = str(log_dir / f"{time_now}.log")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    base_logger = logging.getLogger("lightcurver")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    base_logger.addHandler(file_handler)
    base_logger.setLevel(logging.INFO)


class WorkflowManager:
    """
    A small class that will run our tasks. It serves the purpose of running the tasks
    in the right order given the dependencies in the pipeline_dependency_graph.yaml file.
    We can also use it to assign different versions of the tasks based on the user config.

    todo: implement post checks for each tasks
    todo: implement pre checks so we can skip tasks from the workflow manager? for now pre-checks are done within each
          task.
    """
    def __init__(self, logger=None):
        self.user_config = get_user_config()
        with resources.open_text('lightcurver.pipeline', 'pipeline_dependency_graph.yaml') as file:
            self.pipe_config = yaml.safe_load(file)
        self.task_graph = {}
        self.build_dependency_graph()
        # some tasks can be done in multiple ways, let's define this here
        if self.user_config['plate_solving_strategy'] == 'plate_solve':
            plate_solve_function = plate_solve_all_frames
        elif self.user_config['plate_solving_strategy'] == 'alternate_gaia_solve':
            plate_solve_function = alternate_plate_solve
        else:
            raise AssertionError("The config's plate_solving_strategy should be plate_solve or alternate_gaia_solve")
        self.task_attribution = {
            'initialize_database': initialize_database,
            'read_convert_skysub_character_catalog': read_convert_skysub_character_catalog,
            'plate_solving': plate_solve_function,
            'calculate_common_and_total_footprint': calc_common_and_total_footprint_and_save,
            'query_gaia_for_stars': query_gaia_stars,
            'stamp_extraction': extract_all_stamps,
            'psf_modeling': model_all_psfs,
            'star_photometry': do_star_photometry,
            'calculate_normalization_coefficient': calculate_coefficient,
            'calculate_absolute_zeropoints': calculate_zeropoints,
            'prepare_calibrated_cutouts': prepare_roi_deconv_file,
            'model_calibrated_cutouts': do_deconvolution_of_roi,
        }

        self.post_task_attribution = {
            'plate_solving': check_plate_solving
        }
        assert set(self.task_attribution.keys()) == set([entry['name'] for entry in self.pipe_config['tasks']])

        if logger is None:
            logger = logging.getLogger(__name__)
            setup_base_logger()
        self.logger = logger

    def build_dependency_graph(self):
        for task in self.pipe_config['tasks']:
            task_name = task['name']
            self.task_graph[task_name] = {'dependencies': set(task['dependencies']), 'next': []}
            for dep in task['dependencies']:
                if dep in self.task_graph:
                    self.task_graph[dep]['next'].append(task_name)
                else:
                    self.task_graph[dep] = {'dependencies': set(), 'next': [task_name]}

    def topological_sort(self):
        """
            Getting the tasks in the right order, just in case we have multiple dependencies in the future.

            Returns: list of tasks in the right order.
        """
        #
        in_degree = {task: 0 for task in self.task_graph}
        for task in self.task_graph:
            for next_task in self.task_graph[task]['next']:
                in_degree[next_task] += 1

        queue = deque([task for task in in_degree if in_degree[task] == 0])
        sorted_tasks = []

        while queue:
            task = queue.popleft()
            sorted_tasks.append(task)
            for next_task in self.task_graph[task]['next']:
                in_degree[next_task] -= 1
                if in_degree[next_task] == 0:
                    queue.append(next_task)

        if len(sorted_tasks) != len(self.task_graph):
            raise Exception("A cycle was detected in the task dependencies, or a task is missing.")

        return sorted_tasks

    def run(self):
        self.logger.info(f"Workflow manager: will run all tasks. Working directory:  {self.user_config['workdir']}")
        sorted_tasks = self.topological_sort()
        for task_name in sorted_tasks:
            task = next((item for item in self.pipe_config['tasks'] if item['name'] == task_name), None)
            if task:
                self.execute_task(task)
            post_check = self.post_task_attribution.get(task_name, None)
            if post_check:
                success, message = post_check()
                if not success:
                    self.logger.error(
                        f'Post-check failed for {task_name}. Stopping pipeline with message: {message}'
                    )
                    raise TaskWasNotSuccessful(message)
                else:
                    self.logger.info(f'Post-check successful for task {task_name}, with message: {message}')

    def execute_task(self, task):
        # Assume task_func is a callable for simplicity
        task_func = self.task_attribution.get(task['name'])
        self.logger.info(
            f"Running task {task['name']}. Working directory:  {self.user_config['workdir']}"
        )
        task_func()

    def get_tasks(self):
        return sorted(list(self.task_attribution.keys()))



