import yaml
from importlib import resources
from collections import deque
import logging

from ..structure.user_config import get_user_config
from ..structure.database import initialize_database
from .task_wrappers import (read_convert_skysub_character_catalog,
                            plate_solve_all_frames, calc_common_and_total_footprint_and_save)
from ..processes.cutout_making import extract_all_stamps
from ..processes.star_querying import query_gaia_stars
from ..processes.psf_modelling import model_all_psfs
from ..processes.star_photometry import do_star_photometry
from ..processes.normalization_calculation import calculate_coefficient


class WorkflowManager:
    def __init__(self, logger=None):
        self.user_config = get_user_config()
        with resources.open_text('lightcurver.pipeline', 'pipeline_dependency_graph.yaml') as file:
            self.pipe_config = yaml.safe_load(file)
        self.task_graph = {}
        self.build_dependency_graph()

        _tmp_decoy = lambda: 1
        self.task_attribution = {
            'initialize_database': initialize_database,
            'read_convert_skysub_character_catalog': read_convert_skysub_character_catalog,
            'plate_solving': plate_solve_all_frames,
            'calculate_common_and_total_footprint': calc_common_and_total_footprint_and_save,
            'query_gaia_for_stars': query_gaia_stars,
            'stamp_extraction': extract_all_stamps,
            'psf_modeling': model_all_psfs,
            'star_photometry': do_star_photometry,
            'calculate_normalization_coefficient': calculate_coefficient,
            'prepare_calibrated_cutouts': _tmp_decoy,
        }
        assert set(self.task_attribution.keys()) == set([entry['name'] for entry in self.pipe_config['tasks']])

        if logger is None:
            logger = logging.getLogger(__name__)
            if not logger.hasHandlers():
                # Configure logging to print to the standard output
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
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
        sorted_tasks = self.topological_sort()
        for task_name in sorted_tasks:
            task = next((item for item in self.pipe_config['tasks'] if item['name'] == task_name), None)
            if task:
                self.execute_task(task, self.logger)

    def execute_task(self, task, logger):
        # Assume task_func is a callable for simplicity
        task_func = self.task_attribution.get(task['name'])
        print(f"Executing task: {task['name']}")
        task_func()



