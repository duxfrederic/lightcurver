import yaml
from importlib import resources
from collections import deque
import logging

from ..structure.user_config import get_user_config
from ..structure.database import initialize_database
from .task_definitions import read_convert_skysub_character_catalog


class TaskOutcome:
    def __init__(self):
        self.outcomes = {}

    def set_outcome(self, task_name, outcome_name, value):
        if task_name not in self.outcomes:
            self.outcomes[task_name] = {}
        self.outcomes[task_name][outcome_name] = value

    def check_outcome(self, task_name, outcome_name):
        return self.outcomes.get(task_name, {}).get(outcome_name, False)

    def check_dependencies(self, task):
        dependencies_met = True
        for dependency in task.dependencies:
            pass
        return dependencies_met


class WorkflowManager:
    def __init__(self, logger=None):
        self.user_config = get_user_config()
        with resources.open_text('lightcurver.pipeline', 'pipeline_dependency_graph.yaml') as file:
            self.pipe_config = yaml.safe_load(file)
        self.task_graph = {}
        self.build_dependency_graph()

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
                self.execute_task(task)

    def execute_task(self, task):
        # Assume task_func is a callable for simplicity
        task_func = self.get_task_function(task['name'])
        print(f"Executing task: {task['name']}")
        task_func()

    def get_task_function(self, task_name):
        # This method needs to map task_name to actual function calls
        if task_name == 'read_convert_skysub_character_catalog':
            return read_convert_skysub_character_catalog
        elif task_name == "initialize_database":
            return initialize_database
        return lambda: 1
        pass

