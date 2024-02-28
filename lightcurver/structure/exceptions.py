class NoConfigFilePathInEnvironment(Exception):
    def __init__(self):
        message = """You need to define the path to your configuration file, 
e.g. export LIGHTCURVER_CONFIG="/path/to/user_config.yaml"
Then re-run the pipeline.
"""
        super().__init__(message)
