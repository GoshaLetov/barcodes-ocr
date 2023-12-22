import os
from pathlib import Path

PROJECT_PATH = Path(__file__).parent.parent
CONFIGS_PATH = PROJECT_PATH / 'configs'
EXPERIMENTS_PATH = PROJECT_PATH / 'experiments'
DATA_PATH = os.environ.get('DATA_PATH')
