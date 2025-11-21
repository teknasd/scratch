"""
Environment configuration for debugging and profiling.

Set DEBUG = True to enable profiling decorators.
Set DEBUG = False for production/normal operation.
"""

import os
import os
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from local.env file
ENVS = load_dotenv('scratch/local.env')
logger.info(ENVS)
DEBUG = os.getenv('DEBUG')