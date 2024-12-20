import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))
from typing import Iterable, List, Optional, Union
import logging
import platform
import multiprocessing as mp

INIT_MIN_GAP_BETWEEN_RECORDINGS = 0
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INIT_WAKE_WORDS_SENSITIVITY = 0.6
INIT_WAKE_WORD_ACTIVATION_DELAY = 0.0
INIT_WAKE_WORD_TIMEOUT = 5.0
INIT_WAKE_WORD_BUFFER_DURATION = 0.1
ALLOWED_LATENCY_LIMIT = 100

SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True

#--------log-------
def config_log(use_log_file: bool=True, log_file_name: str="a2t.log"):
	# Initialize the logging configuration with the specified level
	LOG_FORMAT = 'RealTimeSTT: %(name)s - %(levelname)s - %(message)s'

	# Adjust file_log_format to include milliseconds
	FILE_LOG_FORMAT = '%(asctime)s.%(msecs)03d - ' + LOG_FORMAT

	# Get the root logger
	LOGGER = logging.getLogger()
	LOGGER.setLevel(logging.DEBUG)  # Set the root logger's level to DEBUG

	# Remove any existing handlers
	LOGGER.handlers = []

	# Create a console handler and set its level
	CONSOLE_HANDLER = logging.StreamHandler()
	CONSOLE_HANDLER.setLevel(level) 
	CONSOLE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

	# Add the handlers to the logger
	if use_log_file:
		# Create a file handler and set its level
		FILE_HANDLER = logging.FileHandler(log_file_name)
		FILE_HANDLER.setLevel(logging.DEBUG)
		FILE_HANDLER.setFormatter(logging.Formatter(
			FILE_LOG_FORMAT,
			datefmt='%Y-%m-%d %H:%M:%S'
		))

		LOGGER.addHandler(FILE_HANDLER)
	LOGGER.addHandler(CONSOLE_HANDLER)
	return
#//////////////////

try:
	# Only set the start method if it hasn't been set already
	if mp.get_start_method(allow_none=True) is None:
		mp.set_start_method("spawn")
except RuntimeError as e:
	logging.info(f"Start method has already been set. Details: {e}")