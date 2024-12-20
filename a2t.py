import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from typing import Iterable, List, Optional, Union
import multiprocessing as mp
from ctypes import c_bool
import collections
import numpy as np
import traceback
import threading
import datetime
import queue
import halo
import time
import os
import re
import gc

from constants import *
from transcription_worker import TranscriptionWorker
from data_worker import DataWorker
from oww_worker import OWWWorker
from vad_worker import VADWorker

class A2T:
	def __init__(self,
				 model_transcription: str = "./weights/faster-whisper-medium",
				 language: str = "vi",
				 use_log_file: bool = True,
				 
				 # ----transcription worker----
				 compute_type: str = "default",
				 gpu_device_index: Union[int, List[int]] = 0,
				 beam_size: int = 5,
				 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
				 suppress_tokens: Optional[List[int]] = [-1],

				 # ----data worker---
				 input_device_index: int = None,
				 sample_rate: int = SAMPLE_RATE,
				 buffer_size: int = BUFFER_SIZE,
				 use_microphone = True,

				 # ----wakeword worker----
                 use_wakeword: bool = True,
                 wake_words: str = "",
                 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
                 wakeword_backend: str = "openwakeword",
                 openwakeword_inference_framework: str = "onnx",
                 openwakeword_model_paths: str = "./weights/chào_mer_quea.onnx",

                 # ----VAD worker----
                 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
				 silero_use_onnx: bool = True,
				 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
				 int16_max_abs_value: float = INT16_MAX_ABS_VALUE,
				 ):
		config_log(use_log_file, log_file_name="a2t.log")
		self.frames = []
		self.language = language
		self.model_transcription = model_transcription
		self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

		# Recording control flags
		self.is_running = True
		self.is_shut_down = False
		self.is_recording = False
		self.start_recording_on_voice_activity = False
		self.stop_recording_on_voice_deactivity = False
		
		logging.info("----Starting A2T---")
		# if use_extended_logging:
		# 	logging.info("A2T was called with these parameters:")
		# 	for param, value in locals().items():
		# 		logging.info(f"{param}: {value}")

		self.shutdown_event = mp.Event()
		# self.was_interrupted = mp.Event()
		self.interrupt_stop_event = mp.Event()
		self.main_transcription_ready_event = mp.Event()
		self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()
		self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()

		self.audio_buffer = collections.deque(
			maxlen=int((self.sample_rate // self.buffer_size) *
					   self.pre_recording_buffer_duration)
		)
		self.last_words_buffer = collections.deque(
			maxlen=int((self.sample_rate // self.buffer_size) *
					   0.3)
		)
		
		# ----Start transcription worker process
		self.compute_type = compute_type
		self.gpu_device_index = gpu_device_index
		self.beam_size = beam_size
		self.initial_prompt = initial_prompt
		self.suppress_tokens = suppress_tokens
		self.tsw = TranscriptionWorker(child_transcription_pipe,
													child_stdout_pipe,
													model_transcription,
													self.compute_type,
													self.gpu_device_index,
													self.device,
													self.main_transcription_ready_event,
													self.shutdown_event,
													self.interrupt_stop_event,
													self.beam_size,
													self.initial_prompt,
													self.suppress_tokens)
		self.tsw.daemon = True
		self.tsw.start()

		# ----Start audio data reading process
		self.audio_queue = mp.Queue()
		self.sample_rate = sample_rate
		self.buffer_size = buffer_size
		self.input_device_index = input_device_index
		self.use_microphone = mp.Value(c_bool, use_microphone)
		logging.info(f"Initializing audio recording (creating pyAudio input stream, sample rate: {self.sample_rate} buffer size: {self.buffer_size}")
		self.dtw = DataWorker(self.audio_queue,
										self.sample_rate,
										self.buffer_size,
										self.input_device_index,
										self.shutdown_event,
										self.interrupt_stop_event,
										self.use_microphone)
		self.dtw.daemon = True
		self.dtw.start()

		# ----Setup wake word detection
		self.wake_words = wake_words
		self.wake_words_sensitivity = wake_words_sensitivity
		self.wakeword_backend = wakeword_backend
		self.openwakeword_inference_framework = openwakeword_inference_framework
		self.openwakeword_model_paths = openwakeword_model_paths
		if use_wakeword:
			self.owww = OWWWorker(self.wake_words,
									self.wake_words_sensitivity,
									self.wakeword_backend,
									self.openwakeword_inference_framework,
									self.openwakeword_model_paths)


		# ----Setup voice activity detection
		self.webrtc_sensitivity = webrtc_sensitivity
		self.silero_use_onnx = silero_use_onnx
		self.silero_sensitivity = silero_sensitivity
		self.int16_max_abs_value = int16_max_abs_value
		self.vadw = VADWorker(self.sample_rate,
						self.webrtc_sensitivity,
						self.silero_use_onnx,
						self.silero_sensitivity,
						self.int16_max_abs_value)

		# Wait for transcription models to start
		logging.debug('Waiting for main transcription model to start')
		self.main_transcription_ready_event.wait()
		logging.debug('Main transcription model ready')

		self.stdout_thread = threading.Thread(target=self._read_stdout)
		self.stdout_thread.daemon = True
		self.stdout_thread.start()

		logging.debug('A2T initialization completed successfully')

	def _read_stdout(self):
		while not self.shutdown_event.is_set():
			try:
				if self.parent_stdout_pipe.poll(0.1):
					logging.debug("Receive from stdout pipe")
					message = self.parent_stdout_pipe.recv()
					logging.info(message)
			except (BrokenPipeError, EOFError, OSError):
				# The pipe probably has been closed, so we ignore the error
				pass
			except KeyboardInterrupt:  # handle manual interruption (Ctrl+C)
				logging.info("KeyboardInterrupt in read from stdout detected, exiting...")
				break
			except Exception as e:
				logging.error(f"Unexpected error in read from stdout: {e}")
				logging.error(traceback.format_exc())  # Log the full traceback here
				break 
			time.sleep(0.1)
