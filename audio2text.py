import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from typing import Iterable, List, Optional, Union
import torch.multiprocessing as mp
import torch
from ctypes import c_bool
from openwakeword.model import Model
from scipy.signal import resample
from scipy import signal
import signal as system_signal
import faster_whisper
import openwakeword
import collections
import numpy as np
import pvporcupine
import traceback
import threading
import webrtcvad
import itertools
import datetime
import platform
import pyaudio
import logging
import struct
import base64
import queue
import halo
import time
import copy
import os
import re
import gc

TIME_SLEEP = 0.02
class TranscriptionWorker:
	def __init__(self, conn, stdout_pipe, model_path, compute_type, gpu_device_index, device,
				 ready_event, shutdown_event, interrupt_stop_event, beam_size, initial_prompt, suppress_tokens):
		self.conn = conn
		self.stdout_pipe = stdout_pipe
		self.model_path = model_path
		self.compute_type = compute_type
		self.gpu_device_index = gpu_device_index
		self.device = device
		self.ready_event = ready_event
		self.shutdown_event = shutdown_event
		self.interrupt_stop_event = interrupt_stop_event
		self.beam_size = beam_size
		self.initial_prompt = initial_prompt
		self.suppress_tokens = suppress_tokens
		self.queue = queue.Queue()

	def custom_print(self, *args, **kwargs):
		message = ' '.join(map(str, args))
		try:
			self.stdout_pipe.send(message)
		except (BrokenPipeError, EOFError, OSError):
			pass

	def poll_connection(self):
		while not self.shutdown_event.is_set():
			if self.conn.poll(0.01):    #This will return a boolean as to whether there is data to be received and read from the pipe
				try:
					data = self.conn.recv()
					self.queue.put(data)
				except Exception as e:
					logging.error(f"Error receiving data from connection: {e}")
			else:
				time.sleep(TIME_SLEEP)

	def run(self):
		if __name__ == "__main__":
			 system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
			 __builtins__['print'] = self.custom_print

		logging.info(f"Initializing faster_whisper main transcription model {self.model_path}")

		try:
			model = faster_whisper.WhisperModel(
				model_size_or_path=self.model_path,
				device=self.device,
				compute_type=self.compute_type,
				device_index=self.gpu_device_index,
			)
		except Exception as e:
			logging.exception(f"Error initializing main faster_whisper transcription model: {e}")
			raise

		self.ready_event.set()
		logging.debug("Faster_whisper main speech to text transcription model initialized successfully")

		# Start the polling thread
		polling_thread = threading.Thread(target=self.poll_connection)
		polling_thread.start()

		try:
			while not self.shutdown_event.is_set():
				try:
					audio, language = self.queue.get(timeout=0.1)
					try:
						segments, info = model.transcribe(
							audio,
							language=language if language else None,
							beam_size=self.beam_size,
							initial_prompt=self.initial_prompt,
							suppress_tokens=self.suppress_tokens
						)
						transcription = " ".join(seg.text for seg in segments).strip()
						logging.debug(f"Final text detected with main model: {transcription}")
						self.conn.send(('success', (transcription, info)))
					except Exception as e:
						logging.error(f"General error in transcription: {e}")
						self.conn.send(('error', str(e)))
				except queue.Empty:
					continue
				except KeyboardInterrupt:
					self.interrupt_stop_event.set()
					logging.debug("Transcription worker process finished due to KeyboardInterrupt")
					break
				except Exception as e:
					logging.error(f"General error in processing queue item: {e}")
		finally:
			__builtins__['print'] = print  # Restore the original print function
			self.conn.close()
			self.stdout_pipe.close()
			self.shutdown_event.set()  # Ensure the polling thread will stop
			polling_thread.join()  # Wait for the polling thread to finish

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

INIT_HANDLE_BUFFER_OVERFLOW = False
if platform.system() != 'Darwin':
    INIT_HANDLE_BUFFER_OVERFLOW = True
class AudioToTextRecorder:
	def __init__(self,
				 model: str = "./weights/faster-whisper-medium",
				 language: str = "vi",
				 compute_type: str = "default",
				 input_device_index: int = None,
				 gpu_device_index: Union[int, List[int]] = 0,
				 device: str = "cuda",
				 on_recording_start=None,
				 on_recording_stop=None,
				 on_transcription_start=None,
				 use_microphone=True,
				 spinner=True,
				 level=logging.WARNING,

				 # Voice activation parameters
				 silero_sensitivity: float = INIT_SILERO_SENSITIVITY,
				 silero_use_onnx: bool = True,
				 silero_deactivity_detection: bool = False,
				 webrtc_sensitivity: int = INIT_WEBRTC_SENSITIVITY,
				 on_vad_detect_start=None,
				 on_vad_detect_stop=None,
				 on_transcription_start=None,
				 min_gap_between_recordings: float = (INIT_MIN_GAP_BETWEEN_RECORDINGS),

				 # Wake word parameters
				 wakeword_backend: str = "pvporcupine",
				 openwakeword_model_paths: str = "./weights/alexa_v0.1.onnx",
				 openwakeword_inference_framework: str = "onnx",
				 wake_words: str = "",
				 wake_words_sensitivity: float = INIT_WAKE_WORDS_SENSITIVITY,
				 wake_word_activation_delay: float = (INIT_WAKE_WORD_ACTIVATION_DELAY),
				 wake_word_timeout: float = INIT_WAKE_WORD_TIMEOUT,
				 wake_word_buffer_duration: float = INIT_WAKE_WORD_BUFFER_DURATION,
				 on_wakeword_timeout=None,
				 on_wakeword_detection_start=None,
				 on_wakeword_detection_end=None,
				 debug_mode=False,

				 audio_queue = mp.Queue(),
				 sample_rate: int = SAMPLE_RATE,
				 buffer_size: int = BUFFER_SIZE,
				 
				 beam_size: int = 5,
				 initial_prompt: Optional[Union[str, Iterable[int]]] = None,
				 suppress_tokens: Optional[List[int]] = [-1],
				 no_log_file: bool = False,
				 handle_buffer_overflow: bool = INIT_HANDLE_BUFFER_OVERFLOW,
				 allowed_latency_limit: int = ALLOWED_LATENCY_LIMIT,
				 ):
		self.language = language
		self.compute_type = compute_type
		self.input_device_index = input_device_index
		self.gpu_device_index = gpu_device_index
		self.device = device
		self.wake_words = wake_words
		self.wake_word_timeout = wake_word_timeout
		self.wake_word_activation_delay = wake_word_activation_delay
		self.wake_word_buffer_duration = wake_word_buffer_duration
		self.use_microphone = mp.Value(c_bool, use_microphone)
		self.on_wakeword_timeout = on_wakeword_timeout
		self.on_vad_detect_start = on_vad_detect_start
		self.on_vad_detect_stop = on_vad_detect_stop
		self.on_wakeword_detection_start = on_wakeword_detection_start
		self.on_wakeword_detection_end = on_wakeword_detection_end
		self.on_transcription_start = on_transcription_start
		self.debug_mode=debug_mode
		self.beam_size = beam_size

		self.audio_queue = mp.Queue()
		self.buffer_size = buffer_size
		self.sample_rate = sample_rate
		self.recording_stop_time = 0
		self.wake_word_detect_time = 0
		self.silero_working = False
		self.silero_sensitivity = silero_sensitivity
		self.silero_deactivity_detection = silero_deactivity_detection
		self.initial_prompt = initial_prompt
		self.suppress_tokens = suppress_tokens
		self.use_wake_words = wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}
		self.use_extended_logging = use_extended_logging
		self.min_gap_between_recordings = min_gap_between_recordings

		self.handle_buffer_overflow = handle_buffer_overflow
		self.allowed_latency_limit = allowed_latency_limit
		self.listen_start = 0
		self.wakeword_detected = False
		self.state = "inactive"
		self.halo = None
		self.spinner = spinner
		self.is_webrtc_speech_active = False
        self.is_silero_speech_active = False
        self.text_storage = []
        self.realtime_stabilized_text = ""
        self.realtime_stabilized_safetext = ""
        self.frames = []
        self.start_recording_event = threading.Event()
        self.stop_recording_event = threading.Event()
        self.on_recording_start = on_recording_start
		#--------log-------
		# Initialize the logging configuration with the specified level
		log_format = 'RealTimeSTT: %(name)s - %(levelname)s - %(message)s'

		# Adjust file_log_format to include milliseconds
		file_log_format = '%(asctime)s.%(msecs)03d - ' + log_format

		# Get the root logger
		logger = logging.getLogger()
		logger.setLevel(logging.DEBUG)  # Set the root logger's level to DEBUG

		# Remove any existing handlers
		logger.handlers = []

		# Create a console handler and set its level
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level) 
		console_handler.setFormatter(logging.Formatter(log_format))

		# Add the handlers to the logger
		if not no_log_file:
			# Create a file handler and set its level
			file_handler = logging.FileHandler('realtimesst.log')
			file_handler.setLevel(logging.DEBUG)
			file_handler.setFormatter(logging.Formatter(
				file_log_format,
				datefmt='%Y-%m-%d %H:%M:%S'
			))

			logger.addHandler(file_handler)
		logger.addHandler(console_handler)
		#//////////////////
		self.is_shut_down = False
		self.shutdown_event = mp.Event()

		try:
			# Only set the start method if it hasn't been set already
			if mp.get_start_method(allow_none=True) is None:
				mp.set_start_method("spawn")
		except RuntimeError as e:
			logging.info(f"Start method has already been set. Details: {e}")

		logging.info("Starting STT")

		if use_extended_logging:
			logging.info("STT was called with these parameters:")
			for param, value in locals().items():
				logging.info(f"{param}: {value}")

		self.interrupt_stop_event = mp.Event()
		self.was_interrupted = mp.Event()
		self.main_transcription_ready_event = mp.Event()
		self.parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
		self.parent_stdout_pipe, child_stdout_pipe = mp.Pipe()

		# Set device for model
		self.device = "cuda" if self.device == "cuda" and torch.cuda.is_available() else "cpu"

		self.transcript_process = self._start_thread(
			target=AudioToTextRecorder._transcription_worker,
			args=(
				child_transcription_pipe,
				child_stdout_pipe,
				model,
				self.compute_type,
				self.gpu_device_index,
				self.device,
				self.main_transcription_ready_event,
				self.shutdown_event,
				self.interrupt_stop_event,
				self.beam_size,
				self.initial_prompt,
				self.suppress_tokens
			)
		)

		# Start audio data reading process
		if self.use_microphone.value:
			logging.info(
				f"Initializing audio recording (creating pyAudio input stream, sample rate: {self.sample_rate} buffer size: {self.buffer_size}"
			)
			self.reader_process = self._start_thread(
				target=AudioToTextRecorder._audio_data_worker,
				args=(
					self.audio_queue,
					self.sample_rate,
					self.buffer_size,
					self.input_device_index,
					self.shutdown_event,
					self.interrupt_stop_event,
					self.use_microphone
				)
			)

		# Setup wake word detection
		if wake_words or wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
			self.wakeword_backend = wakeword_backend

			self.wake_words_list = [
				word.strip() for word in wake_words.lower().split(',')
			]
			self.wake_words_sensitivity = wake_words_sensitivity
			self.wake_words_sensitivities = [
				float(wake_words_sensitivity)
				for _ in range(len(self.wake_words_list))
			]

			if self.wakeword_backend in {'pvp', 'pvporcupine'}:

				try:
					self.porcupine = pvporcupine.create(
						keywords=self.wake_words_list,
						sensitivities=self.wake_words_sensitivities
					)
					self.buffer_size = self.porcupine.frame_length
					self.sample_rate = self.porcupine.sample_rate

				except Exception as e:
					logging.exception(
						f"Error initializing porcupine wake word detection engine: {e}"
					)
					raise

				logging.debug(
					"Porcupine wake word detection engine initialized successfully"
				)

			elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
					
				# openwakeword.utils.download_models()

				try:
					if openwakeword_model_paths:
						model_paths = openwakeword_model_paths.split(',')
						self.owwModel = Model(
							wakeword_models=model_paths,
							inference_framework=openwakeword_inference_framework
						)
						logging.info(
							f"Successfully loaded wakeword model(s): {openwakeword_model_paths}"
						)
					else:
						self.owwModel = Model(
							inference_framework=openwakeword_inference_framework)
					
					self.oww_n_models = len(self.owwModel.models.keys())
					if not self.oww_n_models:
						logging.error(
							"No wake word models loaded."
						)

					for model_key in self.owwModel.models.keys():
						logging.info(
							f"Successfully loaded openwakeword model: {model_key}"
						)

				except Exception as e:
					logging.exception(
						f"Error initializing openwakeword wake word detection engine: {e}"
					)
					raise

				logging.debug(
					"Open wake word detection engine initialized successfully"
				)
			
			else:
				logging.exception(
					f"Wakeword engine {self.wakeword_backend} unknown/unsupported. Please specify one of: pvporcupine, openwakeword."
				)

		# Setup voice activity detection model WebRTC
		try:
			logging.info(
				f"Initializing WebRTC voice with Sensitivity {webrtc_sensitivity}"
			)
			self.webrtc_vad_model = webrtcvad.Vad()
			self.webrtc_vad_model.set_mode(webrtc_sensitivity)

		except Exception as e:
			logging.exception(
				f"Error initializing WebRTC voice activity detection engine: {e}"
			)
			raise

		logging.debug(
			"WebRTC VAD voice activity detection engine initialized successfully"
		)

		# Setup voice activity detection model Silero VAD
		try:
			self.silero_vad_model, _ = torch.hub.load(
				repo_or_dir="./weights/silero_vad.onnx",
				model="silero_vad",
				verbose=False,
				onnx=silero_use_onnx
			)

		except Exception as e:
			logging.exception(
				f"Error initializing Silero VAD voice activity detection engine: {e}"
			)
			raise

		logging.debug(
			"Silero VAD voice activity detection engine initialized successfully"
		)

		self.audio_buffer = collections.deque(
			maxlen=int((self.sample_rate // self.buffer_size) *
					   self.pre_recording_buffer_duration)
		)
		self.last_words_buffer = collections.deque(
			maxlen=int((self.sample_rate // self.buffer_size) *
					   0.3)
		)
		self.frames = []

		# Recording control flags
		self.is_recording = False
		self.is_running = True
		self.start_recording_on_voice_activity = False
		self.stop_recording_on_voice_deactivity = False

		# Start the recording worker thread
		self.recording_thread = threading.Thread(target=self._recording_worker)
		self.recording_thread.daemon = True
		self.recording_thread.start()

		# Wait for transcription models to start
		logging.debug('Waiting for main transcription model to start')
		self.main_transcription_ready_event.wait()
		logging.debug('Main transcription model ready')

		self.stdout_thread = threading.Thread(target=self._read_stdout)
		self.stdout_thread.daemon = True
		self.stdout_thread.start()

		logging.debug('RealtimeSTT initialization completed successfully')

	def _start_thread(self, target=None, args=()):
		"""
		Implement a consistent threading model across the library.

		This method is used to start any thread in this library. It uses the
		standard threading. Thread for Linux and for all others uses the pytorch
		MultiProcessing library 'Process'.
		Args:
			target (callable object): is the callable object to be invoked by
			  the run() method. Defaults to None, meaning nothing is called.
			args (tuple): is a list or tuple of arguments for the target
			  invocation. Defaults to ().
		"""
		if (platform.system() == 'Linux'):
			thread = threading.Thread(target=target, args=args)
			thread.deamon = True
			thread.start()
			return thread
		else:
			thread = mp.Process(target=target, args=args)
			thread.start()
			return thread

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

	def _transcription_worker(*args, **kwargs):
		worker = TranscriptionWorker(*args, **kwargs)
		worker.run()

	@staticmethod
	def _audio_data_worker(audio_queue,
						target_sample_rate,
						buffer_size,
						input_device_index,
						shutdown_event,
						interrupt_stop_event,
						use_microphone):
		"""
		Worker method that handles the audio recording process.

		This method runs in a separate process and is responsible for:
		- Setting up the audio input stream for recording at the highest possible sample rate.
		- Continuously reading audio data from the input stream, resampling if necessary,
		preprocessing the data, and placing complete chunks in a queue.
		- Handling errors during the recording process.
		- Gracefully terminating the recording process when a shutdown event is set.

		Args:
			audio_queue (queue.Queue): A queue where recorded audio data is placed.
			target_sample_rate (int): The desired sample rate for the output audio (for Silero VAD).
			buffer_size (int): The number of samples expected by the Silero VAD model.
			input_device_index (int): The index of the audio input device.
			shutdown_event (threading.Event): An event that, when set, signals this worker method to terminate.
			interrupt_stop_event (threading.Event): An event to signal keyboard interrupt.
			use_microphone (multiprocessing.Value): A shared value indicating whether to use the microphone.

		Raises:
			Exception: If there is an error while initializing the audio recording.
		"""
		import pyaudio
		import numpy as np
		from scipy import signal
		
		if __name__ == '__main__':
			system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

		def get_highest_sample_rate(audio_interface, device_index):
			"""Get the highest supported sample rate for the specified device."""
			try:
				device_info = audio_interface.get_device_info_by_index(device_index)
				max_rate = int(device_info['defaultSampleRate'])
				
				if 'supportedSampleRates' in device_info:
					supported_rates = [int(rate) for rate in device_info['supportedSampleRates']]
					if supported_rates:
						max_rate = max(supported_rates)
				
				return max_rate
			except Exception as e:
				logging.warning(f"Failed to get highest sample rate: {e}")
				return 48000  # Fallback to a common high sample rate

		def initialize_audio_stream(audio_interface, sample_rate, chunk_size):
			nonlocal input_device_index

			def validate_device(device_index):
				"""Validate that the device exists and is actually available for input."""
				try:
					device_info = audio_interface.get_device_info_by_index(device_index)
					if not device_info.get('maxInputChannels', 0) > 0:
						return False

					# Try to actually read from the device
					test_stream = audio_interface.open(
						format=pyaudio.paInt16,
						channels=1,
						rate=target_sample_rate,
						input=True,
						frames_per_buffer=chunk_size,
						input_device_index=device_index,
						start=False  # Don't start the stream yet
					)

					# Start the stream and try to read from it
					test_stream.start_stream()
					test_data = test_stream.read(chunk_size, exception_on_overflow=False)
					test_stream.stop_stream()
					test_stream.close()

					# Check if we got valid data
					if len(test_data) == 0:
						return False

					return True

				except Exception as e:
					logging.debug(f"Device validation failed: {e}")
					return False

			"""Initialize the audio stream with error handling."""
			while not shutdown_event.is_set():
				try:
					# First, get a list of all available input devices
					input_devices = []
					for i in range(audio_interface.get_device_count()):
						try:
							device_info = audio_interface.get_device_info_by_index(i)
							if device_info.get('maxInputChannels', 0) > 0:
								input_devices.append(i)
						except Exception:
							continue

					if not input_devices:
						raise Exception("No input devices found")

					# If input_device_index is None or invalid, try to find a working device
					if input_device_index is None or input_device_index not in input_devices:
						# First try the default device
						try:
							default_device = audio_interface.get_default_input_device_info()
							if validate_device(default_device['index']):
								input_device_index = default_device['index']
						except Exception:
							# If default device fails, try other available input devices
							for device_index in input_devices:
								if validate_device(device_index):
									input_device_index = device_index
									break
							else:
								raise Exception("No working input devices found")

					# Validate the selected device one final time
					if not validate_device(input_device_index):
						raise Exception("Selected device validation failed")

					# If we get here, we have a validated device
					stream = audio_interface.open(
						format=pyaudio.paInt16,
						channels=1,
						rate=sample_rate,
						input=True,
						frames_per_buffer=chunk_size,
						input_device_index=input_device_index,
					)

					logging.info(f"Microphone connected and validated (input_device_index: {input_device_index})")
					return stream

				except Exception as e:
					logging.error(f"Microphone connection failed: {e}. Retrying...")
					input_device_index = None
					time.sleep(3)  # Wait before retrying
					continue

		def preprocess_audio(chunk, original_sample_rate, target_sample_rate):
			"""Preprocess audio chunk similar to feed_audio method."""
			if isinstance(chunk, np.ndarray):
				# Handle stereo to mono conversion if necessary
				if chunk.ndim == 2:
					chunk = np.mean(chunk, axis=1)

				# Resample to target_sample_rate if necessary
				if original_sample_rate != target_sample_rate:
					num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
					chunk = signal.resample(chunk, num_samples)

				# Ensure data type is int16
				chunk = chunk.astype(np.int16)
			else:
				# If chunk is bytes, convert to numpy array
				chunk = np.frombuffer(chunk, dtype=np.int16)

				# Resample if necessary
				if original_sample_rate != target_sample_rate:
					num_samples = int(len(chunk) * target_sample_rate / original_sample_rate)
					chunk = signal.resample(chunk, num_samples)
					chunk = chunk.astype(np.int16)

			return chunk.tobytes()

		audio_interface = None
		stream = None
		device_sample_rate = None
		chunk_size = 1024  # Increased chunk size for better performance

		def setup_audio():  
			nonlocal audio_interface, stream, device_sample_rate, input_device_index
			try:
				if audio_interface is None:
					audio_interface = pyaudio.PyAudio()
				if input_device_index is None:
					try:
						default_device = audio_interface.get_default_input_device_info()
						input_device_index = default_device['index']
					except OSError as e:
						input_device_index = None

				sample_rates_to_try = [16000]  # Try 16000 Hz first
				if input_device_index is not None:
					highest_rate = get_highest_sample_rate(audio_interface, input_device_index)
					if highest_rate != 16000:
						sample_rates_to_try.append(highest_rate)
				else:
					sample_rates_to_try.append(48000)  # Fallback sample rate

				for rate in sample_rates_to_try:
					try:
						device_sample_rate = rate
						stream = initialize_audio_stream(audio_interface, device_sample_rate, chunk_size)
						if stream is not None:
							logging.debug(f"Audio recording initialized successfully at {device_sample_rate} Hz, reading {chunk_size} frames at a time")
							# logging.error(f"Audio recording initialized successfully at {device_sample_rate} Hz, reading {chunk_size} frames at a time")
							return True
					except Exception as e:
						logging.warning(f"Failed to initialize audio23 stream at {device_sample_rate} Hz: {e}")
						continue

				# If we reach here, none of the sample rates worked
				raise Exception("Failed to initialize audio stream12 with all sample rates.")

			except Exception as e:
				logging.exception(f"Error initializing pyaudio audio recording: {e}")
				if audio_interface:
					audio_interface.terminate()
				return False

		if not setup_audio():
			raise Exception("Failed to set up audio recording.")

		buffer = bytearray()
		silero_buffer_size = 2 * buffer_size  # silero complains if too short

		time_since_last_buffer_message = 0

		try:
			while not shutdown_event.is_set():
				try:
					data = stream.read(chunk_size, exception_on_overflow=False)
					
					if use_microphone.value:
						processed_data = preprocess_audio(data, device_sample_rate, target_sample_rate)
						buffer += processed_data

						# Check if the buffer has reached or exceeded the silero_buffer_size
						while len(buffer) >= silero_buffer_size:
							# Extract silero_buffer_size amount of data from the buffer
							to_process = buffer[:silero_buffer_size]
							buffer = buffer[silero_buffer_size:]

							# Feed the extracted data to the audio_queue
							if time_since_last_buffer_message:
								time_passed = time.time() - time_since_last_buffer_message
								if time_passed > 1:
									logging.debug("_audio_data_worker writing audio data into queue.")
									time_since_last_buffer_message = time.time()
							else:
								time_since_last_buffer_message = time.time()

							audio_queue.put(to_process)
							

				except OSError as e:
					if e.errno == pyaudio.paInputOverflowed:
						logging.warning("Input overflowed. Frame dropped.")
					else:
						logging.error(f"OSError during recording: {e}")
						# Attempt to reinitialize the stream
						logging.error("Attempting to reinitialize the audio stream...")

						try:
							if stream:
								stream.stop_stream()
								stream.close()
						except Exception as e:
							pass
						
						# Wait a bit before trying to reinitialize
						time.sleep(1)
						
						if not setup_audio():
							logging.error("Failed to reinitialize audio stream. Exiting.")
							break
						else:
							logging.error("Audio stream reinitialized successfully.")
					continue

				except Exception as e:
					logging.error(f"Unknown error during recording: {e}")
					tb_str = traceback.format_exc()
					logging.error(f"Traceback: {tb_str}")
					logging.error(f"Error: {e}")
					# Attempt to reinitialize the stream
					logging.info("Attempting to reinitialize the audio stream...")
					try:
						if stream:
							stream.stop_stream()
							stream.close()
					except Exception as e:
						pass
					
					# Wait a bit before trying to reinitialize
					time.sleep(1)
					
					if not setup_audio():
						logging.error("Failed to reinitialize audio stream. Exiting.")
						break
					else:
						logging.info("Audio stream reinitialized successfully.")
					continue

		except KeyboardInterrupt:
			interrupt_stop_event.set()
			logging.debug("Audio data worker process finished due to KeyboardInterrupt")
		finally:
			# After recording stops, feed any remaining audio data
			if buffer:
				audio_queue.put(bytes(buffer))
			
			try:
				if stream:
					stream.stop_stream()
					stream.close()
			except Exception as e:
				pass
			if audio_interface:
				audio_interface.terminate()