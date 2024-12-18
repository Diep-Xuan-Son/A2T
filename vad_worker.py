import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from typing import Iterable, List, Optional, Union
import time
import torch
import logging
import webrtcvad
import threading
import numpy as np
import onnxruntime as ort

class bcolors:
	OKGREEN = '\033[92m'  # Green for active speech detection
	WARNING = '\033[93m'  # Yellow for silence detection
	ENDC = '\033[0m'      # Reset to default color

class OnnxWrapper():
	def __init__(self, path, force_onnx_cpu=False):
		

		opts = ort.SessionOptions()
		opts.inter_op_num_threads = 1
		opts.intra_op_num_threads = 1

		if force_onnx_cpu and 'CPUExecutionProvider' in ort.get_available_providers():
			self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'], sess_options=opts)
		else:
			self.session = ort.InferenceSession(path, sess_options=opts)

		self.reset_states()
		if '16k' in path:
			warnings.warn('This model support only 16000 sampling rate!')
			self.sample_rates = [16000]
		else:
			self.sample_rates = [8000, 16000]

	def _validate_input(self, x, sr: int):
		if x.ndim == 1:
			x = x[None, :]
		if x.ndim > 2:
			raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

		if sr != 16000 and (sr % 16000 == 0):
			step = sr // 16000
			x = x[:,::step]
			sr = 16000

		if sr not in self.sample_rates:
			raise ValueError(f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)")
		if sr / x.shape[1] > 31.25:
			raise ValueError("Input audio chunk is too short")

		return x, sr

	def reset_states(self, batch_size=1):
		self._state = np.zeros((2, batch_size, 128), dtype=float)
		self._context = np.zeros(0)
		# self._state = torch.zeros((2, batch_size, 128)).float()
        # self._context = torch.zeros(0)
		self._last_sr = 0
		self._last_batch_size = 0

	def __call__(self, x, sr: int):

		x, sr = self._validate_input(x, sr)
		num_samples = 512 if sr == 16000 else 256

		if x.shape[-1] != num_samples:
			raise ValueError(f"Provided number of samples is {x.shape[-1]} (Supported values: 256 for 8000 sample rate, 512 for 16000)")

		batch_size = x.shape[0]
		context_size = 64 if sr == 16000 else 32

		if not self._last_batch_size:
			self.reset_states(batch_size)
		if (self._last_sr) and (self._last_sr != sr):
			self.reset_states(batch_size)
		if (self._last_batch_size) and (self._last_batch_size != batch_size):
			self.reset_states(batch_size)

		if not len(self._context):
			self._context = np.zeros((batch_size, context_size))
			# self._context = torch.zeros(batch_size, context_size)

		x = np.concatenate([self._context, x], axis=1)
		# x = torch.cat([self._context, x], dim=1)
		if sr in [8000, 16000]:
			ort_inputs = {'input': x.astype(np.float32), 'state': self._state.astype(np.float32), 'sr': np.array(sr, dtype='int64')}
			ort_outs = self.session.run(None, ort_inputs)
			out, state = ort_outs
			# self._state = torch.from_numpy(state)
			self._state = state
		else:
			raise ValueError()

		self._context = x[..., -context_size:]
		self._last_sr = sr
		self._last_batch_size = batch_size

		# out = torch.from_numpy(out)
		return out

	def audio_forward(self, x, sr: int):
		outs = []
		x, sr = self._validate_input(x, sr)
		self.reset_states()
		num_samples = 512 if sr == 16000 else 256

		if x.shape[1] % num_samples:
			pad_num = num_samples - (x.shape[1] % num_samples)
			x = torch.nn.functional.pad(x, (0, pad_num), 'constant', value=0.0)

		for i in range(0, x.shape[1], num_samples):
			wavs_batch = x[:, i:i+num_samples]
			out_chunk = self.__call__(wavs_batch, sr)
			outs.append(out_chunk)

		stacked = torch.cat(outs, dim=1)
		return stacked.cpu()

class VADWorker():
	def __init__(self, 
				sample_rate: int=16000,
				webrtc_sensitivity: int=3,
				silero_use_onnx: bool=True,
				silero_sensitivity: float=0.4,
				int16_max_abs_value: float=32768.0,
				):
		model_path_silero = "./weights/silero_vad.onnx"
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
			# self.silero_vad_model, _ = torch.hub.load(
			# 	repo_or_dir=model_path_silero,
			# 	model="silero_vad",
			# 	verbose=False,
			# 	onnx=silero_use_onnx
			# )
			self.silero_vad_model = OnnxWrapper(model_path_silero)

		except Exception as e:
			logging.exception(
				f"Error initializing Silero VAD voice activity detection engine: {e}"
			)
			raise

		logging.debug(
			"Silero VAD voice activity detection engine initialized successfully"
		)

		self.silero_working = False
		self.sample_rate = sample_rate
		self.is_silero_speech_active = False
		self.is_webrtc_speech_active = False
		self.silero_sensitivity = silero_sensitivity

	def _is_silero_speech(self, chunk):
		"""
		Returns true if speech is detected in the provided audio data

		Args:
			data (bytes): raw bytes of audio data (1024 raw bytes with
			16000 sample rate and 16 bits per sample)
		"""
		if self.sample_rate != 16000:
			pcm_data = np.frombuffer(chunk, dtype=np.int16)
			data_16000 = signal.resample_poly(
				pcm_data, 16000, self.sample_rate)
			chunk = data_16000.astype(np.int16).tobytes()

		self.silero_working = True
		audio_chunk = np.frombuffer(chunk, dtype=np.int16)
		audio_chunk = audio_chunk.astype(np.float32) / int16_max_abs_value
		vad_prob = self.silero_vad_model(
			audio_chunk,
			SAMPLE_RATE).item()
		is_silero_speech_active = vad_prob > (1 - self.silero_sensitivity)
		# if is_silero_speech_active:
		#     if not self.is_silero_speech_active and self.use_extended_logging:
		#         logging.info(f"{bcolors.OKGREEN}Silero VAD detected speech{bcolors.ENDC}")
		# elif self.is_silero_speech_active and self.use_extended_logging:
		#     logging.info(f"{bcolors.WARNING}Silero VAD detected silence{bcolors.ENDC}")
		self.is_silero_speech_active = is_silero_speech_active
		self.silero_working = False
		return is_silero_speech_active

	def _is_webrtc_speech(self, chunk, all_frames_must_be_true=False):
		"""
		Returns true if speech is detected in the provided audio data

		Args:
			data (bytes): raw bytes of audio data (1024 raw bytes with
			16000 sample rate and 16 bits per sample)
		"""
		# speech_str = f"{bcolors.OKGREEN}WebRTC VAD detected speech{bcolors.ENDC}"
		# silence_str = f"{bcolors.WARNING}WebRTC VAD detected silence{bcolors.ENDC}"
		if self.sample_rate != 16000:
			pcm_data = np.frombuffer(chunk, dtype=np.int16)
			data_16000 = signal.resample_poly(
				pcm_data, 16000, self.sample_rate)
			chunk = data_16000.astype(np.int16).tobytes()

		# Number of audio frames per millisecond
		frame_length = int(16000 * 0.01)  # for 10ms frame
		num_frames = int(len(chunk) / (2 * frame_length))
		speech_frames = 0

		for i in range(num_frames):
			start_byte = i * frame_length * 2
			end_byte = start_byte + frame_length * 2
			frame = chunk[start_byte:end_byte]
			if self.webrtc_vad_model.is_speech(frame, 16000):
				speech_frames += 1
				if not all_frames_must_be_true:
					# if self.debug_mode:
					#     logging.info(f"Speech detected in frame {i + 1} of {num_frames}")
					# if not self.is_webrtc_speech_active and self.use_extended_logging:
					# 	logging.info(speech_str)
					self.is_webrtc_speech_active = True
					return True
		if all_frames_must_be_true:
			# if self.debug_mode and speech_frames == num_frames:
			#     logging.info(f"Speech detected in {speech_frames} of {num_frames} frames")
			# elif self.debug_mode:
			#     logging.info(f"Speech not detected in all {num_frames} frames")
			speech_detected = speech_frames == num_frames
			# if speech_detected and not self.is_webrtc_speech_active and self.use_extended_logging:
			#     logging.info(speech_str)
			# elif not speech_detected and self.is_webrtc_speech_active and self.use_extended_logging:
			#     logging.info(silence_str)
			self.is_webrtc_speech_active = speech_detected
			return speech_detected
		else:
			# if self.debug_mode:
			#     logging.info(f"Speech not detected in any of {num_frames} frames")
			# if self.is_webrtc_speech_active and self.use_extended_logging:
			#     logging.info(silence_str)
			self.is_webrtc_speech_active = False
			return False

	def _check_voice_activity(self, data):
		"""
		Initiate check if voice is active based on the provided data.

		Args:
			data: The audio data to be checked for voice activity.
		"""
		self._is_webrtc_speech(data)

		# First quick performing check for voice activity using WebRTC
		if self.is_webrtc_speech_active:

			if not self.silero_working:
				self.silero_working = True

				# Run the intensive check in a separate thread
				threading.Thread(
					target=self._is_silero_speech,
					args=(data,)).start()

	def _is_voice_active(self):
		"""
		Determine if voice is active.

		Returns:
			bool: True if voice is active, False otherwise.
		"""
		# print(f"----webrtc check: {self.is_webrtc_speech_active}")
		# print(f"----silero check: {self.is_silero_speech_active}")
		return self.is_webrtc_speech_active and self.is_silero_speech_active 


SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INIT_SILERO_SENSITIVITY = 0.4
INIT_WEBRTC_SENSITIVITY = 3
INT16_MAX_ABS_VALUE = 32768.0
if __name__ == '__main__':
	from data_worker import DataWorker, mp, c_bool
	import queue

	audio_queue = mp.Queue()
	sample_rate = SAMPLE_RATE
	buffer_size = BUFFER_SIZE
	input_device_index = None
	shutdown_event = mp.Event()
	interrupt_stop_event = mp.Event()
	use_microphone = True
	use_microphone = mp.Value(c_bool, use_microphone)

	dtw = DataWorker(audio_queue,
					sample_rate,
					buffer_size,
					input_device_index,
					shutdown_event,
					interrupt_stop_event,
					use_microphone)
	# dtw.daemon = True
	dtw.start()

	webrtc_sensitivity = INIT_WEBRTC_SENSITIVITY
	silero_use_onnx = True
	silero_sensitivity = INIT_SILERO_SENSITIVITY
	int16_max_abs_value = INT16_MAX_ABS_VALUE
	vadw = VADWorker(sample_rate,
					webrtc_sensitivity,
					silero_use_onnx,
					silero_sensitivity,
					int16_max_abs_value)

	while True:
		try:
			data = dtw.audio_queue.get(timeout=0.1)
			data_copy = data[:]
			vadw._check_voice_activity(data_copy)
			print(f"----Check VAD: {vadw._is_voice_active()}")
		except queue.Empty:
			continue