import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from typing import Iterable, List, Optional, Union
import time
import logging
import pvporcupine
import onnxruntime as ort
import numpy as np
from openwakeword.model import OWWModel

class OWWWorker():
	def __init__(self, 
				wake_words: str="",
				wake_words_sensitivity: float=0.6,
				wakeword_backend: str="openwakeword",
				openwakeword_inference_framework: str="onnx",
				openwakeword_model_paths: str="./weights/alexa_v0.1.onnx",
				):
		# Setup wake word detection
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
			try:
				if openwakeword_model_paths:
					model_paths = openwakeword_model_paths.split(',')
					self.owwModel = OWWModel(
						wakeword_models=model_paths,
						inference_framework=openwakeword_inference_framework,
						melspec_model_path="./weights/melspectrogram.onnx",
						embedding_model_path="./weights/embedding_model.onnx",
					)
					logging.info(
						f"Successfully loaded wakeword model(s): {openwakeword_model_paths}"
					)
				else:
					self.owwModel = OWWModel(
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

	def _process_wakeword(self, data):
		"""
		Processes audio data to detect wake words.
		"""
		if self.wakeword_backend in {'pvp', 'pvporcupine'}:
			pcm = struct.unpack_from(
				"h" * self.buffer_size,
				data
			)
			porcupine_index = self.porcupine.process(pcm)
			# if self.debug_mode:
			#     logging.info(f"wake words porcupine_index: {porcupine_index}")
			return self.porcupine.process(pcm)

		elif self.wakeword_backend in {'oww', 'openwakeword', 'openwakewords'}:
			pcm = np.frombuffer(data, dtype=np.int16)
			prediction = self.owwModel.predict(pcm)
			max_score = -1
			max_index = -1
			wake_words_in_prediction = len(self.owwModel.prediction_buffer.keys())
			# self.wake_words_sensitivities
			if wake_words_in_prediction:
				for idx, mdl in enumerate(self.owwModel.prediction_buffer.keys()):
					scores = list(self.owwModel.prediction_buffer[mdl])
					if scores[-1] >= self.wake_words_sensitivity and scores[-1] > max_score:
						max_score = scores[-1]
						max_index = idx
				# if self.debug_mode:
				#     logging.info(f"wake words oww max_index, max_score: {max_index} {max_score}")
				return max_index  
			else:
				# if self.debug_mode:
				#     logging.info(f"wake words oww_index: -1")
				return -1

		# if self.debug_mode:        
		#     logging.info("wake words no match")
		return -1

SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INIT_WAKE_WORDS_SENSITIVITY = 0.6
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

	wake_words = "alexa"
	wake_words_sensitivity = INIT_WAKE_WORDS_SENSITIVITY
	wakeword_backend = "openwakeword"
	openwakeword_inference_framework = "onnx"
	openwakeword_model_paths ="./weights/alexa_v0.1.onnx"
	owww = OWWWorker(wake_words,
					wake_words_sensitivity,
					wakeword_backend,
					openwakeword_inference_framework,
					openwakeword_model_paths)

	while True:
		try:
			data = dtw.audio_queue.get(timeout=0.1)
			data_copy = data[:]
			wakeword_index = owww._process_wakeword(data_copy)
			print(f"----Check WW: {wakeword_index}")
		except queue.Empty:
			continue