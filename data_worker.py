import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from typing import Iterable, List, Optional, Union
import time
import pyaudio
import logging
import traceback
import threading
import numpy as np
from scipy import signal
from ctypes import c_bool
import multiprocessing as mp
import signal as system_signal

class DataWorker(threading.Thread):
	def __init__(self, 
				audio_queue=None,
				target_sample_rate: int=16000,
				buffer_size: int=512,
				input_device_index: int=None,
				shutdown_event=None,
				interrupt_stop_event=None,
				use_microphone: bool=True
				):
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
		threading.Thread.__init__(self)
		self.audio_queue = audio_queue
		self.target_sample_rate = target_sample_rate
		self.buffer_size = buffer_size
		self.input_device_index = input_device_index
		self.shutdown_event = shutdown_event
		self.interrupt_stop_event = interrupt_stop_event
		self.use_microphone = use_microphone

		self.audio_interface = None
		self.stream = None
		self.device_sample_rate = None
		self.chunk_size = 1024  # Increased chunk size for better performance

		self.buffer = bytearray()
		self.silero_buffer_size = 2 * buffer_size  # silero complains if too short
		self.time_since_last_buffer_message = 0

	@staticmethod
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

	def initialize_audio_stream(self, audio_interface, sample_rate, chunk_size):
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
					rate=self.target_sample_rate,
					input=True,
					frames_per_buffer=self.chunk_size,
					input_device_index=device_index,
					start=False  # Don't start the stream yet
				)

				# Start the stream and try to read from it
				test_stream.start_stream()
				test_data = test_stream.read(self.chunk_size, exception_on_overflow=False)
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
		while not self.shutdown_event.is_set():
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
				if self.input_device_index is None or self.input_device_index not in input_devices:
					# First try the default device
					try:
						default_device = audio_interface.get_default_input_device_info()
						if validate_device(default_device['index']):
							self.input_device_index = default_device['index']
					except Exception:
						# If default device fails, try other available input devices
						for device_index in input_devices:
							if validate_device(device_index):
								self.input_device_index = device_index
								break
						else:
							raise Exception("No working input devices found")

				# Validate the selected device one final time
				if not validate_device(self.input_device_index):
					raise Exception("Selected device validation failed")

				# print(f"----sample_rate: {sample_rate}")
				# If we get here, we have a validated device
				stream = audio_interface.open(
					format=pyaudio.paInt16,
					channels=1,
					rate=sample_rate,
					input=True,
					frames_per_buffer=chunk_size,
					input_device_index=self.input_device_index,
				)

				logging.info(f"Microphone connected and validated (input_device_index: {self.input_device_index})")
				return stream

			except Exception as e:
				logging.error(f"Microphone connection failed: {e}. Retrying...")
				self.input_device_index = None
				time.sleep(3)  # Wait before retrying
				continue

	@staticmethod
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

	def setup_audio(self,):  
		try:
			if self.audio_interface is None:
				self.audio_interface = pyaudio.PyAudio()
			if self.input_device_index is None:
				try:
					default_device = self.audio_interface.get_default_input_device_info()
					self.input_device_index = default_device['index']
				except OSError as e:
					self.input_device_index = None

			sample_rates_to_try = [16000]  # Try 16000 Hz first
			if self.input_device_index is not None:
				highest_rate = self.get_highest_sample_rate(self.audio_interface, self.input_device_index)
				if highest_rate != 16000:
					sample_rates_to_try.append(highest_rate)
			else:
				sample_rates_to_try.append(48000)  # Fallback sample rate

			for rate in sample_rates_to_try:
				try:
					self.device_sample_rate = rate
					self.stream = self.initialize_audio_stream(self.audio_interface, self.device_sample_rate, self.chunk_size)
					if self.stream is not None:
						logging.debug(f"Audio recording initialized successfully at {self.device_sample_rate} Hz, reading {self.chunk_size} frames at a time")
						# logging.error(f"Audio recording initialized successfully at {self.device_sample_rate} Hz, reading {self.chunk_size} frames at a time")
						return True
				except Exception as e:
					logging.warning(f"Failed to initialize audio23 stream at {self.device_sample_rate} Hz: {e}")
					continue

			# If we reach here, none of the sample rates worked
			raise Exception("Failed to initialize audio stream12 with all sample rates.")

		except Exception as e:
			logging.exception(f"Error initializing pyaudio audio recording: {e}")
			if self.audio_interface:
				self.audio_interface.terminate()
			return False

	def run(self,):
		# if __name__ == "__main__":
			# system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN) # to ignore ctrl+c
			# system_signal.signal(system_signal.SIGINT, system_signal.SIGQUIT) # to exit ctrl+c

		if not self.setup_audio():
			raise Exception("Failed to set up audio recording.")

		try:
			while not self.shutdown_event.is_set():
				try:
					data = self.stream.read(self.chunk_size, exception_on_overflow=False)
					
					if self.use_microphone.value:
						processed_data = self.preprocess_audio(data, self.device_sample_rate, self.target_sample_rate)
						self.buffer += processed_data

						# Check if the buffer has reached or exceeded the silero_buffer_size
						while len(self.buffer) >= self.silero_buffer_size:
							# Extract silero_buffer_size amount of data from the buffer
							to_process = self.buffer[:self.silero_buffer_size]
							self.buffer = self.buffer[self.silero_buffer_size:]

							# Feed the extracted data to the audio_queue
							if self.time_since_last_buffer_message:
								time_passed = time.time() - self.time_since_last_buffer_message
								if time_passed > 1:
									logging.debug("_audio_data_worker writing audio data into queue.")
									self.time_since_last_buffer_message = time.time()
							else:
								self.time_since_last_buffer_message = time.time()

							self.audio_queue.put(to_process)
							

				except OSError as e:
					if e.errno == pyaudio.paInputOverflowed:
						logging.warning("Input overflowed. Frame dropped.")
					else:
						logging.error(f"OSError during recording: {e}")
						# Attempt to reinitialize the stream
						logging.error("Attempting to reinitialize the audio stream...")

						try:
							if self.stream:
								self.stream.stop_stream()
								self.stream.close()
						except Exception as e:
							pass
						
						# Wait a bit before trying to reinitialize
						time.sleep(1)
						
						if not self.setup_audio():
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
						if self.stream:
							self.stream.stop_stream()
							self.stream.close()
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

				# print(f"---audio_queue: {len(self.audio_queue.get(timeout=0.1))}")

		except KeyboardInterrupt:
			self.interrupt_stop_event.set()
			logging.debug("Audio data worker process finished due to KeyboardInterrupt")
		finally:
			# After recording stops, feed any remaining audio data
			if buffer:
				self.audio_queue.put(bytes(buffer))
			
			try:
				if self.stream:
					self.stream.stop_stream()
					self.stream.close()
			except Exception as e:
				pass
			if self.audio_interface:
				self.audio_interface.terminate()

	def record(self, record_seconds=5, filename="record.wav"):
		from array import array
		import wave

		if not self.setup_audio():
			raise Exception("Failed to set up audio recording.")

		#starting recording
		frames=[]
		for i in range(0, int(self.target_sample_rate/self.chunk_size*record_seconds)):
			data = self.stream.read(self.chunk_size, exception_on_overflow=False)
			data_chunk=array('h',data)
			vol=max(data_chunk)
			if(vol>=500):
				print("something said")
				frames.append(data)
			else:
				print("nothing")

		#end of recording
		self.stream.stop_stream()
		self.stream.close()
		self.audio_interface.terminate()
		#writing to file
		wavfile=wave.open(filename,'wb')
		wavfile.setnchannels(1)
		wavfile.setsampwidth(self.audio_interface.get_sample_size(pyaudio.paInt16))
		wavfile.setframerate(self.target_sample_rate)
		wavfile.writeframes(b''.join(frames))#append frames recorded to file
		wavfile.close()


SAMPLE_RATE = 16000
BUFFER_SIZE = 512
if __name__ == '__main__':
	# system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)

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
	dtw.daemon = True
	dtw.start()

	# #record to file
	# record_seconds=5
	# filename="p_record3.wav"
	# dtw.record(record_seconds, filename)