import sys 
from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if ROOT not in sys.path:
	sys.path.append(str(ROOT))

from typing import Iterable, List, Optional, Union
import time
import queue
import logging
import threading
import faster_whisper
import multiprocessing as mp
import signal as system_signal
import torch

class TranscriptionWorker:
	def __init__(self, 
				conn=None, 
				stdout_pipe=None, 
				model_path: str="./weights/faster-whisper-medium", 
				compute_type: str = "default", 
				gpu_device_index: Union[int, List[int]] = 0, 
				device: str = "cuda",
				ready_event=None, 
				shutdown_event=None, 
				interrupt_stop_event=None, 
				beam_size: int = 5,
				initial_prompt: Optional[Union[str, Iterable[int]]] = None,
				suppress_tokens: Optional[List[int]] = [-1],
				):
		self.conn = conn
		self.stdout_pipe = stdout_pipe
		self.model_path = model_path
		self.compute_type = compute_type
		self.gpu_device_index = gpu_device_index
		# Set device for model
		self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
		self.ready_event = ready_event
		self.shutdown_event = shutdown_event
		self.interrupt_stop_event = interrupt_stop_event
		self.beam_size = beam_size
		self.initial_prompt = initial_prompt
		self.suppress_tokens = suppress_tokens
		self.queue = queue.Queue()

		self.time_sleep = 0.02

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
					print(f"----data: {data}")
					self.queue.put(data)
				except Exception as e:
					logging.error(f"Error receiving data from connection: {e}")
			else:
				time.sleep(self.time_sleep)

	def run(self):
		# if __name__ == "__main__":
		# 	 system_signal.signal(system_signal.SIGINT, system_signal.SIG_IGN)
		# 	 __builtins__.print = self.custom_print

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
						print(f"----segments: {transcription}")
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

SAMPLE_RATE = 16000
BUFFER_SIZE = 512
INT16_MAX_ABS_VALUE = 32768.0
if __name__=="__main__":
	parent_transcription_pipe, child_transcription_pipe = mp.Pipe()
	parent_stdout_pipe, child_stdout_pipe = mp.Pipe()
	model = "./weights/faster-whisper-small"
	compute_type = "default"
	gpu_device_index = 0
	device = "cuda"
	main_transcription_ready_event = mp.Event()
	shutdown_event = mp.Event()
	interrupt_stop_event = mp.Event()
	beam_size = 5
	initial_prompt = None
	suppress_tokens = [-1]

	worker = TranscriptionWorker(child_transcription_pipe,
								child_stdout_pipe,
								model,
								compute_type,
								gpu_device_index,
								device,
								main_transcription_ready_event,
								shutdown_event,
								interrupt_stop_event,
								beam_size,
								initial_prompt,
								suppress_tokens)
	thread = threading.Thread(target=worker.run, args=())
	thread.daemon = True
	thread.start()
	# worker.run()
	# __builtins__.print = worker.custom_print
	# main_transcription_ready_event.wait()
	#-----------get data----------
	from data_worker import DataWorker, mp, c_bool
	import queue
	import numpy as np

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
	#/////////////////////////////

	language = "vi"
	int16_max_abs_value = INT16_MAX_ABS_VALUE
	print(f"----Checking A2T----")
	frames = []
	while True:
		try:
			data = dtw.audio_queue.get(timeout=0.1)
			# exit()
			frames.append(data)
			if len(frames) == SAMPLE_RATE//dtw.chunk_size:
				print(f"----Checking A2T----")
				audio_array = np.frombuffer(b''.join(frames), dtype=np.int16)
				audio = audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
				parent_transcription_pipe.send((audio, language))
				# if parent_transcription_pipe.poll(timeout=5):  # Wait for 5 seconds
				status, result = parent_transcription_pipe.recv()
				print(f"----status: {status}")
				print(f"----result: {result}")
		except queue.Empty:
			continue