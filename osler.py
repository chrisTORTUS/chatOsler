from tkinter import *
import intents
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os
import base64
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import json
from google.cloud import vision
from PIL import ImageGrab, Image, ImageDraw
import pandas as pd
import numpy as np
import time
import pyautogui
import openai
from PIL import Image, ImageTk
import cv2
# from picovoice import Picovoice
# from pvrecorder import PvRecorder
import pyperclip
import json
import re
import sys
from google.cloud import speech
import pyaudio
from six.moves import queue
import sounddevice as sd
from scipy.io.wavfile import write
# import mutagen
# from mutagen.wave import WAVE
import eyed3
import requests
import epic_screens
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import threading
import subprocess

from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

class Actor:
	def __init__(self) -> None:
		self.consultation_transcript = ""
		self.transcript_summary = ""
		self.consultation_entities = {}
		self.mrn_flag = False
		self.user_utterance_text = ''
		self.patient_mrn_str = ''
		self.patient_mrn_digits = '111'
		self.med_hx = ''
		self.letters_hx = ''

		self.global_mrn = ''

	def get_element_center(self, loc):
		'''
		Method to get the center of element's bounding box
		'''
		corner_x, corner_y = loc[0], loc[1]
		width, height = loc[2], loc[3]
		x, y = corner_x/2 + width/4, corner_y/2 + height/4
		return x, y

	def click_screenshot(self, screenshot, confidence=0.8):
		'''
		Method to click on a matching screenshot.
		'''
		# loc = pyautogui.locateOnScreen(root_path + f"demo_screenshots/{screenshot}", confidence=confidence)
		loc = pyautogui.locateOnScreen(f"demo_screenshots/{screenshot}")
		if loc is None:
			print('cant find it!')
			return
			# raise Exception("Matching image not found on screen.")
		x, y = self.get_element_center(loc)
		print(f"Mouse click at: {x, y}")
		pyautogui.click(x, y)
		
	def activate_application(self, app_name):
		applescript_code = f'''
		tell application "{app_name}"
			activate
		end tell
		'''

		process = subprocess.Popen(['osascript', '-e', applescript_code],
								stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
		output, error = process.communicate()

		if error:
			print("Error executing AppleScript:", error)
			return

	def new_tab(self):
		'''
		Opens a new tab.
		'''
		pyautogui.hotkey("ctrl", "t")

	def type_string(self, char_string, interval=0.2):
		'''
		Types a given string.
		'''
		pyautogui.write(char_string, interval=interval)

	def press_key(self, key, presses=1):
		'''
		Presses a given key.
		'''
		pyautogui.press(key, presses=presses)

	def press_command(self, command):
		'''
		Performs a given hotkey command.
		'''
		if command == "copy":
			pyautogui.hotkey("ctrl", "c")
		elif command == "paste":
			pyautogui.hotkey("ctrl", "v")
		elif command == "tab_back":
			pyautogui.hotkey("alt", "tab")
		else:
			raise Exception(f"Command {command} not recognized.")

	def scroll(self, offset):
		'''
		Vertical scrolling.
		'''
		pyautogui.scroll(offset)

	def _wake_word_callback(self):
		img = Image.open("demo_screenshots/osler_awake.png").resize((self._width, self._height))
		osler_awake = ImageTk.PhotoImage(img)
		self._label.configure(image=osler_awake)
		self._label.image = osler_awake

	def listen_print_loop(self, responses):
		"""Iterates through server responses and prints them.

		The responses passed is a generator that will block until a response
		is provided by the server.

		Each response may contain multiple results, and each result may contain
		multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
		print only the transcription for the top alternative of the top result.

		In this case, responses are provided for interim results as well. If the
		response is an interim one, print a line feed at the end of it, to allow
		the next result to overwrite it, until the response is a final one. For the
		final one, print a newline to preserve the finalized transcription.
		"""
		num_chars_printed = 0
		for response in responses:
			if not response.results:
				continue

			# The `results` list is consecutive. For streaming, we only care about
			# the first result being considered, since once it's `is_final`, it
			# moves on to considering the next utterance.
			result = response.results[0]
			if not result.alternatives:
				continue

			# Display the transcription of the top alternative.
			transcript = result.alternatives[0].transcript

			# Display interim results, but with a carriage return at the end of the
			# line, so subsequent lines will overwrite them.
			#
			# If the previous result was longer than this one, we need to print
			# some extra spaces to overwrite the previous result
			overwrite_chars = " " * (num_chars_printed - len(transcript))

			if not result.is_final:
				# sys.stdout.write(transcript + overwrite_chars + "\r")
				# sys.stdout.flush()

				# num_chars_printed = len(transcript)
				pass

			else:
				
				# print(transcript + overwrite_chars)
				output = transcript + overwrite_chars
				self.consultation_transcript += output
				output = output.lower()

				if "stop recording" in output:
					break

				pyperclip.copy(transcript + overwrite_chars)
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')

				# Exit recognition if any of the transcribed phrases could be
				# one of our keywords.
				if re.search(r"\b(exit|quit)\b", transcript, re.I):
					print("Exiting..")
					break

				num_chars_printed = 0


	def transcribe(self):
		# See http://g.co/cloud/speech/docs/languagesv
		# for a list of supported languages.
		language_code = "en-US"  # a BCP-47 language tag

		client = speech.SpeechClient()
		config = speech.RecognitionConfig(
			encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
			sample_rate_hertz=RATE,
			language_code=language_code,
			model='medical_conversation'
		)

		streaming_config = speech.StreamingRecognitionConfig(
			config=config, interim_results=True
		)

		with MicrophoneStream(RATE, CHUNK) as stream:
			audio_generator = stream.generator()
			requests = (
				speech.StreamingRecognizeRequest(audio_content=content)
				for content in audio_generator
			)

			responses = client.streaming_recognize(streaming_config, requests)

			# Now, put the transcription responses to use.
			self.listen_print_loop(responses)

	def match_screen(self):
		# get text representation of current screen
		current_screen = ""

		model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

		screens_ls = [
			epic_screens.PATIENT_LOOKUP,
			epic_screens.SCHEDULE,
			epic_screens.PATIENT_PAGE
		]

		epic_embeddings = model.encode(screens_ls)
		screen_embeddings = model.encode(current_screen)

		cos_scores = cosine_similarity(screen_embeddings.reshape(1, -1), epic_embeddings)
		cos_scores_torch = torch.from_numpy(cos_scores)
		cos_max = torch.max(cos_scores_torch).item()
		cos_argmax = torch.argmax(cos_scores_torch, dim=1)
		cos = cos_argmax[0].item()


		print(cos_scores)
		intent = screens_ls[cos]
		print(f"Intent matched: {intent}")

	def act(self, intent):
		if intent == intents.START_CONSULTATION_NOTE:
			self.new_consultation_mrn()
		elif intent == intents.TRANSCRIBE_CONSULTATION:
			self.transcribe_consultation()
		# elif intent == intents.QUERY_ORDERS:
		#     self.query_orders()
		# elif intent == intents.QUERY_MEDS:
		#     self.query_meds()
		elif intent == intents.WRITE_LETTER:
			self.write_referral()
		# elif intent == intents.PLACE_ORDERS:
		#     self.place_orders()
		# elif intent == intents.FILE_DIAGNOSES:
		#     self.file_diagnoses()
		# elif intent == intents.ANSWER_QUESTIONS:
		#     self.ask_general_consultation_question()
		else:
			raise ValueError("unsupported intent '%s'" % intent)
		
	def get_user_voice_response(self):
		fs = 44100  # Sample rate
		seconds = 6  # Duration of recording

		myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
		sd.wait()  # Wait until recording is finished
		write('user_response.wav', fs, myrecording)  # Save as WAV file

	def leopard_transcribe(self):
		transcript, words = LEOPARD.process_file('user_response.wav')
		print(transcript)
		for word in words:
			print(
			"{word=\"%s\" start_sec=%.2f end_sec=%.2f confidence=%.2f}"
			% (word.word, word.start_sec, word.end_sec, word.confidence))
		return transcript
	
	def str_to_digit(self, nstr):
		digit = ''
		is_digit = True
		if nstr == 'zero':
			digit = '0'
		elif nstr == 'one':
			digit = '1'
		elif nstr == 'two':
			digit = '2'
		elif nstr == 'three':
			digit = '3'
		elif nstr == 'four':
			digit = '4'
		elif nstr == 'five':
			digit = '5'
		elif nstr == 'six':
			digit = '6'
		elif nstr == 'seven':
			digit = '7'
		elif nstr == 'eight':
			digit = '8'
		elif nstr == 'nine':
			digit = '9'
		else:
			print('error converting string to digit')
			is_digit = False
		return digit, is_digit
	
	def convert_string_to_num(self, num_str):
		num_str_ls = num_str.split(' ')
		digits_str = ''
		for num_str in num_str_ls:
			digits_str += self.str_to_digit(num_str)
		return digits_str
	
	def extract_mrn_from_utterance(self, utterance_str):
		str_ls = utterance_str.split(' ')
		mrn = ''
		for s in str_ls:
			digit, is_digit = self.str_to_digit(s)
			if is_digit:
				mrn += digit
		return mrn
	
	def extract_mrn_from_text(self, utterance_str):
		str_ls = utterance_str.split(' ')
		mrn = ''
		for s in str_ls:
			if s.isdigit():
				mrn = s
		return mrn
		
	def ask_general_consultation_question(self):
		# play the audio file of the question
		PLAYER.play_song("ask_general_consultation_question.wav")
		time.sleep(2)

		# record the user response and write to a  wav audio file
		self.get_user_voice_response()

		# use picovoice leopard to transcribe the audio response file
		question = self.leopard_transcribe()

		# combine the quetsion with the consultation transcript
		question_about_consultation_prompt = 'INSTRUCTION: You are a medical doctor who has just performed a consultation and is provided with a transcript of the consultation. Answer a question about the consultation as accurately as possible. The consultation transcritp and question about it will follow\n'
		question_about_consultation_prompt += '\nCONSULTATION TRANSCRIPT: \n' + self.consultation_transcript
		question_about_consultation_prompt += '\nQUESTION ABOUT CONSULTATION: \n' + question + '?\n\n'
		response=openai.Completion.create(
		model="text-davinci-003",
		prompt=question_about_consultation_prompt,
		max_tokens=2500,
		temperature=0
		)
		answer = json.loads(str(response))
		answer = answer['choices'][0]['text']

		# print the answer
		print(answer)

		# create the audio file from the text
		# Language in which you want to convert
		language = 'en'
		
		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
		myobj = gTTS(text=answer, lang=language, slow=False)
		
		# Saving the converted audio in a mp3 file named
		myobj.save("consulation_answer.wav")

		# get the length of the audio file to know how long to sleep while playing
		audio = eyed3.load("consulation_answer.wav")
		length_in_seconds = int(audio.info.time_secs)

		#play the audio file
		PLAYER.play_song("consulation_answer.wav")
		time.sleep(length_in_seconds + 1)

	def extract_letters(self):
		time.sleep(2)

		pyautogui.keyDown('ctrl')
		pyautogui.press('space')
		pyautogui.keyUp('ctrl')
		time.sleep(1)

		pyperclip.copy('chart review')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
		time.sleep(1)

		pyautogui.press('down')
		time.sleep(0.5)
		pyautogui.press('enter')
		time.sleep(2)

		self.click_screenshot("letters.png")
		time.sleep(2)
		self.click_screenshot("recent_letters.png")
		time.sleep(1)

		letters = ''

		for i in range(5):
			pyautogui.press("enter")
			time.sleep(2)

			pyautogui.click()
			time.sleep(1)

			pyautogui.keyDown('command')
			pyautogui.press('a')
			pyautogui.keyUp('command')
			time.sleep(1)

			pyautogui.keyDown('command')
			pyautogui.press('c')
			pyautogui.keyUp('command')
			letters += pyperclip.paste()
			time.sleep(1)

			pyautogui.keyDown('option')
			pyautogui.keyDown('command')
			pyautogui.press('left')
			pyautogui.keyUp('option')
			pyautogui.keyUp('command')
			time.sleep(2)

			pyautogui.press('down')
			time.sleep(1)

		self.letters_hx = letters

	def glance_patient_search_results(self):
		# telling the user that a glance is being done
		txt.insert(END, "\n" + "OSLER -> Analysing the screen...")

		parsed_screen = parse_screen()
		sys_instr = '''You are a medical admin assistant with one simple task, which is, given a text representation of a Epic EHR patient lookup window for some MRN (medical record number), classify whether:
		
		(1) The MRN matches exactly 1 patient
		(2) The MRN matches more than 1 patient
		(3) The MRN does not match any patients
		
		You simply have to respond either: '1', '2', or '3'

		Here are some examples:

		Example screen 1:
		========================
		<text id=0>Search for a Patient</text>
		<link id=1></link>
		<link id=2>L</link>
		<link id=3></link>
		<link id=4>Privacy View</link>
		<text id=5>Forename</text>
		<input id=6></input>
		<link id=7>MDT Conference</link>
		<text id=8>NHS No.</text>
		<link id=9></link>
		<input id=10>190</input>
		<link id=11></link>
		<link id=12></link>
		<button id=13>0 Find Patient</button>
		<link id=14></link>
		<text id=15>No patients were found .</text>
		<input id=16></input>
		<text id=17>Surname</text>
		<link id=18></link>
		<text id=19>Phone</text>
		<link id=20></link>
		<link id=21></link>
		<link id=22>Edit Assignments</link>
		<link id=23>➡Create New Patient</link>
		<link id=24>create a new patient</link>
		<text id=25>MRN</text>
		<link id=26>PROOF OF CONCEPT ( PO</link>
		<link id=27>Answer Pt - Qnr ( Captive )</link>
		<text id=28>Update the search terms to try again , or</text>
		<link id=29>Open Slots</link>
		<link id=30>tv</link>
		<text id=31>ID Type</text>
		<link id=32>U</link>
		<link id=33>Slicer Dicer</link>
		<link id=34>Personalise</link>
		<link id=35>My Incomplete Notes 1</link>
		<input id=36>O</input>
		<link id=37>Letter Queue 15</link>
		<link id=38>Procedure Catalogue</link>
		<text id=39>Patient</text>
		<link id=40>Sign My Visits</link>
		<input id=41>0</input>
		<link id=42>Devices</link>
		<link id=43>MDT</link>
		<link id=44>Status Board</link>
		<link id=45>NOA</link>
		<link id=46>16</link>
		<input id=47>□</input>
		<link id=48>Reschedule MDT Meeting</link>
		<link id=49>Schedule</link>
		<link id=50>Epic</link>
		<link id=51>Check In</link>
		<input id=52></input>
		<link id=53></link>
		<link id=54></link>
		<link id=55>Orders Only</link>
		<button id=56>Results</button>
		<link id=57>Tel Encounter</link>
		<link id=58>Telephone Call</link>
		<text id=59>Sex</text>
		<link id=60>www</link>
		<link id=61>/ Service Task</link>
		<link id=62></link>
		<link id=63>4</link>
		<link id=64>Notes</link>
		<link id=65></link>
		<link id=66>Remind Me</link>
		<link id=67></link>
		<link id=68>CT</link>
		<link id=69></link>
		<link id=70>View</link>
		<link id=71>Citrix Viewer</link>
		<text id=72>Postcode</text>
		<link id=73>Check Out</link>
		<link id=74>Abstract</link>
		<link id=75>JUL 25</link>
		<link id=76>3+</link>
		<input id=77></input>
		<link id=78>Go to In Basket for other tasks</link>
		<link id=79></link>
		<link id=80>☎</link>
		<link id=81>Print</link>
		<link id=82></link>
		<text id=83>Privacy View is on</text>
		<link id=84></link>
		<link id=85>Clear</link>
		<link id=86></link>
		<link id=87></link>
		<link id=88>Video Visit</link>
		<link id=89></link>
		<text id=90>Birthdate</text>
		<link id=91>GREAT ORMOND STREET ...</link>
		<input id=92></input>
		<link id=93>EpicCare</link>
		<link id=94>X</link>
		<link id=95>Hold MDT</link>
		<link id=96>42</link>
		<text id=97>Service Area</text>
		<link id=98>> Log Out</link>
		<text id=99>Start Time</text>
		<link id=100>G</link>
		<link id=101>Tue 25 Jul 11:03</link>
		<link id=102>Q</link>
		<link id=103></link>
		<link id=104>Status</link>
		<link id=105></link>
		<text id=106>TAN , CHRISTOPHER</text>
		<link id=107></link>
		<link id=108>X Sign Encounter</link>
		<link id=109>Patient Lookup</link>
		<link id=110>Chart</link>
		<link id=111>TAN , CHRISTOPHER</link>
		<link id=112>Print AVS</link>
		<link id=113></link>
		<link id=114>X Cancel</link>
		<text id=115>Perform a search on the local patient index</text>
		<link id=116>Patient Encounter</link>
		<link id=117></link>
		<link id=118></link>
		<text id=119>3 /</text>
		<link id=120>□</link>
		<link id=121>Patient Encounter</link>
		<text id=122>DNA</text>
		<text id=123>Patient Encounter</text>
		<link id=124></link>
		<link id=125></link>
		<link id=126></link>
		<link id=127></link>
		<link id=128></link>
		<link id=129></link>
		<text id=130>All Done !</text>
		<text id=131>My Patients</text>
		<text id=132>Recent Patients</text>
		<link id=133></link>
		<link id=134></link>
		<link id=135></link>
		<link id=136></link>
		<img id=137></img>
		<link id=138>X</link>
		<link id=139></link>
		<link id=140></link>
		<link id=141>L</link>
		<link id=142>www</link>
		<text id=143></text>
		<link id=144></link>
		<link id=145></link>
		<text id=146>Patient Lookup</text>
		<link id=147>My Patients</link>
		<link id=148></link>
		<link id=149></link>
		<link id=150>dh [ 6</link>
		<text id=151>190</text>
		<link id=152>tv</link>
		<link id=153>Privacy View is on .</link>
		<link id=154>?</link>
		<button id=155>X</button>
		<link id=156>www</link>
		<text id=157></text>
		<text id=158>?</text>
		<link id=159>Q</link>
		<link id=160></link>
		<text id=161></text>
		<link id=162></link>
		<text id=163></text>
		<input id=164>Accept</input>
		<link id=165></link>
		<link id=166></link>
		<input id=167>Recent Patients</input>
		<link id=168>DNA</link>
		<text id=169>TAN , CHRISTOPHER Privacy View is on .</text>
		<text id=170></text>
		<link id=171>Start Time</link>
		<link id=172></link>
		<img id=173></img>
		<img id=174></img>
		<text id=175>Hold MDT</text>
		<link id=176></link>
		<link id=177></link>
		<img id=178></img>
		<link id=179>1</link>
		<link id=180></link>
		<text id=181></text>
		<link id=182>42</link>
		<link id=183>/</link>
		<text id=184></text>
		<link id=185></link>
		<link id=186>Accept</link>
		<link id=187>All Done !</link>
		<text id=188>3</text>
		========================
		ANSWER: 3

		Example screen 2:
		========================
		<text id=0>Search for a Patient</text>
		<link id=1></link>
		<link id=2>L</link>
		<input id=3>3</input>
		<link id=4></link>
		<input id=5></input>
		<text id=6>Forename</text>
		<link id=7>Privacy View</link>
		<link id=8>MDT Conference</link>
		<link id=9></link>
		<input id=10>111</input>
		<link id=11></link>
		<link id=12></link>
		<link id=13></link>
		<link id=14></link>
		<link id=15></link>
		<link id=16></link>
		<text id=17>SESEMANN , Klara - 111</text>
		<text id=18>NHS No.</text>
		<text id=19>Surname</text>
		<link id=20>Edit Assignments</link>
		<text id=21>1619 South University</text>
		<text id=22>Ethnicity :</text>
		<text id=23>NHS Number :</text>
		<text id=24>No e - mail address on file</text>
		<link id=25>PROOF OF CONCEPT ( PO</link>
		<link id=26>Answer Pt - Qnr ( Captive )</link>
		<link id=27>➡Create New Patient</link>
		<text id=28>9 y.o. Female</text>
		<text id=29>608-251-7777 ( H</text>
		<link id=30>tv</link>
		<text id=31>Born 11/11/2013</text>
		<link id=32>Open Slots</link>
		<link id=33>U</link>
		<link id=34>Slicer Dicer</link>
		<link id=35>Personalise</link>
		<link id=36>My Incomplete Notes 1</link>
		<link id=37>Letter Queue 15</link>
		<input id=38>O</input>
		<button id=39>0 Find Patient</button>
		<link id=40>Procedure Catalogue</link>
		<text id=41>347 895 5467</text>
		<text id=42>Phone</text>
		<text id=43>MRN</text>
		<text id=44>England</text>
		<input id=45></input>
		<text id=46>BS6 7EY</text>
		<text id=47>: Decline to Answer</text>
		<link id=48>Sign My Visits</link>
		<link id=49>Devices</link>
		<link id=50>MDT Prep</link>
		<link id=51>Status Board</link>
		<text id=52>ID Type</text>
		<text id=53>Language :</text>
		<input id=54>0</input>
		<link id=55>16</link>
		<link id=56>Reschedule MDT Meeting</link>
		<link id=57>Epic</link>
		<link id=58>NOA</link>
		<link id=59>Check In</link>
		<link id=60>Schedule</link>
		<link id=61></link>
		<link id=62></link>
		<text id=63>Decline to Answer</text>
		<img id=64></img>
		<link id=65>Orders Only</link>
		<input id=66></input>
		<text id=67>Sex</text>
		<link id=68>Telephone Call</link>
		<text id=69>Patient</text>
		<link id=70>Tel Encounter</link>
		<link id=71>www</link>
		<text id=72>Race :</text>
		<text id=73>English</text>
		<input id=74>□</input>
		<link id=75>/ Service Task</link>
		<link id=76></link>
		<text id=77>Bristol</text>
		<link id=78>JUL 25</link>
		<link id=79>4</link>
		<link id=80></link>
		<link id=81>Remind Me</link>
		<link id=82></link>
		<link id=83>Notes</link>
		<link id=84>View</link>
		<link id=85>CT</link>
		<link id=86>Citrix Viewer</link>
		<link id=87>Check Out</link>
		<link id=88>This patient has the MRN that was entered</link>
		<link id=89></link>
		<link id=90>Go to In Basket for other tasks</link>
		<text id=91>Birthdate</text>
		<link id=92>Abstract</link>
		<link id=93>3+</link>
		<text id=94>11/11/2013</text>
		<link id=95>Date Of Birth</link>
		<link id=96>Secure</link>
		<link id=97></link>
		<link id=98>☎</link>
		<link id=99>Print Print</link>
		<link id=100></link>
		<button id=101>Results</button>
		<link id=102></link>
		<text id=103>Privacy View is on</text>
		<link id=104>GREAT ORMOND STREET ...</link>
		<text id=105>111</text>
		<link id=106>Video Visit</link>
		<link id=107></link>
		<text id=108>111</text>
		<text id=109>Service Area</text>
		<link id=110>✓Clear</link>
		<link id=111></link>
		<link id=112>EpicCare</link>
		<link id=113></link>
		<link id=114>Hold MDT</link>
		<link id=115>Status</link>
		<link id=116>42</link>
		<link id=117>Log Out</link>
		<text id=118>NHS No.</text>
		<link id=119>X Cancel</link>
		<text id=120>Nn GP on file</text>
		<text id=121>Start Time</text>
		<link id=122>G</link>
		<link id=123>Q</link>
		<text id=124>Match</text>
		<link id=125></link>
		<link id=126>Tue 25 Jul 11:06</link>
		<text id=127>TAN , CHRISTOPHER</text>
		<link id=128>Legal Sex</link>
		<link id=129></link>
		<link id=130></link>
		<text id=131>70.00</text>
		<text id=132>SESEMANN , KLARA</text>
		<link id=133></link>
		<input id=134></input>
		<link id=135>MRN</link>
		<text id=136>Postcode</text>
		<link id=137>Phone</link>
		<link id=138>Street Address</link>
		<link id=139>Patient Lookup</link>
		<link id=140>X Sign Encounter</link>
		<link id=141>Chart</link>
		<link id=142></link>
		<link id=143>Print AVS</link>
		<link id=144>Patient Encounter</link>
		<link id=145>TAN , CHRISTOPHER</link>
		<link id=146></link>
		<text id=147>3 /</text>
		<link id=148>Patients</link>
		<link id=149>11/11/2013</link>
		<link id=150>608-251-7777</link>
		<text id=151>Patient Name</text>
		<link id=152></link>
		<button id=153>✔Accept</button>
		<link id=154>Patient Encounter</link>
		<text id=155>DNA</text>
		<link id=156></link>
		<link id=157></link>
		<text id=158>Patient Encounter</text>
		<text id=159>608-251-7777</text>
		<link id=160></link>
		<link id=161></link>
		<text id=162>Perform a search on the local patient index</text>
		<link id=163></link>
		<link id=164></link>
		<link id=165></link>
		<input id=166>ID Type O</input>
		<text id=167>Bristol</text>
		<link id=168></link>
		<text id=169>All Done !</text>
		<link id=170></link>
		<link id=171></link>
		<img id=172></img>
		<link id=173>Legal Sex</link>
		<text id=174>MRN</text>
		<link id=175>X</link>
		<link id=176></link>
		<link id=177></link>
		<link id=178></link>
		<link id=179>L</link>
		<link id=180>www</link>
		<link id=181>✔Accept</link>
		<link id=182></link>
		<text id=183>Patient Lookup</text>
		<link id=184></link>
		<link id=185></link>
		<link id=186></link>
		<link id=187>Date Of Birth</link>
		<link id=188>tv</link>
		<link id=189>Results</link>
		<link id=190>Privacy View is on .</link>
		<link id=191>?</link>
		<button id=192>X</button>
		<link id=193>dh [ 6</link>
		<text id=194></text>
		<text id=195>Recent Patients</text>
		<link id=196>www</link>
		<text id=197></text>
		<text id=198>?</text>
		<link id=199>Q</link>
		<img id=200></img>
		<link id=201></link>
		<text id=202></text>
		<link id=203>Street Address</link>
		<link id=204></link>
		<link id=205>11/11/2013</link>
		<text id=206></text>
		<link id=207></link>
		<link id=208></link>
		<link id=209></link>
		<input id=210>Recent Patients</input>
		<link id=211>0 Find Patient</link>
		<text id=212></text>
		<input id=213>Street Address NHS No.</input>
		<link id=214>DNA</link>
		<text id=215>TAN , CHRISTOPHER Privacy View is on .</text>
		<link id=216></link>
		<link id=217></link>
		<link id=218>Start Time</link>
		<text id=219>Hold MDT</text>
		<img id=220></img>
		<img id=221></img>
		<link id=222></link>
		<link id=223>1</link>
		<img id=224></img>
		<link id=225>MRN</link>
		<text id=226>Language : English</text>
		<link id=227>All Done !</link>
		<text id=228></text>
		<link id=229></link>
		<link id=230>F</link>
		<link id=231>42</link>
		<link id=232>/</link>
		<link id=233></link>
		<text id=234></text>
		<link id=235></link>
		<input id=236>X Cancel</input>
		<button id=237></button>
		========================
		Answer: 1

		Example screen 3:
		========================
		<text id=0>Search for a Patient</text>
		<text id=1>01/01/2016</text>
		<text id=2>14/07/2015</text>
		<text id=3>England</text>
		<text id=4>England</text>
		<text id=5>01/01/2010</text>
		<text id=6>06/11/2010</text>
		<link id=7></link>
		<text id=8>England</text>
		<link id=9>L</link>
		<link id=10></link>
		<text id=11>Forename</text>
		<input id=12></input>
		<text id=13>2002323</text>
		<text id=14>10.57</text>
		<input id=15>20</input>
		<link id=16>Privacy View</link>
		<link id=17>MDT Conference</link>
		<link id=18></link>
		<link id=19></link>
		<link id=20></link>
		<text id=21>M</text>
		<link id=22></link>
		<input id=23></input>
		<link id=24></link>
		<link id=25></link>
		<text id=26>2002321</text>
		<link id=27></link>
		<text id=28>Surname</text>
		<link id=29>Edit Assignments</link>
		<text id=30>2002326</text>
		<text id=31>10.57</text>
		<text id=32>2002325</text>
		<text id=33>NHS No.</text>
		<text id=34>Legal Sex</text>
		<link id=35>PROOF OF CONCEPT ( PO</link>
		<link id=36>Answer Pt - Qnr ( Captive )</link>
		<text id=37>Phone</text>
		<link id=38>➡Create New Patient</link>
		<link id=39>tv</link>
		<link id=40>Open Slots</link>
		<input id=41>O</input>
		<link id=42>U</link>
		<link id=43>Slicer Dicer</link>
		<link id=44>Personalise</link>
		<link id=45>My Incomplete Notes 1</link>
		<link id=46>Letter Queue 15</link>
		<text id=47>MRN</text>
		<link id=48>Procedure Catalogue</link>
		<input id=49></input>
		<input id=50></input>
		<link id=51>Sign My Visits</link>
		<text id=52>ID Type</text>
		<link id=53>Devices</link>
		<link id=54>MDT</link>
		<text id=55>01234567890</text>
		<text id=56>11 Little Lane , Littlehampton , LH12 9UY , England</text>
		<link id=57>Status Board</link>
		<link id=58>16</link>
		<button id=59>0 Find Patient</button>
		<link id=60>Reschedule MDT Meeting</link>
		<input id=61>0</input>
		<link id=62>Epic</link>
		<link id=63>Check In</link>
		<link id=64></link>
		<link id=65>Schedule</link>
		<link id=66></link>
		<text id=67>Street Address</text>
		<link id=68></link>
		<text id=69>Sex</text>
		<text id=70>No phone numbers on file</text>
		<link id=71>Orders Only</link>
		<text id=72>M</text>
		<text id=73>10.57</text>
		<link id=74>ISLETS TRANSPLANT EVALUATION ACC ...</link>
		<link id=75>Telephone Call</link>
		<link id=76>www</link>
		<link id=77>Tel Encounter</link>
		<text id=78>NHS No.</text>
		<text id=79>Patient</text>
		<text id=80>MRN</text>
		<link id=81>KIDNEY TRANSPLANT EVALUATION ACC ...</link>
		<text id=82>Birthdate</text>
		<link id=83>Service Task</link>
		<input id=84>□</input>
		<text id=85>No e - mail address on file</text>
		<text id=86>Phone</text>
		<link id=87>JUL 25</link>
		<link id=88></link>
		<link id=89>4</link>
		<link id=90></link>
		<link id=91></link>
		<link id=92>Remind Me</link>
		<link id=93>CT</link>
		<link id=94>HEART TRANSPLANT EVALUATION ACCO</link>
		<link id=95>View</link>
		<link id=96>Notes</link>
		<link id=97></link>
		<link id=98>Citrix Viewer</link>
		<link id=99>DECEASED</link>
		<link id=100>Check Out</link>
		<link id=101>Abstract</link>
		<link id=102>3+</link>
		<img id=103></img>
		<text id=104>NHS Number : Not on file</text>
		<text id=105>No date of birth on file</text>
		<link id=106>Secure</link>
		<link id=107>Go to In Basket for other tasks</link>
		<text id=108>Privacy View is on</text>
		<text id=109>< E6340 ></text>
		<link id=110>INTESTINE TRANSPLANT EVALUATION A</link>
		<link id=111>☎</link>
		<text id=112>F</text>
		<link id=113>Print Print</link>
		<link id=114></link>
		<link id=115></link>
		<text id=116>Date of Birth</text>
		<link id=117></link>
		<link id=118></link>
		<link id=119>Video Visit</link>
		<img id=120></img>
		<text id=121>Service Area</text>
		<text id=122>No GP on file</text>
		<link id=123></link>
		<link id=124>EpicCare</link>
		<link id=125>NHS Number : Not on file</link>
		<link id=126></link>
		<text id=127>Start Time</text>
		<link id=128>✓Clear</link>
		<link id=129>Hold MDT</link>
		<text id=130>Recent Patients</text>
		<text id=131>F</text>
		<link id=132>42</link>
		<text id=133>CCB 20180812 TEST RSH , Research - < E6340 ></text>
		<link id=134>Log Out</link>
		<text id=135>20</text>
		<link id=136>X Cancel</link>
		<link id=137>G</link>
		<text id=138>TAN , CHRISTOPHER</text>
		<input id=139></input>
		<link id=140>Q</link>
		<link id=141>Tue 25 Jul 11:09</link>
		<link id=142></link>
		<link id=143></link>
		<link id=144>Status</link>
		<img id=145></img>
		<link id=146></link>
		<text id=147>Bridge Four , Shattered Plains , ZZ99 3WZ , Engl</text>
		<link id=148>Patient Lookup</link>
		<button id=149>Results</button>
		<text id=150>Postcode</text>
		<link id=151>X Sign Encounter</link>
		<text id=152>CCB 20180812 TEST RSH , RESEARCH ( ak ...</text>
		<link id=153>Patient Encounter</link>
		<text id=154>Aliases : 20180812</text>
		<link id=155>Chart</link>
		<link id=156></link>
		<link id=157>Results</link>
		<link id=158>Print AVS</link>
		<link id=159></link>
		<text id=160>Perform a search on the local patient index</text>
		<text id=161>3 /</text>
		<text id=162>My Patients</text>
		<text id=163>Legal Sex : Not on file</text>
		<text id=164>10.57</text>
		<text id=165>Gender Identity : Not on file</text>
		<text id=166>DNA</text>
		<link id=167></link>
		<link id=168></link>
		<link id=169></link>
		<text id=170>England</text>
		<link id=171></link>
		<text id=172>Patient Encounter</text>
		<input id=173>ID Type O</input>
		<link id=174></link>
		<link id=175></link>
		<link id=176>Patient Name</link>
		<link id=177></link>
		<img id=178></img>
		<link id=179></link>
		<link id=180></link>
		<link id=181>✓ Accept</link>
		<link id=182></link>
		<link id=183></link>
		<link id=184></link>
		<img id=185></img>
		<link id=186></link>
		<button id=187>✓ Accept</button>
		<link id=188></link>
		<link id=189>X</link>
		<text id=190>All Done !</text>
		<link id=191></link>
		<link id=192>L</link>
		<link id=193>www</link>
		<link id=194></link>
		<link id=195></link>
		<text id=196>Patient Lookup</text>
		<text id=197>Patient Name</text>
		<link id=198></link>
		<link id=199>NA</link>
		<link id=200>tv</link>
		<img id=201></img>
		<link id=202>?</link>
		<link id=203>dh [ 6</link>
		<button id=204>X</button>
		<link id=205>Date of Birth</link>
		<link id=206>www</link>
		<link id=207>GREAT ORMOND STREET ...</link>
		<text id=208></text>
		<text id=209>Match</text>
		<text id=210>?</text>
		<link id=211>Q</link>
		<link id=212>Privacy View is on .</link>
		<link id=213></link>
		<text id=214></text>
		<link id=215></link>
		<text id=216>GREAT ORMOND STREET ...</text>
		<text id=217></text>
		<link id=218></link>
		<text id=219>TAN , CHRISTOPHER Privacy View is on .</text>
		<link id=220>Gender Identity : Not on file</link>
		<link id=221></link>
		<text id=222></text>
		<text id=223>10.57</text>
		<link id=224>DNA</link>
		<text id=225>Bridge Four</text>
		<link id=226>0 Find Patient</link>
		<link id=227>Patient Encounter</link>
		<text id=228>E6340 ></text>
		<link id=229></link>
		<link id=230>All Done !</link>
		<text id=231>F</text>
		<link id=232></link>
		<text id=233>Hold MDT</text>
		<img id=234></img>
		<img id=235></img>
		<text id=236>11 Little Lane ,</text>
		<text id=237>10.57</text>
		<link id=238>1</link>
		<link id=239></link>
		<img id=240></img>
		<img id=241></img>
		<link id=242></link>
		<text id=243></text>
		<link id=244>42</link>
		<link id=245>TAN , CHRISTOPHER</link>
		<link id=246>/</link>
		<text id=247></text>
		<link id=248></link>
		<link id=249>Start Time</link>
		<link id=250>Patient Encounter</link>
		<link id=251></link>
		<link id=252></link>
		========================
		Answer: 2

		The current screen now follows:
		'''

		url = "https://api.openai.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + openai.api_key
		}

		conversation = [{"role": "system", "content": sys_instr}]
		conversation.append({"role": "user", "content": parsed_screen})

		payload = {
		"model": "gpt-4-32k",
		"messages": conversation,
		"temperature": 0,
		"max_tokens": 1
		# "stop": "\n"
		}
		response = requests.post(url, headers=headers, json=payload)
		if response.status_code == 200:
			suggested_command = response.json()["choices"][0]["message"]["content"]
			usage = response.json()["usage"]
			return suggested_command, usage
		else:
			print(f"Error: {response.status_code} - {response.text}")

	def new_consultation_mrn(self):
		while True:
			# screenshot and parse current screen
			parsed_screen = parse_screen()
			current_screen = match_screen(parsed_screen)
			txt.insert(END, "\n" + "OSLER -> The current epic screen is: " + current_screen)
			self.activate_application('Citrix Viewer')
			if current_screen == 'schedule':
				# press f10 for search activities bar
				pyautogui.press('f10')
				time.sleep(2)

				# search for write note activity
				pyperclip.copy('write')
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')
				time.sleep(2)

				# press enter to select write note activity
				pyautogui.press('enter')
				time.sleep(2)
			if current_screen == 'patient_lookup':
				print('global_mrn: ', self.global_mrn)
				pyperclip.copy(self.global_mrn)
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')
				time.sleep(2)

				pyautogui.press('enter')
				time.sleep(1)

				# at this point there are three different possible outcomes so need to use UIED to check and handle
				mrn_search_outcome, usage = self.glance_patient_search_results()
				print('mrn search outcome: ', mrn_search_outcome)
				if mrn_search_outcome == '1':
					txt.insert(END, "\n" + "OSLER -> Great! This MRN matches exactly one patient")
					pyautogui.press('enter')
					time.sleep(1)
					pyautogui.press('enter')
					time.sleep(5)
				elif mrn_search_outcome == '2':
					txt.insert(END, "\n" + "OSLER -> Sorry, this MRN matches more than one patient.")
					break
				elif mrn_search_outcome == '3':
					txt.insert(END, "\n" + "OSLER -> Sorry, this MRN does not match any patient. Please try again.")
					break
				else:
					print('error with processing the result from glancing')

			if current_screen == 'chart_review':
				# ctrl space
				pyautogui.keyDown('ctrl')
				pyautogui.press('space')
				pyautogui.keyUp('ctrl')
				time.sleep(2)

				# search for write note activity
				pyperclip.copy('write note')
				pyautogui.keyDown('command')
				pyautogui.press('v')
				pyautogui.keyUp('command')
				time.sleep(2)

				# select write note activity
				pyautogui.press('down')
				time.sleep(1)
				pyautogui.press('enter')
				time.sleep(2)
				pyautogui.press('enter')
				time.sleep(5)

			if current_screen == 'documentation':
				# click the create note button
				self.click_screenshot('create_note.png', confidence=0.6)
				time.sleep(2)
				pyautogui.press('f3')
				time.sleep(2)

				# release the function button
				pyautogui.keyUp('fn')
				time.sleep(1)

				# add smart text medicines and problem list
				pyautogui.write('.med', interval=0.1)
				time.sleep(1)
				pyautogui.press('enter')
				time.sleep(1)

				# add smart text medicines and problem list
				pyautogui.write('.diagprob', interval=0.1)
				time.sleep(1)
				pyautogui.press('enter')
				time.sleep(1)

				# copying the patient medical history and medications and saving to memory
				pyautogui.keyDown('command')
				pyautogui.press('a')
				pyautogui.keyUp('command')

				time.sleep(1)

				pyautogui.keyDown('command')
				pyautogui.press('c')
				pyautogui.keyUp('command')

				time.sleep(0.5)
				pyautogui.press('right')

				self.med_hx = pyperclip.paste()
				break


	def transcribe_consultation(self):
		# add header
		pyperclip.copy('\n\n--------- Consultation Transcription ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		self.transcribe()

		# stop recording banner
		pyperclip.copy('\n\n--------- Recording Stopped ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		self.summarise_transcription()

		self.consultation_entities = ask_gpt(self.consultation_transcript)
		print('extracted metadata from consultation')

		print(str(self.consultation_entities))

	def place_orders(self):
		orders_list = self.get_orders_from_gpt_call(self.consultation_entities)
		for order in orders_list:
			pyautogui.keyDown('command')
			pyautogui.press('o')
			pyautogui.keyUp('command')
			time.sleep(1)

			pyperclip.copy(order)
			pyautogui.keyDown('command')
			pyautogui.press('v')
			pyautogui.keyUp('command')
			time.sleep(0.5)

			pyautogui.press('enter')
			time.sleep(1)
			pyautogui.press('enter')

			pyautogui.keyDown('option')
			pyautogui.keyDown('command')
			pyautogui.press('v')
			pyautogui.keyUp('command')
			pyautogui.keyUp('option')
			time.sleep(1)

	def file_diagnoses(self):
		diagnosis_list = self.get_diagnoses_from_gpt_call(self.consultation_entities)
		pyautogui.keyDown('command')
		pyautogui.press('g')
		pyautogui.keyUp('command')
		time.sleep(1)
		
		for diagnosis in diagnosis_list:
			pyperclip.copy(diagnosis)
			pyautogui.keyDown('command')
			pyautogui.press('v')
			pyautogui.keyUp('command')
			time.sleep(1)

			pyautogui.press('enter')
			time.sleep(1)
			pyautogui.press('enter')
			time.sleep(1)

		pyautogui.press('escape')

	def summarise_transcription(self):
		url = "https://api.openai.com/v1/chat/completions"
		headers = {
			"Content-Type": "application/json",
			"Authorization": "Bearer " + openai.api_key
		}

		system_instruction = '''
		You are a medical office assistant drafting documentation for a physician. You will be provided with a MEDICAL HISTORY and a CONSULTATION TRANSCRIPT. DO NOT ADD any content that isn't specifically mentioned in the CONSULTATION TRANSCRIPT or the MEDICAL HISTORY. From the attached transcript and medical history, generate a SOAP note based on the below template format for the physician to review, include all the relevant information and do not include any information that isn't explicitly mentioned in the transcript.If nothing is mentioned just returned[NOT MENTIONED].

		Template for Clinical SOAP Note Format:

		Subjective: The “history” section
		- HPI: include any mentioned symptom dimensions, chronological narrative of patients complains, information obtained from other sources(always identify source if not the patient).
		- Pertinent past medical history.
		- Pertinent review of systems mentioned, for example, “Patient has not had any stiffness or loss
		of motion of other joints.”
		- Current medications mentioned(list with daily dosages).
		Objective: The physical exam and laboratory data section
		- Vital signs including oxygen saturation when indicated.
		- Focussed physical exam.
		- All pertinent labs, x - rays, etc.completed at the visit.
		Assessment / Problem List: Your assessment of the patients problems
		- Assessment: A one sentence description of the patient and major problem
		- Problem list: A numerical list of problems identified
		- All listed problems need to be supported by findings in subjective and objective areas above.Try to take the assessment of the major problem to the highest level of diagnosis that you can, for example, “low back sprain caused by radiculitis involving left 5th LS nerve root.”
		- Any differential diagnoses mentioned in the transcript, if not just leave this blank as DIFFERENTIAL DIAGNOSIS:
		Plan: Any plan for the patient mentioned in the transcript
		- Divide any diagnostic and treatment plans for each differential diagnosis.
		- Your treatment plan should include: patient education pharmacotherapy if any, other therapeutic procedures.You must also address plans for follow - up(next scheduled visit, etc.)
		Please provide your response in a bullet point list for each heading.'''

		user_message = SOAP_user_msg_template
		user_message = user_message.replace("$medical_history", self.med_hx)
		user_message = user_message.replace("$consultation_transcript", self.consultation_transcript)

		conversation = [{"role": "system", "content": system_instruction}]
		conversation.append({"role": "user", "content": user_message})

		payload = {
		"model": "gpt-4",
		"messages": conversation,
		"temperature": 0,
		"max_tokens": 500
		# "stop": "\n"
		}
		response = requests.post(url, headers=headers, json=payload)
		if response.status_code == 200:
			suggested_command = response.json()["choices"][0]["message"]["content"]
			usage = response.json()["usage"]
			# return suggested_command, usage
		else:
			print(f"Error: {response.status_code} - {response.text}")

		
		# write consultation summary to notes
		pyperclip.copy(suggested_command)
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

	
	def summarise_transcription1(self):
		# add header
		pyperclip.copy('\n\n--------- Consultation Summary ---------\n\n')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

		# get GPT consultation summary
		meta_consultation_summarisation = 'INSTRUCTION: Summarise the below MEDICAL HISTORY and CONSULTATION TRANSCRIPT between patient and doctor into short notes, under the following headings: 1. Detailed summary of the patient symptoms  2. Medicines 3. Allergies 4. Family History 5. Social History 6. Examination findings 7. Impression 8. Plan\n'
		meta_consultation_summarisation += 'MEDICAL HISTORY: \n' + self.med_hx
		meta_consultation_summarisation += '\nCONSULTATION TRANSCRIPT: \n' + self.consultation_transcript + '\n\n'
		response=openai.Completion.create(
		model="text-davinci-003",
		prompt=meta_consultation_summarisation,
		max_tokens=2500,
		temperature=0
		)
		consultation_summary = json.loads(str(response))
		consultation_summary = consultation_summary['choices'][0]['text']
		self.transcript_summary = consultation_summary

		# write consultation summary to notes
		pyperclip.copy(consultation_summary)
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')

	def get_orders_from_gpt_call(self, response):
		orders_ls = []
		for order in response['orders']:
			orders_ls.append(order['name'])
		return orders_ls
	
	def get_diagnoses_from_gpt_call(self, response):
		diagnoses_ls = []
		for diagnosis in response['visit_diagnoses']:
			diagnoses_ls.append(diagnosis['name'])
		return diagnoses_ls
	
	def get_meds_from_gpt_call(self, response):
		meds_ls = []
		for med in response['medicine']:
			meds_ls.append(med['name'])
		return meds_ls

	def speak_orders_list(self, orders_list):
		# The text to be converted into audio
		text = 'The orders I got from the consultation were '
		for i in range(len(orders_list)):
			text += orders_list[i]
			if i < len(orders_list) - 1:
				text += ' and '
		return text

	def speak_meds_list(self, meds_list):
		# The text to be converted into audio
		text = 'The medicines I got from the consultation were '
		for i in range(len(meds_list)):
			text += meds_list[i]
			if i < len(meds_list) - 1:
				text += ' and '
		return text

	def query_orders(self):
		# get the list of orders extracted from the consultation
		orders_list = self.get_orders_from_gpt_call(self.consultation_entities)

		# convert the list of orders into the text to speak
		audio_text = self.speak_orders_list(orders_list)

		# create the audio file from the text
		# Language in which you want to convert
		language = 'en'
		
		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
		myobj = gTTS(text=audio_text, lang=language, slow=False)
		
		# Saving the converted audio in a mp3 file named
		myobj.save("extracted_orders_list.wav")

		# get the length of the audio file to know how long to sleep while playing
		audio = eyed3.load("extracted_orders_list.wav")
		length_in_seconds = int(audio.info.time_secs)

		#play the audio file
		PLAYER.play_song("extracted_orders_list.wav")
		time.sleep(length_in_seconds + 1)

	def query_meds(self):
		# get the list of orders extracted from the consultation
		meds_list = self.get_meds_from_gpt_call(self.consultation_entities)

		# convert the list of orders into the text to speak
		audio_text = self.speak_meds_list(meds_list)

		# create the audio file from the text
		# Language in which you want to convert
		language = 'en'
		
		# Passing the text and language to the engine, 
		# here we have marked slow=False. Which tells 
		# the module that the converted audio should 
		# have a high speed
		myobj = gTTS(text=audio_text, lang=language, slow=False)
		
		# Saving the converted audio in a mp3 file named
		myobj.save("extracted_meds_list.wav")

		# get the length of the audio file to know how long to sleep while playing
		audio = eyed3.load("extracted_meds_list.wav")
		length_in_seconds = int(audio.info.time_secs)

		#play the audio file
		PLAYER.play_song("extracted_meds_list.wav")
		time.sleep(length_in_seconds + 1)

	def write_referral(self):
		self.activate_application('Citrix Viewer')

		# press f10 for search activities bar
		pyautogui.press('f10')
		time.sleep(2)

		# search for write note activity
		pyperclip.copy('letter')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
		pyautogui.press('enter')
		time.sleep(3)

		# input MRN 111
		pyperclip.copy(self.global_mrn)
		pyperclip.copy('111')
		pyautogui.keyDown('command')
		pyautogui.press('v')
		pyautogui.keyUp('command')
		time.sleep(3)

		# press enter 3 times
		pyautogui.press('enter')
		time.sleep(3)
		pyautogui.press('enter')
		time.sleep(3)
		pyautogui.press('enter')
		time.sleep(8)

		# select clinic letter
		self.click_screenshot("select_clinic_letter.png", confidence=0.6)
		time.sleep(3)

		# add recipient as patient 1
		pyautogui.keyDown('command')
		pyautogui.keyDown('option')
		pyautogui.press('1')
		time.sleep(3)

		#play the letter pending audio file
		# PLAYER.play_song("letter_pending.wav")
		# time.sleep(4)

		# # get GPT to write referral letter
		# referral_letter_prompt = 'Write a letter to the patients GP including all of the following information, include the patients background medical history, medications, a summary of the consultation and a plan:\n\n'
		# referral_letter_prompt += self.transcript_summary

		# response=openai.Completion.create(
		# model="text-davinci-003",
		# prompt=referral_letter_prompt,
		# max_tokens=1500,
		# temperature=0
		# )

		# referral_letter = json.loads(str(response))
		# print(referral_letter['choices'][0]['text'])


		# # select clinic letter text area
		# self.click_screenshot("clinic_letter_box.png")
		# time.sleep(1)
		# self.click_screenshot("clinic_letter_box.png")
		# time.sleep(0.5)
		# pyperclip.copy(referral_letter['choices'][0]['text'])
		# pyautogui.keyDown('command')
		# pyautogui.press('v')
		# pyautogui.keyUp('command')
 



def init_ml_client(subscription_id, resource_group, workspace):
	return MLClient(
		DefaultAzureCredential(), subscription_id, resource_group, workspace
	)

ml_client = init_ml_client(
	"af5d9edb-37c3-40a4-a58f-5b97efbbac8d",
	"hello-rg",
	"osler-perception"
)

def read_image(path_to_image):
	with open(path_to_image, "rb") as f:
		return f.read()

def predict_image_object_detection_sample(
		ml_client,
		endpoint_name,
		deployment_name,
		path_to_image
):
	request_json = {
		"image" : base64.encodebytes(read_image(path_to_image)).decode("utf-8")
	}	

	request_fn = "request.json"

	with open(request_fn, "w") as request_f:
		json.dump(request_json, request_f)

	response = ml_client.online_endpoints.invoke(
		endpoint_name=endpoint_name,
		deployment_name=deployment_name,
		request_file=request_fn
	)

	detections = json.loads(response)

	return detections

def detect_text(path):
	"""Detects text in the file."""
	client = vision.ImageAnnotatorClient(credentials=credentials)

	with open(path, 'rb') as image_file:
		content = image_file.read()

	image = vision.Image(content=content)

	response = client.text_detection(image=image)
	texts = response.text_annotations
	# print('Texts:')

	# for text in texts:
	#     # print(f'\n"{text.description}"')

	#     vertices = ([f'({vertex.x},{vertex.y})'
	#                 for vertex in text.bounding_poly.vertices])

	#     # print('bounds: {}'.format(','.join(vertices)))

	if response.error.message:
		raise Exception(
			'{}\nFor more info on error messages, check: '
			'https://cloud.google.com/apis/design/errors'.format(
				response.error.message))
		
	return response

# not including bboxes just yet
def html_from_UIE(df_row, idx):
	elem_type = df_row['displayNames']
	bbox = df_row['bboxes']
	inner_text = df_row['predicted text']
	html = f"""<{elem_type} id={idx}>{inner_text}</{elem_type}>"""
	return html

def df_to_html(df):
	s = ''
	for index, row in df.iterrows():
		s += html_from_UIE(row, index) + '\n'
	return s

def bb_intersection_over_minArea(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / min(boxAArea, boxBArea)
	# return the intersection over union value
	return iou

def strls2str(strls):
	s = ''
	for elem in strls:
		s += elem + ' '
	return s[:-1]

def add_text_to_uie(response, ocr):
	conf_threshold = 0
	i = 0

	ids = []
	texts = []
	labels = []
	bboxes = []

	for detection in response["boxes"]:
		if detection["score"] < conf_threshold:
			continue
		text = []
		box = detection["box"]
		x_min, y_min, x_max, y_max = (
			box["topX"],
			box["topY"],
			box["bottomX"],
			box["bottomY"]
		)
		uie_box = [
			x_min * 1280, y_min * 1080, x_max * 1280, y_max * 1080
		]
		for annotation in ocr.text_annotations[1:]:
			top_left = annotation.bounding_poly.vertices[0]
			bottom_right = annotation.bounding_poly.vertices[2]
			ocr_box = [top_left.x, top_left.y, bottom_right.x, bottom_right.y]
			iou = bb_intersection_over_minArea(uie_box, ocr_box)
			if iou > 0.8:
				text.append(annotation.description)   
		text = strls2str(text)

		ids.append(i)
		texts.append(text)
		labels.append(detection["label"])
		bboxes.append([x_min, y_min, x_max, y_max])

		i += 1

	response_df = pd.DataFrame.from_dict({
		"displayNames": labels,
		"bboxes": bboxes,
		"predicted text": texts
	})
	return response_df

def parse_screen():
		print('parsing screen...')
		current_screen = ImageGrab.grab()  # Take the screenshot
		screen_size = current_screen.size
		current_screen = current_screen.resize((RESIZE_WIDTH,RESIZE_HEIGHT))
		current_screen.save('current_screen.png')

		# send screenshot to UIED model to get UIEs
		# print('sending screenshot to tortus UIED model...')
		response = predict_image_object_detection_sample(
			ml_client,
			endpoint_name="uied",
			deployment_name="yolov5",
			path_to_image="current_screen.png"
		)

		# send screenshot to Google OCR to get text
		# print('sending screenshot to google OCR...')
		ocr = detect_text('current_screen.png')

		# merge OCR with UIEs
		# print('merging OCR and UIED...')
		merged_df = add_text_to_uie(response, ocr)
		merged_df.to_csv('uied.csv')
				
		# covert to LLM template format
		# print('converting to LLM template format from dataframe...')
		llm_format = df_to_html(merged_df)
		
		return llm_format

def match_intent(utterance):
	model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

	intent_ls = [
		intents.START_CONSULTATION_NOTE,
		intents.TRANSCRIBE_CONSULTATION,
		intents.SUMMARISE_CONSULTATION,
		intents.PLACE_ORDERS,
		intents.FILE_DIAGNOSES,
		intents.ANSWER_QUESTIONS,
		intents.WRITE_LETTER,
		intents.QUERY_MEDS,
		intents.QUERY_ORDERS,
]

	intent_embeddings = model.encode(intent_ls)
	utterance_embeddings = model.encode(utterance)
	
	cos_scores = cosine_similarity(utterance_embeddings.reshape(1, -1), intent_embeddings)
	cos_scores_torch = torch.from_numpy(cos_scores)
	cos_max = torch.max(cos_scores_torch).item()
	cos_argmax = torch.argmax(cos_scores_torch, dim=1)
	cos = cos_argmax[0].item()

	intent = intent_ls[cos]
	print(f"Intent matched: {intent}")

	return intent, cos_max

def match_screen(current_screen):
	model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")

	screens_ls = [
		epic_screens.PATIENT_LOOKUP,
		epic_screens.SCHEDULE,
		epic_screens.CHART_REVIEW,
		epic_screens.DOCUMENTATION
	]

	screen_labels = ['patient_lookup', 'schedule', 'chart_review', 'documentation']

	epic_embeddings = model.encode(screens_ls)
	screen_embeddings = model.encode(current_screen)

	cos_scores = cosine_similarity(screen_embeddings.reshape(1, -1), epic_embeddings)
	cos_scores_torch = torch.from_numpy(cos_scores)
	cos_max = torch.max(cos_scores_torch).item()
	cos_argmax = torch.argmax(cos_scores_torch, dim=1)
	cos = cos_argmax[0].item()


	print(cos_scores)
	intent = screens_ls[cos]
	screen_name = screen_labels[cos]

	return screen_name

 
# Send function
def send():
	msg = "You -> " + e.get()
	txt.insert(END, "\n" + msg)
 
	user = e.get().lower()
	e.delete(0, END)
		
	# Run the rest(user) function asynchronously using a thread
	threading.Thread(target=msg2task, args=(user,)).start()
	# rest(user)

def msg2task(user_msg):
	# match the user command to intents
	intent, score = match_intent(user_msg)
	print(score)

	# if matched intent is starting a new consult note, attempt extract mrn from user message
	if intent == 'start a consultation note':
		actor.global_mrn = actor.extract_mrn_from_text(user_msg)
		print('mrn: ', actor.global_mrn)
	
	# display matched intent to user
	osler_message = "It looks like you asked me to perform the task: "
	txt.insert(END, "\n" + "OSLER -> " + osler_message + intent)
	# e.delete(0, END)
		
	# perform task
	actor.act(intent)

# GUI
root = Tk()
root.title("OSLER")
 
BG_GRAY = "#ABB2B9"
BG_COLOR = "#17202A"
TEXT_COLOR = "#EAECEE"
 
FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 1080
# DEVICE_SIZE = (1440, 900)
DEVICE_SIZE = (1791, 1119)

from google.oauth2 import service_account
credentials = service_account.Credentials.from_service_account_file('./tortus-374118-e15fd1ca5b60.json')


# to prevent the huggingface tokenizer parallelisation error
os.environ["TOKENIZERS_PARALLELISM"] = "false"

actor = Actor()

global_mrn = ''

lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="OSLER", font=FONT_BOLD, pady=10, width=20, height=1).grid(
	row=0)
 
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)
 
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)
 
e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)
 
send_button = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
			  command=send).grid(row=2, column=1)
 
root.mainloop()