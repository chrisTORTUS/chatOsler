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
import epic_screens
import actor
 
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
		print('taking screenshot...')
		current_screen = ImageGrab.grab()  # Take the screenshot
		screen_size = current_screen.size
		current_screen = current_screen.resize((RESIZE_WIDTH,RESIZE_HEIGHT))
		current_screen.save('current_screen.png')

		# send screenshot to UIED model to get UIEs
		print('sending screenshot to tortus UIED model...')
		response = predict_image_object_detection_sample(
			ml_client,
			endpoint_name="uied",
			deployment_name="yolov5",
			path_to_image="current_screen.png"
		)

		# send screenshot to Google OCR to get text
		print('sending screenshot to google OCR...')
		ocr = detect_text('current_screen.png')

		# merge OCR with UIEs
		print('merging OCR and UIED...')
		merged_df = add_text_to_uie(response, ocr)
		merged_df.to_csv('uied.csv')
				
		# covert to LLM template format
		print('converting to LLM template format from dataframe...')
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
		epic_screens.PATIENT_PAGE
	]

	screen_labels = ['patient lookup', 'schedule', 'patient info']

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
	screen_name = screen_labels[cos]

	return screen_name

 
# Send function
def send():
	send = "You -> " + e.get()
	txt.insert(END, "\n" + send)
 
	user = e.get().lower()
	e.delete(0, END)

	# match the user command to intents
	intent, score = match_intent(user)
	print(score)
	
	# display matched intent to user
	osler_message = "It looks like you asked me to perform the task: "
	txt.insert(END, "\n" + "OSLER -> " + osler_message + intent)
	# e.delete(0, END)
	
	# screenshot and parse current screen
	parsed_screen = parse_screen()
	matched_screen = match_screen(parsed_screen)

	# display current screen to user
	osler_message = "The current screen is: "
	txt.insert(END, "\n" + "OSLER -> " + osler_message + matched_screen)


	


lable1 = Label(root, bg=BG_COLOR, fg=TEXT_COLOR, text="OSLER", font=FONT_BOLD, pady=10, width=20, height=1).grid(
    row=0)
 
txt = Text(root, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT, width=60)
txt.grid(row=1, column=0, columnspan=2)
 
scrollbar = Scrollbar(txt)
scrollbar.place(relheight=1, relx=0.974)
 
e = Entry(root, bg="#2C3E50", fg=TEXT_COLOR, font=FONT, width=55)
e.grid(row=2, column=0)
 
send = Button(root, text="Send", font=FONT_BOLD, bg=BG_GRAY,
              command=send).grid(row=2, column=1)
 
root.mainloop()