import cv2
import dlib
import numpy as np
from keras.models import load_model
from imutils import face_utils
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#detect face
def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    if len(rects) == 0:
        return []
    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]
    return rects

#detect eyes
def detect_eye(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#grayscale
	gray_face = detect(gray, minimumFeatureSize=(80, 80))

	#return none if no face detected
	if len(gray_face) == 0:
		return None
	elif len(gray_face) > 1:
		face = gray_face[0]
	elif len(gray_face) == 1:
		[face] = gray_face

	# crop face
	face_rect = dlib.rectangle(left=int(face[0]), top=int(face[1]),right=int(face[2]), bottom=int(face[3]))

	#crop eyes
	shape = predictor(gray, face_rect)

	x1 = shape.part(36).x
	x2 = shape.part(39).x
	y1 = shape.part(37).y
	y2 = shape.part(40).y
	left_eye = gray[y1-10:y2+10,x1-10:x2+10]

	x1 = shape.part(42).x
	x2 = shape.part(45).x
	y1 = shape.part(43).y
	y2 = shape.part(46).y
	right_eye = gray[y1-10:y2+10,x1-10:x2+10]

	left_eye = cv2.resize(left_eye, (34, 26))
	right_eye = cv2.resize(right_eye, (34, 26))
	right_eye = cv2.flip(right_eye, 1)

	return left_eye, right_eye


def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img

camera = cv2.VideoCapture(0)
model = load_model('blinkModel.hdf5')

# blinks is the number of total blinks
#state is 'open', 'close' (1 and 0)
# blink_list is list of blink state (0,1,0,1,...)

blinks = 0
blinks_list = []
state = ''
while True:
	ret, frame = camera.read()


	# detect eyes
	eyes = detect_eye(frame)
	if eyes is None:
		continue
	else:
		left_eye, right_eye = eyes

	# predictions of the two eyes
	prediction_left = model.predict(cnnPreprocess(left_eye))
	prediction_right = model.predict(cnnPreprocess(right_eye))

	# blinks counter
	if prediction_left >= 0.5 and prediction_right >= 0.5:
		state = 'open'
	if prediction_left < 0.5 and prediction_right < 0.5:
		state = 'close'
	if (len(blinks_list) == 0 or blinks_list[-1] == 0) and state == 'open':  # open after close eye
		blinks_list.append(1)
		if len(blinks_list) >= 3:  # if 3 latest elements follow pattern 'open > close > open' (1>0>1) => count
			blinks += 1
	if (len(blinks_list) == 0 or blinks_list[-1] == 1) and state == 'close':  # close after open eye
		blinks_list.append(0)

	# draw the total number of blinks on the frame along with
	# the state for the frame
	cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	cv2.putText(frame, "State: {}".format(state), (200, 30),
				cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	# show the frame
	cv2.imshow('blinks counter', frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord('q'):
		break