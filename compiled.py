import cv2
import sys
import numpy as np
import math
from skimage import io, color

# Create a VideoCapture object and get video from webcam
# 0 for HD(if connected, otherwise internal), 1 for internal (if HD connected)
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (640,480))

# Create the haar cascade - used for lighting and stuff
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            L = 0
            A = 0
            B = 0
            rows = 0
            cols = 0
            # Draws rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #forehead = frame[y:y+90, x:x+w]
            #cv2.rectangle(frame, (x, y), (x+w, y+80), (0,255,0),2)
            # gets forehead from image and turns into lab
            face_frame = frame[y:y+h, x:x+w]
            rgb_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
            lab_face = color.rgb2lab(rgb_face)

            # Gets dimension of face_frame
            rows = len(lab_face)
            cols = len(lab_face[0])

            # Thresholds: 6N lounge, test_vid1, 2, and 3 = 7
            threshold = 7
            gain = 0.8
            for i in range(rows):
                for j in range(cols):
                    if((lab_face[i][j][1] > threshold)):
                        # print(lab[i][j][1])
                        face_frame[i][j][0] = 0
                        face_frame[i][j][1] = 0
                        # Colors red channel where 255 is light and 0 is darker
                        face_frame[i][j][2] = 255 - (gain * lab_face[i][j][1]) * (255/25)


        # Saves frame to video output
        out.write(frame)
        # Displays frame
        cv2.imshow('Video', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # If receiving input from file, then processes until exit key or has read all frames
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
out.release()
cv2.destroyAllWindows()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
while (cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:


        #rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #lab = color.rgb2lab(rgbFrame)

        numrows = len(frame)
        numcols = len(frame[0])
        for i in range(numrows):
            for j in range(numcols):
                if((frame[i][j][2] <117)):
                    # print(lab[i][j][1])
                    frame[i][j][0] = 0
                    frame[i][j][1] = 0
                    #frame[i][j][2] = 255 - (0.8 * lab[i][j][1]) * (255/25)
                # else   :
                #     frame[i][j][0] = 60
                #     frame[i][j][1] = 60
                #     frame[i][j][2] = 60


        # rgb2 = color.lab2rgb(lab)
        # standardizedFrame = cv2.cvtColor(rgb2, cv2.COLOR_RGB2BGR)
        #
        # print(str(frame[0,0])+"\t"+str(lab[0,0]))


        # Display the resulting frame
        cv2.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

# Create a VideoCapture object and get video from webcam
# 0 for internal, 1 for external
cap = cv2.VideoCapture(0)


# Create the haar cascade - used for lighting and stuff
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")

# Read until video is completed
def run():
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            #print("Found {0} faces!".format(len(faces)))


            #forehead = None
            L = 0
            A = 0
            B = 0

            for (x, y, w, h) in faces:
                rows = 0
                cols = 0

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # gets forehead from image and turns into lab

                # MIDDLE FACE REGION
                forehead = frame[y:y+150, x:x+w]
                rgb_forehead = cv2.cvtColor(forehead, cv2.COLOR_BGR2RGB)
                lab_forehead = color.rgb2lab(rgb_forehead)
                #
                rows = len(lab_forehead)
                cols = len(lab_forehead[0])
                for i in range(rows):
                    for j in range(cols):
                        #for w in range(len(lab_forehead[0][0])):
                        #    print(lab_forehead[i][j][w])
                        L += lab_forehead[i][j][0]
                        A += lab_forehead[i][j][1]
                        B += lab_forehead[i][j][2]
                numpix = rows*cols
                L = L/(numpix)
                A = A/(numpix)
                B = B/(numpix)

                # print("Lightness: ", L)
                # print("green-red: ", A)
                # print("blue-yellow: ", B)
            print(L)
            print(A)
            print(B)
            print("\n")

                # do we want L ??
                # # gets left cheek from image and turns into lab
                # left_cheek = image[y+150:y+h, x:x+100]
                # rgb_left_cheek = cv2.cvtColor(left_cheek, cv2.COLOR_BGR2RGB)
                # lab_left_cheek = color.rgb2lab(rgb_left_cheek)
                # # gets right cheek from image and turns into lab
                # right_cheek = image[y+150:y+h, x+180:x+w]
                # rgb_right_cheek = cv2.cvtColor(right_cheek, cv2.COLOR_BGR2RGB)
                # lab_right_cheek = color.rgb2lab(rgb_right_cheek)

            cv2.imshow('Video', frame)

            #print(forehead)



            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
run()
# When everything done, release the video capture object
cap.release()