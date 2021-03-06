import cv2
import numpy as np

from skimage import io, color

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture("output.avi")

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
                    frame[i][j][0] = 0
                    frame[i][j][1] = 0

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