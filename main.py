#code created by Aaron Shenny for INTEL bootcamp PROJECT
#Read the REEADME file before opening this.

import cv2
from matplotlib import pyplot as plt

# Load the cascade
stopSignCascade = cv2.CascadeClassifier('./face.xml')

# Initialize video capture from the default camera
cap = cv2.VideoCapture(0)
print('Caputuring Started....')
while True:
    # Read the frame from the camera
    ret, frame = cap.read()

    if not ret:
        # If the frame cannot be captured, break the loop
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect stop signs in the frame
    stopSigns = stopSignCascade.detectMultiScale(gray)

    # Draw rectangles around the detected stop signs and label as "person"
    for (x, y, width, height) in stopSigns:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (230, 255, 78), 5)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 255, 78), 2)

    # Display the frame
    cv2.imshow('Face Detection', frame)

    # Check for keyboard input; if 'q' is pressed, exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
print('Capuring Stopped')