import cv2
import time
# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Open the webcam
cap = cv2.VideoCapture(0)

# Variables for capturing every 1 seconds
capture_interval = 1  # in seconds
last_capture_time = time.time()

# Create a variable to count the captured frames
frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Only capture frames if face with eyes are detected
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray=gray[y:y+h,x:x+w]
            roi_color=frame[y:y+h,x:x+h]
            eyes=eye_cascade.detectMultiScale(roi_gray)
            for(ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,200,0),1)

        # Capture the frame if 1 seconds have passed since the last capture
        if time.time() - last_capture_time >= capture_interval:
            frame_count += 1
            capture_name = f"captured_frame_{frame_count}.jpg"
            cv2.imwrite(capture_name, frame)
            print(f"Captured {capture_name}")
            last_capture_time = time.time()

    # Display the frame with rectangles around detected faces
    cv2.imshow('Face Detection', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
