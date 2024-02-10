import face_recognition
import cv2
import numpy as np
import csv 
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces

Aadithiyan_image = face_recognition.load_image_file("Aadithiyan.jpg")
Aadithiyan_encoding = face_recognition.face_encodings(Aadithiyan_image)[0]

swetha_image = face_recognition.load_image_file("swetha.jpeg")
swetha_encoding = face_recognition.face_encodings(swetha_image)[0]

prabu_image = face_recognition.load_image_file("prabu.jpg")
prabu_encoding = face_recognition.face_encodings(prabu_image)[0]

dinesh_image = face_recognition.load_image_file("dinesh.jpg")
dinesh_encoding = face_recognition.face_encodings(dinesh_image)[0]

bhuvanesh_image = face_recognition.load_image_file("bhuvanesh.jpg")
bhuvanesh_encoding = face_recognition.face_encodings(bhuvanesh_image)[0]

harish_image = face_recognition.load_image_file("harish.jpg")
harish_encoding = face_recognition.face_encodings(harish_image)[0]

hemanth_image = face_recognition.load_image_file("hemanth.jpg")
hemanth_encoding = face_recognition.face_encodings(hemanth_image)[0]

ruthika_image = face_recognition.load_image_file("ruthika.jpg")
ruthika_encoding = face_recognition.face_encodings(ruthika_image)[0]


known_face_encodings = [Aadithiyan_encoding ,swetha_encoding, prabu_encoding , dinesh_encoding,bhuvanesh_encoding,harish_encoding,hemanth_encoding,ruthika_encoding]
known_face_names = ["Aadithiyan" ,"swetha", "prabu" , "dinesh","bhuvanesh","harish","hemanth","ruthika"]

# List of expected students

students = known_face_names.copy()

face_locations = []
face_encodings = []

# current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

#open csv for writing 
f = open(f"{current_date}.csv" , "w+" , newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame ,(0,0) , fx=0.25 , fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame , cv2.COLOR_BGR2RGB) 
    
    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame , face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)
        
        if(matches[best_match_index]):
            name = known_face_names[best_match_index]
            
            # Add the text if a person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale = 1.5
                fontColor = (255,0,0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)
                
                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H:%M:%S")
                    lnwriter.writerow([name,current_time])
    
    cv2.imshow("Camera" , frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
video_capture.release()
cv2.destroyAllWindows()
f.close()
