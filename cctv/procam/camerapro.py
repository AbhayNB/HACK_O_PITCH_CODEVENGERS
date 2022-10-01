import numpy as np
import face_recognition as fr
import cv2
import mysql.connector as sc
import dlib
from datetime import date
from datetime import time
from datetime import datetime
import time
from pyttsx3 import init
engine=init('sapi5')
#client=wolframalpha.Client('AYGGRU-P3473P9ETG')
voice=engine.getProperty('voices')
engine.setProperty('voice', voice[1].id)
def speak(audio):
    print('JERRY: '+ audio)
    engine.say(audio)
    engine.runAndWait()
con= sc.connect(host='localhost',user='root',passwd='',database='ai_cam')
print(con.is_connected())
cur2=con.cursor()
tid=1
dect=dlib.get_frontal_face_detector()
video_capture = cv2.VideoCapture(0)
bruno_image = fr.load_image_file("sunil_sir.jpg")
bruno_image2 = fr.load_image_file("IMG-20220926-WA0010.jpg")
img_count = 0
face_locations1 = fr.face_locations(bruno_image)
face_locations2 = fr.face_locations(bruno_image2)
bruno_face_encoding = fr.face_encodings(bruno_image,face_locations1)[0]

bruno_face_encoding2 = fr.face_encodings(bruno_image2,face_locations2)[0]
known_face_encondings = [bruno_face_encoding,bruno_face_encoding2]
known_face_names = ["Sunil Sir","Saurabh Sir"]
f_count=0
fe=0
fc_rec={"Saurabh Sir":bruno_face_encoding2}
print(fc_rec.values())
print(bruno_face_encoding2)
p=[]
p.append(bruno_face_encoding2)
t=time.localtime()
q=["Saurabh Sir"]
print(p)
print(q)
entry={}
while True:
    folder= str(date.today())
    dt=datetime.now()
    sdt=datetime.isoformat(dt)
    t=time.localtime()
    ct=time.strftime("%H",t)
    ict=int(ct)
    f_count+=1
    ret, frame = video_capture.read()

    rgb_frame = frame[:, :, ::-1]
    faces = dect(rgb_frame)
    num=len(faces)
    print(num)
    if num != fe:
        print('change')
        fe=num
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        name = "Unknown{}".format(f_count)
        matches = fr.compare_faces(known_face_encondings, face_encoding)
        matches2 = fr.compare_faces(p, face_encoding)
        

        face_distances = fr.face_distance(known_face_encondings, face_encoding)
        face_distances2 = fr.face_distance(p, face_encoding)
        best_match_index = np.argmin(face_distances)
        best_match_index2 = np.argmin(face_distances2)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        elif matches2[best_match_index2]:
            name = q[best_match_index2]
        else:
            q.append(name)
            p.append(face_encodings[0])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        if name not in entry.keys():
            entry[name]= ict
            im_nam= name+"_{}.png".format(ct)
            f_nam= name+"_face_{}.png".format(ct)
            f_frame=frame[top-35:bottom+35,left-35:right+35]
            cv2.imwrite(f_nam,f_frame)
            cv2.imwrite(im_nam,frame)
            stm='''insert into entery(name,image,time)
                values('{}',load_file('{}'),'{}')'''.format(im_nam,'F/cvv/face_rec/{}'.format(folder,im_nam),sdt)
            cur2.execute(stm)
            speak('.  !.    ......  hi '+'Wellcome'+name)
            con.commit()
        


        elif entry[name]+1== ict:
            entry[name]= ict
            im_nam= name+"_{}.png".format(ct)
            cv2.imwrite(im_nam,frame)
            f_nam = name + "_face_{}.png".format(ct)
            f_frame = frame[top-35:bottom+35, left-35:right+35]
            cv2.imwrite(f_nam, f_frame)
            stm='''insert into entery(name,image,time)
                values('{}',load_file('{}'),'{}')'''.format(im_nam,'F/cvv/face_rec/{}'.format(im_nam),sdt)
            cur2.execute(stm)
            con.commit()
        
        

    cv2.imshow('Webcam_facerecognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
