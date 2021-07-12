from flask import Flask, render_template, Response
import cv2
import numpy as np


from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model



from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import imutils
import time
import cv2
import os






import os
import cvlib as cv

app=Flask(__name__)
camera = cv2.VideoCapture(0)
def detect_and_predict_mask(frame,faceNet,maskNet):
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))

    #this function performs the meansubraction(combat illumination),scaling
    #params image,scalefactor(scaling),size of the image,mean
    #swaping is done on the given option
    
    #giving the blob to neural network and getting the prediction
    
    faceNet.setInput(blob)
    detection=faceNet.forward()
    print(detection.shape)
    
    
    faces=[]
    locs=[]
    preds=[]
    for i in range (0,detection.shape[2]):
            confidence=detection[0,0,i,2]
            
            
            if confidence>0.5:
                box=detection[0,0,i,3:7]*np.array([w,h,w,h])
                (startx,starty,endx,endy)=box.astype("int")
                
                (startx,starty)=(max(0,startx),max(0,starty))
                (enndx,endy)=(min(w-1,endx),min(h-1,endy))
                
                
                face=frame[starty:endy,startx:endx]
                face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
                face=cv2.resize(face,(224,224))
                face=img_to_array(face)
                face=preprocess_input(face)
                
                
                faces.append(face)
                locs.append((startx,starty,endx,endy))
            #face coordinates passed to masknet model
            if len(faces)>0:
                faces=np.array(faces,dtype="float32")
                preds=maskNet.predict(faces,batch_size=32)
                
            return(locs,preds)
            #locs->coordinates of the rectangle
            #preds

            #face detector model
prototxtpath=r"C:\Users\Aravind\face_detector\deploy.prototxt"
weightsPath=r"C:\Users\Aravind\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet=cv2.dnn.readNet(prototxtpath,weightsPath)#this automatically detect the origin frame work and call the appropriate function

maskNet=load_model("facemask_detector.model")

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            vs = VideoStream(src=0).start()
            frame=vs.read()#every frame is read
            frame=imutils.resize(frame,width=400)#resizing the takenframe with width 400
            
            (locs,preds)=detect_and_predict_mask(frame,faceNet,maskNet)
            
            
            
            for(box,pred) in zip(locs,preds):#map the similar index of the locs and pred
                (startx,starty,endx,endy)=box
                (mask,withoutMask)=pred
                #condition to predict the mask or no mask and the color
                label="mask" if mask>withoutMask else"no_mask"
                color=(0,255,0) if label=="mask" else(0,0,255)
                
                
                label="{}:{:.2f}%".format(label,max(mask,withoutMask)*100)#probability percentage to say how much sure our model prediction
        
                cv2.putText(frame,label,(startx,starty-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)#method used to put a string on the image
                cv2.rectangle(frame,(startx,starty),(endx,endy),color,2)#to draw bounty box to model

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame = buffer.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__=='__main__':
    app.run(debug=True)
