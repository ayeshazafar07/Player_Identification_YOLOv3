import cv2
import numpy as np

#WEB CAM INPUT
def detect_WebCam_video():

    #CUSTOM CLASSES FOR THE DETECTION 
    classes = ['Marcelo','Sergio_Ramos','Luka_Modric','Cristiano_Ronaldo']
    
    #PATH TO THE CUSTOM WEIGHTS THAT WE OBTAINED FROM TRAINING AND CONFIG FILES
    net = cv2.dnn.readNetFromDarknet("yolov3_files/yolov3_custom.cfg",r"yolov3_files/yolov3_custom_8000.weights")

    #CAPTURING THE VIDEO TO PERFORM DETECTION ON IT
    cap = cv2.VideoCapture(0)
    #LOOP RUNS UNTIL CAMERA CLOSED BY USER
    while 1:
        #DIVIDING THE VIDEO INTO FRAMES TO PERFORM DETECTION ON EACH ONE AND DISPLAY RESULTS SIMULTANEOUSLY 
        #ON SCREEN WITH FOUND PROBABILITY AROUND THE BOUNDING BOX
        _, img = cap.read()
        img = cv2.resize(img,(1280,720))
        hight,width,_ = img.shape
        blob = cv2.dnn.blobFromImage(img, 1/255,(416,416),(0,0,0),swapRB = True,crop= False)
        net.setInput(blob)
        output_layers_name = net.getUnconnectedOutLayersNames()
        layerOutputs = net.forward(output_layers_name)

        #COMPARING DONE NOW MAKING THE BOUNDING BOX AROUND DETECTION PERSON
        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3]* hight)
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,.5,.4)

        boxes =[]
        confidences = []
        class_ids = []

        for output in layerOutputs:
            for detection in output:
                score = detection[5:]
                class_id = np.argmax(score)
                confidence = score[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * hight)
                    w = int(detection[2] * width)
                    h = int(detection[3]* hight)

                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    boxes.append([x,y,w,h])
                    confidences.append((float(confidence)))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes,confidences,.8,.4)
        font = cv2.FONT_HERSHEY_PLAIN

        #SETTING UP THE COLOR OF THE BOUNDING BOX TO MAKE IT DIFFERENT FROM OTHER CLASS MEMBERS
        #ATTACHING THE CORRECT LABEL- THE ONE WITH HIGHEST PROBABILITY
        #THEN FINALLY DISPLAYING IT ON THE SCREEN
        colors = np.random.uniform(0,255,size =(len(boxes),3))
        if  len(indexes)>0:
            for i in indexes.flatten():
                x,y,w,h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                color = colors[i]
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,label + " " + confidence, (x,y+400),font,2,color,2)

        cv2.imshow('img',img)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

    