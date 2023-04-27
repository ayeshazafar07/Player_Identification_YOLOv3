import numpy as np
import cv2
import random 
import glob

def predictPlayer(image):
    #LOADING THE YOLO TRAINED WEIGHTS AND CONFIG FILE TO USE FOR DETECTION
    net = cv2.dnn.readNet("./yolov3_files/yolov3_custom_8000.weights", "./yolov3_files/yolov3_custom.cfg")

    #CUSTOM CLASSES FOR THE DETECTION 
    classes = ['Marcelo','Sergio_Ramos','Luka_Modric','Cristiano_Ronaldo']

    #IMAGE FOLDER PATH
    #COULDNT FIGURE THE RIGHT WAY TO INSERT PATH OF FOLDER AS INPUT 
    #SO HARD CODED THE PATH
    #IF YOU WANT TO CHANGE OR TEST SPECIFIC IMAGES- ADD THEM IN THE TEST_IMAGES FOLDER 
    images_path = glob.glob(r"./test_images/*.PNG")

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    #SETTING UP THE COLOR OF THE BOUNDING BOX TO MAKE IT DIFFERENT FROM OTHER CLASS MEMBERS
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #SHUFFLING THE IMAGES OF FOLDER
    #NO REAL PURPOSE, JUST TO MAKE RANDOM GUESSES
    #WILL WORK WITHOUT SHUFFLING THE FOLDER IMAGES TOO
    random.shuffle(images_path)
    
    #LOOPING THROUGH THE IMAGES IN FOLDER
    #COMPARING THEM
    #FINDING CLOSEST MATCH AND PROBABILITY OF IT
    #THEN CREATING A BOUNDING BOX AROUND THE DETECTED PERSON
    for img_path in images_path:
        #LOADING IMAGE ONE AT A TIME
        img = cv2.imread(img_path)
        img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        #PERFORMING DETECTION ON CONCERNED IMAGE
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        #COMPARING AND FINDING WHICH CLASS MATCHES THE BEST 
        #FINDING INDEXES OF BOUNDING BOX ACCORDING TO THE ONE WITH MAX SCORE THEN
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    #PERSON DETECTION DONE HERE
                    print(class_id)
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    #SETTING UP RECTANGLE COORDINATES
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        #PRINTING THE INDEXES WHERE PERSON IS DETECTED IN IMAGE 
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

        #FINALLY DISPLAYING THE IMAGE WITH DETECTED PERSON
        cv2.imshow("Image", img)
        #MOVE TO NECT PERSON WHEN q PRESSED
        key = cv2.waitKey(0)
        if cv2.waitKey(1) == ord('q'):
            #DESTROY CURRENT WINDOW AND MOVE TO NEXT IMAGE
            cv2.destroyAllWindows()

    cv2.destroyAllWindows()
    return img

