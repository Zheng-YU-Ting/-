import os
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from PIL import Image

n=1

datasets = ['trainFER-facecut-224']#資料夾
#datasets = ['trainFER_data-augmentation-Keras', 'testFER_data-augmentation-Keras']

class_names = ["angry","fear","happy","neutral","sad","surprise"]
#class_names = ["surprise"]

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

# Iterate through training and test sets
for dataset in datasets:
    
    print("Loading {}".format(dataset))
    
    # Iterate through each folder corresponding to a category
    for folder in os.listdir(dataset):
        #print(folder)
        label = class_names_label[folder]
        
        # Iterate through each image in our folder
        for file in tqdm(os.listdir(os.path.join(dataset, folder))):
            # Get the path name of the image
            img_path_up = os.path.join(os.path.join(dataset, folder))
            img_path = os.path.join(os.path.join(dataset, folder), file)
            #print(img_path)
            # Open and resize the img
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            #print(image.shape)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 透過轉換函式轉為灰階影像
            #cv讀照片，顏色莫認為BGR，需轉為RGB
            '''
            test_image = Image.open(img_path)
            plt.imshow(test_image)
            plt.show()
            '''
            color = (255, 0, 0)  # 定義框的顏色

            # OpenCV 人臉識別分類器
            face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            
            # 調用偵測識別人臉函式
            faceRects = face_classifier.detectMultiScale(
                image, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            
            #print(faceRects)

            # 大於 0 則檢測到人臉
            if len(faceRects):
    
                # 框出每一張人臉
                for faceRect in faceRects: 
                    x, y, w, h = faceRect
                    cv2.rectangle(image, (x, y), (x + h, y + w), color, 2)
                    crop_img = image[y:y+w, x:x+h]
                    
                    cv2.imwrite(img_path_up+"/"+file, crop_img)
                    
                    size = crop_img.shape
                    if (size[0] < 100):
                        os.remove(img_path)
                    n+=1
                    break
            
                # 將結果圖片輸出
                
                #cv2.imwrite(file+str(n)+'.jpg', crop_img)
            else:
                os.remove(img_path)
            #print("remove")
            
            
            '''
            test_image = Image.open(img_path)
            plt.imshow(test_image)
            plt.show()
            '''

                

