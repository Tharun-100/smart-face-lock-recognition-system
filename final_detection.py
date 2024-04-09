import cv2
import torch
# from siamese_network import SiameseNetwork
from dataset_for_siamese import ImageNameDataset
import os 
import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils

from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from ultralytics import YOLO

import serial 
ser=serial.Serial('COM13',9600)



# Load the model
model = YOLO(r"C:\Users\Tharun\Desktop\NON_core\PROJECT\Face_recognition_demo\Detect_yolo_files\Model_results\best.pt")


cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    success, img = cap.read()

    # if success:
    # Perform inference on the frame
    results = model(img, verbose=False)
    res=(results[0].boxes.xyxy.tolist())
    filename = "C:\\Users\\Tharun\\Desktop\\NON_core\\PROJECT\\Face_recognition_demo\\folder\\detected_faces\\cropped_face.png"
    if res:
        x, y, w, h =res[0]
        x=int(x)
        y=int(y)
        w=int(w)
        h=int(h)

        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 2)
        # Crop the face from the original image
        face_crop = img[y:h, x:w]        
        # cv2.imshow('Cropped Face', face_crop)
        
        # Save the cropped face
        
        if face_crop is not None:
            cv2.imwrite(filename, face_crop)
    else:
        # cv2.imshow('Cropped Face', face_crop)
        print("No detection")
        cv2.imshow('Face Detection', img)

        ser.write(b'0')             # Sending '0' as a signal when no face detected
        print("Shhhhh")

        cv2.waitKey(2000)                           # TODO 1
        continue
    # cv2.imshow('Face Detection', img)
   

    #### FACE - RECOGNITION ###
        # Check if a GPU is available and set the device accordingly

    #create the Siamese Neural Network
    class SiameseNetwork(nn.Module):

        def __init__(self):
            super(SiameseNetwork, self).__init__()

            # Setting up the Sequential of CNN Layers
            self.Convolution_layers= nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=11,stride=4),#  JUST CHANGED SO THAT WE CAN USE THE MODEL FOR THE COLOUR IMAGES.
                nn.ReLU(inplace=True),

                nn.MaxPool2d(3, stride=2),

                nn.Conv2d(96, 256, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),

                nn.Conv2d(256, 384, kernel_size=3,stride=1),
                nn.ReLU(inplace=True)
            )

            # Setting up the Fully Connected Layers
            self.fully_connected_layers = nn.Sequential(
                nn.Linear(384, 1024),
                nn.ReLU(inplace=True),

                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),

                nn.Linear(256,2)
            )

        def forward_once(self, x):
            # This function will be called for both images
            # It's output is used to determine the similiarity
            output = self.Convolution_layers(x)
            flattened_output = output.view(output.size()[0], -1)#
            output = self.fully_connected_layers(flattened_output)
            return output

        def forward(self, input1, input2):
            # In this function we pass in both images and obtain both vectors
            # which are returned
            output1 = self.forward_once(input1)
            output2 = self.forward_once(input2)
            return output1, output2
    # Define the Contrastive Loss Function
    class ContrastiveLoss(torch.nn.Module):
        def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

        def forward(self, output1, output2, label):
        # Calculate the euclidian distance and calculate the contrastive loss
            euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                        (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


            return loss_contrastive
        
# Creating some helper functions
    def imshow(img, text=None):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_plot(iteration,loss):
        plt.plot(iteration,loss)
        plt.show()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the SiameseNetwork
    siamese_model = SiameseNetwork()

    # Load the pre-trained weights if available
    # Adjust the path accordingly based on where your model weights are saved
    # model_weights_path ="C:\\Users\\Tharun\\Desktop\\NON_core\\PROJECT\\Face_recognition_demo\\Siamese_network models\\siamese_face_cpu.pt"
    model_weights_path=r"C:\Users\Tharun\Desktop\NON_core\PROJECT\Face_recognition_demo\Siamese_network models\siamese_face_cpu.pt"
    siamese_model.load_state_dict(torch.load(model_weights_path, map_location=device))

    # Set the model to evaluation mode
    siamese_model.eval()

    # Move the model to the appropriate device (CPU or GPU)
    siamese_model.to(device)



# creating the Known dataset for the model:
# folder_dataset = datasets.ImageFolder(root="C:\\Users\\Tharun\\Desktop\\STUDY\\PROJECT\\Face_recognition_demo\\known_images")

    # Resize the images and transform to tensors
    if os.path.exists(filename):

        transformation = transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor()])
        root1="C:\\Users\\Tharun\\Desktop\\NON_core\\PROJECT\\Face_recognition_demo\\folder\\known_images"
        # Initialize the network
        known_dataset= ImageNameDataset(folder_path=root1,transform=transformation)

        # creating the detected datasets for the model:
        # unknown_folder=datasets.ImageFolder(root="C:\\Users\\Tharun\\Desktop\\STUDY\\PROJECT\\Face_recognition_demo\\detected_faces")
        root2="C:\\Users\\Tharun\\Desktop\\NON_core\\PROJECT\\Face_recognition_demo\\folder\\detected_faces"
        unknown_dataset=ImageNameDataset(folder_path=root2,transform=transformation)

        test_dataloader1 = DataLoader(known_dataset,  batch_size=1, shuffle=True)
        test_dataloader2=DataLoader(unknown_dataset,  batch_size=1, shuffle=True)


        # Grab one image that we are going to test
        dataiter1 = iter(test_dataloader1)
        dataiter2=iter(test_dataloader2)
        names=[]
        images=[]
        distance=[]# this is to store the values for all the predictors:
        ## NOW WE ARE ONLY WORKING ON THE ONE IMAGES: LET'S WHAT IF WE GET THE MULTIPLE IMAGES: LATER

        for data1 in test_dataloader1: # for each image in the database: we test the detected image
            x0,label=data1
            # we will find the dissimialritty
            names.append(label[0])
            images.append(x0)
            # dist=[]
            for data2 in test_dataloader2:
                x1,_=data2 # from the above we will work 
                # Concatenate the two images together
                concatenated = torch.cat((x0, x1), 0)
                output1, output2 = siamese_model(x0.to(device), x1.to(device))# here the model returns the output: of neural net:
                euclidean_distance = F.pairwise_distance(output1, output2)
                distance.append(euclidean_distance.item())
                # imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {euclidean_distance.item():.2f}')
            # if dist :
            #     distance.append(dist)
        ## Now find the minimum distance among all the distances:
        min_element = min(distance)

        # Find the index of the minimum element
        min_index = distance.index(min_element)

        known_image=images[min_index]
        # known_image=torch.tensor(known_image)
        # Now convert this known image as the tensor image as the 
        image_name=names[min_index]

        data_iter = iter(test_dataloader2)
        unknown_image, _ = next(data_iter)
        concatenated = torch.cat((known_image, unknown_image), 0)

        # response=0 # "Not Detected"
        # response2=1 #"looks suspicious!"
        dict={"aryan":2,"tharun":3,"aryabhatta":4}

        response = 0
        # response1="Welcome!"+image_name

        '''
        0: Nothing
        1: Alarm
        2: Open the door
        '''

        if(min_element < 0.5):
            response = dict[image_name]
            if(response == 2):
                ser.write(b'2')
            elif(response == 3):
                ser.write(b'3')
            elif(response == 4):
                ser.write(b'4')

            print("Known Person ", response)
            cv2.waitKey(3600)

            # imshow(torchvision.utils.make_grid(concatenated), f'Name:{image_name} Dissimilarity: {min_element:.2f}')
        else:
            ser.write(b'1')         # Sending alarm signal
            print("Access Denied")
            cv2.waitKey(600)
        
        # cv2.waitKey(5000)                   # TODO

        # if(min_element<1):
        #     # if image_name is not None and min_element<0.5:
        #     if min_element<0.5:

        #         response = dict[image_name]
        #         if(response == 2):
        #             ser.write(b'2')
        #         elif(response == 3):
        #             ser.write(b'3')
        #         elif(response == 4):
        #             ser.write(b'4')

        #         print("Known Person ", response)
        #         # imshow(torchvision.utils.make_grid(concatenated), f'Name:{image_name} Dissimilarity: {min_element:.2f}')

        #     elif min_element>0.5:
        #         ser.write(b'1')         # Sending alarm signal
        #         print("Access Denied")
        # else:
        #     ser.write(b'0')             # Sending '0' as a signal when no face detected
        #     print("Shhhhh")


        # if(min_element<2):
        #     imshow(torchvision.utils.make_grid(concatenated), f'Name:{image_name} Dissimilarity: {min_element:.2f}')
        # else:
        #     imshow(torchvision.utils.make_grid(unknown_image), f'Looks Suspicious!')
        
    if os.path.exists(filename):
        os.remove(filename) 
    if cv2.waitKey(1) & 0xFF == ord('b'):
            break   


    cv2.imshow('Cropped Face', face_crop)
    cv2.imshow('Face Detection', img)

cap.release()
cv2.destroyAllWindows()
