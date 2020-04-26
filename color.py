import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import math
import random
import os

class ImageColorizer:
    def __init__(self,numHiddenLayers,epochs,neuronsEachLayer,inputVal,outputVal,learningRate):
        self._numHiddenLayers = numHiddenLayers
        self._neuronsEachLayer = neuronsEachLayer
        self._epochs = epochs
        self._weights_red, self._weights_green, self._weights_blue, self._bias_red, self._bias_green, self._bias_blue = self.initialize_weights()
        self._x,self._y = inputVal,outputVal
        self._learningRate = learningRate
    
    def activationFunction(self, x):
        return np.tanh(x)

    def activationDerivative(self, x):
        return 1.0 - np.power(np.tanh(x), 2)

    def initialize_weights(self):
        weights_red = []
        weights_green = []
        weights_blue = []
        # print(weights)
        w = np.random.rand(1,self._neuronsEachLayer[0])
        weights_red.append(w)
        weights_green.append(w)
        weights_blue.append(w)
        for num in range(1,self._numHiddenLayers):
            w = np.random.rand(self._neuronsEachLayer[num-1],self._neuronsEachLayer[num])
            weights_red.append(w)
            weights_green.append(w)
            weights_blue.append(w)
        w = np.random.rand(self._neuronsEachLayer[self._numHiddenLayers-1],1)
        weights_red.append(w)
        weights_green.append(w)
        weights_blue.append(w)

        bias_red = []
        bias_green = []
        bias_blue = []
        for num in range(0, self._numHiddenLayers):
            b = np.zeros((1, self._neuronsEachLayer[num]))
            bias_red.append(b)
            bias_green.append(b)
            bias_blue.append(b)         
        b = np.zeros((1, 1))
        bias_red.append(b)
        bias_green.append(b)
        bias_blue.append(b)
        return weights_red, weights_green, weights_blue, bias_red, bias_green, bias_blue

    ## final layer activation function
    def sigmoid(self,x):
        y = 1/(1+math.exp(-x))
        return y
    
    def derivative_sigmoid(self,x):
        return (self.sigmoid(x)*(1-self.sigmoid(x)))

    # def compute_loss(self,loss):
    #     return loss - self._y[self._numHiddenLayers]

    def fit(self, images, rgb_imgs):
        
        hidden_red = []
        hidden_green = []
        hidden_blue = []
        for img in range(len(images)):
            print(img) 
            width, height = images[img].shape[0], images[img].shape[1]
            size = width * height
            ### forward pass : 
            count = 0
            for x in images[img].flatten():
                # print(x)
                for i in range(self._numHiddenLayers):
                    h1 = self.activationFunction(np.dot(np.transpose(self._weights_red[i]),x) + np.transpose(self._bias_red[i])) if i == 0 else self.activationFunction(np.dot(np.transpose(self._weights_red[i]), hidden_red[i-1]) + np.transpose(self._bias_red[i]))
                    hidden_red.append(h1)
                    h2 = self.activationFunction(np.dot(np.transpose(self._weights_green[i]),x) + np.transpose(self._bias_green[i])) if i == 0 else self.activationFunction(np.dot(np.transpose(self._weights_green[i]), hidden_green[i-1]) + np.transpose(self._bias_green[i]))
                    hidden_green.append(h2)
                    h3 = self.activationFunction(np.dot(np.transpose(self._weights_blue[i]),x) + np.transpose(self._bias_blue[i])) if i == 0 else self.activationFunction(np.dot(np.transpose(self._weights_blue[i]), hidden_blue[i-1]) + np.transpose(self._bias_blue[i]))
                    hidden_blue.append(h3)
                output1 = np.dot(self._weights_red[self._numHiddenLayers].T, hidden_red[self._numHiddenLayers-1]) + self._bias_red[self._numHiddenLayers].T
                output2 = np.dot(self._weights_green[self._numHiddenLayers].T, hidden_green[self._numHiddenLayers-1]) + self._bias_green[self._numHiddenLayers].T
                output3 = np.dot(self._weights_blue[self._numHiddenLayers].T, hidden_blue[self._numHiddenLayers-1]) + self._bias_blue[self._numHiddenLayers].T
                rgb_flat = np.zeros((1,3))
                rgb_flat[0][0] = rgb_imgs[img].flatten()[count]
                rgb_flat[0][1] = rgb_imgs[img].flatten()[count + size]
                rgb_flat[0][2] = rgb_imgs[img].flatten()[count + (2*size)]
                # print(rgb_flat)
                # loss = np.power(output - rgb_flat, 2)
                count += 1
                # print("Count:",count)
            #backward pass
                derivatives_w_red = []
                derivatives_z_red = []
                derivatives_b_red = []
                derivatives_w_green = []
                derivatives_z_green = []
                derivatives_b_green = []
                derivatives_w_blue = []
                derivatives_z_blue = []
                derivatives_b_blue = []
                # dw = np.zeros((1,self._neuronsEachLayer[-1]))
                # dz = np.zeros((1,1))
                # db = np.zeros((1, self._neuronsEachLayer[-1]))
                dz_red = output1 - rgb_flat[0][0].reshape(1,1)
                dw_red = np.dot(dz_red,hidden_red[self._numHiddenLayers-1].T)
                db_red = np.sum(dz_red,axis =1,keepdims=True)
                derivatives_w_red.append(dw_red) 
                derivatives_z_red.append(dz_red) 
                derivatives_b_red.append(db_red) 
                dz_green = output2 - rgb_flat[0][1].reshape(1,1)
                dw_green = np.dot(dz_green,hidden_green[self._numHiddenLayers-1].T)
                db_green = np.sum(dz_green,axis =1,keepdims=True)
                derivatives_w_green.append(dw_green) 
                derivatives_z_green.append(dz_green) 
                derivatives_b_green.append(db_green) 
                dz_blue = output3 - rgb_flat[0][2].reshape(1,1)
                dw_blue = np.dot(dz_blue,hidden_blue[self._numHiddenLayers-1].T)
                db_blue = np.sum(dz_blue,axis =1,keepdims=True)
                derivatives_w_blue.append(dw_blue) 
                derivatives_z_blue.append(dz_blue) 
                derivatives_b_blue.append(db_blue) 


                count1 = 0
                for i in range(self._numHiddenLayers-1,0,-1):
                    tempVar1 = np.dot(self._weights_red[i+1],derivatives_z_red[count1])*self.activationDerivative(hidden_red[i])
                    tempVar2 = np.dot(self._weights_green[i+1],derivatives_z_green[count1])*self.activationDerivative(hidden_red[i])
                    tempVar3 = np.dot(self._weights_blue[i+1],derivatives_z_blue[count1])*self.activationDerivative(hidden_red[i])
                    dz_red = tempVar1
                    dw_red = np.dot(derivatives_z_red[count1], hidden_red[i].T)
                    db_red = np.sum(derivatives_z_red[count1], axis = 1, keepdims = True)
                    derivatives_w_red.append(dw_red) 
                    derivatives_z_red.append(dz_red) 
                    derivatives_b_red.append(db_red)
                    dz_green = tempVar2
                    dw_green = np.dot(derivatives_z_green[count1], hidden_green[i].T)
                    db_green = np.sum(derivatives_z_green[count1], axis = 1, keepdims = True)
                    derivatives_w_green.append(dw_green) 
                    derivatives_z_green.append(dz_green) 
                    derivatives_b_green.append(db_green)
                    dz_blue = tempVar3
                    dw_blue = np.dot(derivatives_z_blue[count1], hidden_blue[i].T)
                    db_blue = np.sum(derivatives_z_blue[count1], axis = 1, keepdims = True)
                    derivatives_w_blue.append(dw_blue) 
                    derivatives_z_blue.append(dz_blue) 
                    derivatives_b_blue.append(db_blue)
                    count1+=1

                # dz[0] = np.dot(self._weights[0].T,dz[1])*self.activationDerivative(hidden[0])
                dw_red = np.dot(derivatives_z_red[0],x.T)
                db_red = np.sum(derivatives_z_red[0],axis =1,keepdims=True)
                derivatives_w_red.append(dw_red)
                derivatives_b_red.append(db_red)
                dw_green = np.dot(derivatives_z_green[0],x.T)
                db_green = np.sum(derivatives_z_green[0],axis =1,keepdims=True)
                derivatives_w_green.append(dw_green)
                derivatives_b_green.append(db_green)
                dw_blue = np.dot(derivatives_z_blue[0],x.T)
                db_blue = np.sum(derivatives_z_blue[0],axis =1,keepdims=True)
                derivatives_w_blue.append(dw_blue)
                derivatives_b_blue.append(db_blue)

                for i in range(self._numHiddenLayers + 1):
                    self._weights_red[i] -= derivatives_w_red[self._numHiddenLayers-i]
                    self._bias_red[i] -= derivatives_b_red[self._numHiddenLayers-i]
                    self._weights_green[i] -= derivatives_w_green[self._numHiddenLayers-i]
                    self._bias_green[i] -= derivatives_b_green[self._numHiddenLayers-i]
                    self._weights_blue[i] -= derivatives_w_blue[self._numHiddenLayers-i]
                    self._bias_blue[i] -= derivatives_b_blue[self._numHiddenLayers-i]
                # if(count%400==0):
                #     print(output , rgb_flat)
                #     input()


    def convert(self, image):
        rows, cols = image.shape[0], image.shape[1]
        print(rows, cols, len(image.flatten()))
        red = np.zeros((rows*cols, 1))
        green = np.zeros((rows*cols, 1))
        blue = np.zeros((rows*cols, 1))
        count = 0
        hidden_red = []
        hidden_green = []
        hidden_blue = []
        for x in image.flatten():
            for i in range(self._numHiddenLayers):
                h1 = self.activationFunction(np.dot(np.transpose(self._weights_red[i]),x) + np.transpose(self._bias_red[i])) if i == 0 else self.activationFunction(np.dot(np.transpose(self._weights_red[i]), hidden_red[i-1]) + np.transpose(self._bias_red[i]))
                hidden_red.append(h1)
                h2 = self.activationFunction(np.dot(np.transpose(self._weights_green[i]),x) + np.transpose(self._bias_green[i])) if i == 0 else self.activationFunction(np.dot(np.transpose(self._weights_green[i]), hidden_green[i-1]) + np.transpose(self._bias_green[i]))
                hidden_green.append(h2)
                h3 = self.activationFunction(np.dot(np.transpose(self._weights_blue[i]),x) + np.transpose(self._bias_blue[i])) if i == 0 else self.activationFunction(np.dot(np.transpose(self._weights_blue[i]), hidden_blue[i-1]) + np.transpose(self._bias_blue[i]))
                hidden_blue.append(h3)
            output1 = self.sigmoid(np.dot(self._weights_red[self._numHiddenLayers].T, hidden_red[self._numHiddenLayers-1]) + self._bias_red[self._numHiddenLayers].T)
            output2 = self.sigmoid(np.dot(self._weights_green[self._numHiddenLayers].T, hidden_green[self._numHiddenLayers-1]) + self._bias_green[self._numHiddenLayers].T)
            output3 = self.sigmoid(np.dot(self._weights_blue[self._numHiddenLayers].T, hidden_blue[self._numHiddenLayers-1]) + self._bias_blue[self._numHiddenLayers].T)
                



            red[count] = output1
            green[count] = output2
            blue[count] = output3

            count += 1

        red_channel = red.reshape(rows, cols)
        green_channel = green.reshape(rows, cols)
        blue_channel = blue.reshape(rows, cols)
        return red_channel, green_channel, blue_channel


neuronsEachLayer = [3,2,1]
numHiddenLayers = 3
numImages = 1
X = []
Y = []

def rgb_grayscale(img):
    
        r = img[:, :, 0]
        g = img[:, :, 1]
        b = img[:, :, 2]

        gray = 0.21*r + 0.72*g + 0.07*b

        return gray

# for i in range(numImages):
increment = 0
for data in os.listdir("./flower_images/flower_images/"):
    y = mpimg.imread("./flower_images/flower_images/"+data)
    x = rgb_grayscale(y)
    X.append(x)
    Y.append(y)
    if(increment>25):
        break
    increment+=1


# red_og = y[:,:,0]
im = rgb_grayscale(mpimg.imread('fig.png'))
imgs = ImageColorizer(numHiddenLayers,1,neuronsEachLayer,X,Y,0.01)
imgs.fit(X, Y)
red_c, green_c, blue_c = imgs.convert(im)
print(np.min(red_c), np.max(red_c), np.min(green_c), np.max(green_c), np.min(blue_c), np.max(blue_c))
