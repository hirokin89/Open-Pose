from PIL import Image
import cv2
import sys
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import torch
from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

# Define Model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net = OpenPoseNet()

# 色情報の標準化
color_mean = [0.485, 0.456, 0.406]
color_std = [0.229, 0.224, 0.225]

# Read Weight data
net_weights = torch.load(
    './weights/pose_model_scratch.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}

for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]


state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)
net.eval()
net.to(device)
print('Finish Read information of model')

test_video = 'test.mp4'

cap = cv2.VideoCapture(test_video)


frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

n = 0
while cap.isOpened():
    ret,frame = cap.read()
    
    frame = cv2.resize(frame, dsize=(int(frame.shape[1] / 3), int(frame.shape[0] / 3)))
    #print(frame)
    
    tmp = frame.astype(np.float32) / 255.
    preprocessing_frame = tmp.copy()[:,:,::-1]

    for i in range(3):
        preprocessing_frame[:,:,i] = preprocessing_frame[:,:,i] - color_mean[i]
        preprocessing_frame[:,:,i] = preprocessing_frame[:,:,i] / color_std[i]
    

    img = preprocessing_frame.transpose((2, 0, 1)).astype(np.float32)

    img = torch.from_numpy(img)
    x = img.unsqueeze(0)
    x = x.to(device)

    predicted_outputs, _ = net(x)

    pafs = predicted_outputs[0][0].cpu().detach().numpy().transpose(1, 2, 0)
    heatmaps = predicted_outputs[1][0].cpu().detach().numpy().transpose(1, 2, 0)

    '''
    for part in range(19):
        heat_map = heatmaps[:, :, part] 
        heat_map = Image.fromarray(np.uint8(cm.jet(heat_map)*255))
        heat_map = np.asarray(heat_map.convert('RGB'))
    '''
    _, result_img, _, _ = decode_pose(frame, heatmaps, pafs)
    #cv2.imshow('test',result_img)

    cv2.imwrite('./result/' + str(n) + '.jpg', result_img)
    n = n + 1
