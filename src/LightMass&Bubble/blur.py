'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-07-11 21:01:41
LastEditors: BHM-Bob
LastEditTime: 2022-11-26 12:47:14
Description: 
'''
import time

import cv2
import torch

W = 1600
H = 900
FPS = 24
sumDots = torch.randint(10,20, size = [1]).item()
print("sumDots:",sumDots)

moveXLen = torch.randint(-5,5,size = [sumDots], device = 'cuda').reshape(sumDots,1,1)
moveYLen = torch.randint(-5,5,size = [sumDots], device = 'cuda').reshape(sumDots,1,1)
moveXLen = moveXLen + 5 * moveXLen.eq(torch.zeros(size = [sumDots,1,1], device = 'cuda'))
moveYLen = moveYLen + 5 * moveXLen.eq(torch.zeros(size = [sumDots,1,1], device = 'cuda'))
dTimeStep = 0.1 * torch.rand(sumDots,dtype = torch.float32, device = 'cuda')
nowTimeStep = torch.rand(sumDots, device = 'cuda')
nextTimeStep = torch.randint(-100,100,size = [sumDots], device = 'cuda')

RGBScale = torch.tensor([255.0],dtype = torch.float32, device = 'cuda')

dotsX = torch.randint(low = 1, high = W,size = [sumDots], device = 'cuda').reshape(sumDots,1,1)
dotsY = torch.randint(low = 1, high = H,size = [sumDots], device = 'cuda').reshape(sumDots,1,1)
dotsCol = torch.rand(3,sumDots,dtype = torch.float32, device = 'cuda').reshape(3,sumDots,1,1)
#[sumDots,1,1] [sumDots,1,1] [3,sumDots,1,1]
print(dotsX.shape,dotsY.shape,dotsCol.shape)

xAxis, yAxis = torch.arange(0,W, device = 'cuda'), torch.arange(0,H, device = 'cuda')
mapX, mapY = torch.meshgrid(xAxis, yAxis, indexing='xy')
mapX, mapY = mapX.reshape([1,H,W]).repeat(sumDots, 1, 1), mapY.reshape([1,H,W]).repeat(sumDots, 1, 1)
frame = torch.zeros(size = [H,W,3], device = 'cuda',dtype = torch.float32)
#(num, H, W) (num, H, W) (H, W, 3)
print(mapX.shape,mapY.shape,frame.shape)

@torch.jit.script
def UpdateDots1Coordinate(dotsX:torch.Tensor, moveXLen:torch.Tensor,
                          maxW:int, sumDots:int):
    dotsX += moveXLen

    minEq = torch.zeros(size = [sumDots,1,1], device = 'cuda',dtype = torch.float32)
    maxEq = torch.ones(size = [sumDots,1,1], device = 'cuda',dtype = torch.float32)

    #WARING: to avoid x is 0, not simply using cp.sign(), this can make problems
    #less than 0 is 1, else is 0
    mask1 = torch.less(dotsX,minEq)
    #bigger than maxW is 1, else is 0
    mask2 = torch.greater(dotsX,maxW*maxEq)
    #err is -1, else is 1
    mask = 1 - 2 * (mask1+mask2)
    #reMark the moveXLen
    moveXLen *= mask

    return dotsX, moveXLen

@torch.jit.script
def UpdateDotsCol(dotsCol:torch.Tensor, nowTimeStep:torch.Tensor,
                  nextTimeStep, dTimeStep:torch.Tensor, sumDots:int):
    #update nowTimeStep
    nowTimeStep += dTimeStep
    #update color:~[0,1]
    dotsCol[0,:,:,:] = (0.5 * torch.sin(nowTimeStep) + 0.5).reshape(sumDots,1,1)
    dotsCol[1,:,:,:] = (0.5 * torch.cos(nowTimeStep) + 0.5).reshape(sumDots,1,1)
    dotsCol[2,:,:,:] = (0.5 * torch.sin(0.8 * nowTimeStep) + 0.5).reshape(sumDots,1,1)

    return dotsCol

@torch.jit.script
def mean_filter(img:torch.Tensor, kernelSize:tuple[int, int] = (3, 3)):
    """    do img mean filter
    img : torch.Tensor with shape [C, H, W]
    kernal_size : tuple(int, int)->(w, h) or int
    """
    pad = (kernelSize[0] // 2, kernelSize[1] // 2)
    # [N, C, H, W]
    tmp = img.unsqueeze(0)
    N, C, H, W = tmp.shape
    # [N, C*patch_size, patch_num]
    tmp = torch.nn.functional.unfold(tmp, kernelSize, 1, pad, stride = 1)
    # [N, C, patch_size, patch_num]
    tmp = tmp.reshape(N, C, kernelSize[0] * kernelSize[1], H*W)
    # [C, H, W]
    return tmp.mean(dim = 2).reshape(C, H, W)

@torch.jit.script
def CacuOnce(dotsX:torch.Tensor, dotsY:torch.Tensor, dotsCol:torch.Tensor,
             mapX:torch.Tensor, mapY:torch.Tensor, frame:torch.Tensor,
             RGBScale:torch.Tensor):
    #cacu 1 / (dx**2 + dy**2)
    dx = mapX.subtract(dotsX)
    dy = mapY.subtract(dotsY)
    #if there has 0
    zeroT = torch.zeros([1], dtype = torch.int64, device = 'cuda')
    dx.add_(dx.eq(zeroT).to(dtype = torch.int64))
    dy.add_(dy.eq(zeroT).to(dtype = torch.int64))
    lengths = dx.pow(2).add(dy.pow(2))
    lengths = torch.sin(0.00001 * lengths) / lengths
    #lengths:[num, H, W], sumLengths:[1, H, W]
    sumLengths = lengths.sum(dim = 0, keepdim=True)
    #cacu ratio : [num, H, W]
    ratio = lengths / sumLengths
    #frame[:,:,0]:[H,W,]  ratio:[num, H, W]  dotsCol[0,:,:,:]:[num,1,1]
    #frame[H, W, C]
    frame[:,:,0] = ratio.mul(dotsCol[0,:,:,:]).sum(dim = 0)
    frame[:,:,1] = ratio.mul(dotsCol[1,:,:,:]).sum(dim = 0)
    frame[:,:,2] = ratio.mul(dotsCol[2,:,:,:]).sum(dim = 0)
    
    #frame[H, W, C] =*255.> [C, H, W]
    img = torch.multiply(frame,RGBScale).permute(2, 0, 1)
    #frame[C, H, W] =blur> [C, H, W] => [H, W, C]
    return img.permute(1, 2, 0).to(device = 'cpu', dtype = torch.uint8)

    # blur way 2
    # img = mean_filter(torch.multiply(frame,RGBScale).permute(2, 0, 1), (9, 9))
    # return img.permute(1, 2, 0).to(device = 'cpu', dtype = torch.uint8).numpy()

startTime = time.time()
FPSTime = time.time()
while cv2.waitKey(1) != ord('q') :
    print(f"{1 / (time.time() - FPSTime):.1f} fps")
    FPSTime = time.time()

    CPUFrame = cv2.blur(CacuOnce(dotsX, dotsY, dotsCol, mapX, mapY, frame, RGBScale).numpy(), (9, 9))
    dotsX, moveXLen = UpdateDots1Coordinate(dotsX, moveXLen, W, sumDots)
    dotsY, moveYLen = UpdateDots1Coordinate(dotsY, moveYLen, H, sumDots)
    dotsCol = UpdateDotsCol(dotsCol, nowTimeStep, nextTimeStep, dTimeStep, sumDots)

    cv2.imshow('animation', CPUFrame)

cv2.destroyAllWindows()