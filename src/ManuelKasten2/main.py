'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-07-11 21:01:41
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-01 13:55:58
Description: 
'''
import cv2
import time
import torch

W = 1025 # magic, may have to be twice as much as kmap is remainder
H = 1024 # magic, may have to be twice as much as kmap is remainder
FPS = 24

rand_state = torch.rand([3, W*H], dtype = torch.float64, device = 'cuda')

@torch.jit.script
def cacu_once(rand_state:torch.Tensor, n:int = 1, W:int = 1024, H:int = 1024):
    """
    Puts `n` random floats into the head of `rand_state` as a queue model,
    and calculates a new `k_map`.

    - `rgb` is converted to `torch.int32`.
    - Elements greater than 255 in `rgb` are replaced with `511 - element`.
    
    the C++ implementation of this function:
    ```C++
    namespace ManuelKasten2
    {
        unsigned char RD(int i, int j) {
            static double k; k += (double)rand() / RAND_MAX; int l = k; l %= 512; return l > 255 ? 511 - l : l;
        }

        unsigned char GR(int i, int j) {
            static double k; k += (double)rand() / RAND_MAX; int l = k; l %= 512; return l > 255 ? 511 - l : l;
        }

        unsigned char BL(int i, int j) {
            static double k; k += (double)rand() / RAND_MAX; int l = k; l %= 512; return l > 255 ? 511 - l : l;
        }
    }
    ```
    """
    rand_state = torch.roll(rand_state, shifts=n, dims = -1)
    rand_state[:, :n] = torch.rand([3, n], dtype = torch.float64, device = 'cuda')    
    k_map = torch.cumsum(rand_state, dim = -1).to(torch.int32) % 512
    
    rgb = torch.where(k_map > 255, 511 - k_map, k_map)
    return rgb.reshape(3, H, W).permute(1, 2, 0).to(device = 'cpu', dtype = torch.uint8), rand_state

startTime = time.time()
FPSTime = time.time()
while cv2.waitKey(1) != ord('q') :
    print(f"{1 / (time.time() - FPSTime):.1f} fps")
    FPSTime = time.time()
    
    CPUFrame, rand_state = cacu_once(rand_state, 2500, W, H)

    cv2.imshow('animation', CPUFrame.numpy())

cv2.destroyAllWindows()