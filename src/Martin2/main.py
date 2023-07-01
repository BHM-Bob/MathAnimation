'''
Author: BHM-Bob G 2262029386@qq.com
Date: 2022-07-11 21:01:41
LastEditors: BHM-Bob 2262029386@qq.com
LastEditTime: 2023-07-01 14:45:19
Description: 
'''
import cv2
import time
import torch
import random

W = 512 # magic, may have to be twice as much as kmap is remainder
H = 512 # magic, may have to be twice as much as kmap is remainder
FPS = 24

rand_state = torch.zeros([H, W, 3], dtype = torch.int16, device = 'cuda')

@torch.jit.script
def cacu_once(rand_state: torch.Tensor, n: int = 1, W: int = 1024, H: int = 1024):
    """
    ```C++
    #define r(n)(rand()%n)
    unsigned char RD(int i, int j) {
        static char c[1024][1024];
        while (!c[i][j]) {
            c[i][j] = (!r(999)) ? r(256) : c[(i - r(2)) % 1024][(j - r(2)) % 1024];
            i = (i + r(2)) % 1024;
            j = (j + r(2)) % 1024;
        }
        return c[i][j];
    }
    // origin: return!c[i][j]?c[i][j]=!r(999)?r(256):RD((i+r(2))%1024,(j+r(2))%1024):c[i][j];
    ```
    """
    r_999_mask = torch.randint(9, [H, W, 3], dtype = torch.int16, device = 'cuda').eq(0)
    rgb = torch.where(r_999_mask,
                      torch.randint(256, [H, W, 3], dtype = torch.int16, device = 'cuda'),
                      torch.zeros([H, W, 3], dtype = torch.int16, device = 'cuda'))
    
    for i in range(H):
        for j in range(W):
            while not rgb[i, j, :].any():
                rgb[i, j, :] = rgb[i - torch.randint(0, 2, [1])[0] % H, j - torch.randint(0, 2, [1])[0] % W, :]
                i = (i + torch.randint(0, 2, [1])[0]) % H
                j = (j + torch.randint(0, 2, [1])[0]) % W
                
    return rgb.to(device = 'cpu', dtype = torch.uint8), rand_state

startTime = time.time()
FPSTime = time.time()
while cv2.waitKey(1) != ord('q') :
    print(f"{1 / (time.time() - FPSTime):.1f} fps")
    FPSTime = time.time()
    
    CPUFrame, rand_state = cacu_once(rand_state, 50, W, H)

    cv2.imshow('animation', CPUFrame.numpy())

cv2.destroyAllWindows()