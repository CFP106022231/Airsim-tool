# Airsim-tool

[Airsim website](https://microsoft.github.io/AirSim/)

## Introduction

由微軟開發的虛擬環境，可以輸出相機姿態、深度、光流等 Ground Truth。現已不再維護更新此版本，不過仍可以下載運用。
有分三種模式：車車模式、無人機模式、計算機視覺模式。前兩個是有真實的車和無人機，計算機模式則是沒有質量的點，因此不會撞到東西（但好像同時也拿不到imu資訊）。
此project附上的檔案是使用計算機模式的。
![image](https://github.com/CFP106022231/Airsim-tool/assets/48315120/a6e9d2d6-9c9e-4fef-8603-531494e60243)

Airsim 以 unreal engine 的材質輸出data，可以串C++或python之類的把資料取出。
輸出的資料如下：
1.	Raw image：BGR圖像
2.	姿態：平移(x,y,z) 和以四元數表示的旋轉。
3.	光流：我們有修改，輸出光流的大小和角度。OF[...,0]為光流大小，需再自行乘以解析度（如128），單位才是pixel/frame。OF[...,1]為角度，單位弧度。
4.	深度

[hackmd交接note](https://hackmd.io/@hsinyuyuuu/Sk-xaaK2a)
