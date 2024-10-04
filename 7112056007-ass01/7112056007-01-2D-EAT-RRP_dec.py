import cv2
import numpy as np
import os
import copy
import math

# 定義2D-EAT函數
def two_dimensional_equilateral_arnold_inverse_transform(image, a, b):
    image_size = image.shape[0]
    image_change = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixcel_change = np.mod(np.dot(np.array([[a*b+1, -a], [-b, 1]]), np.array([[i], [j]])), image_size)
            image_change[pixcel_change[0][0], pixcel_change[1][0]] = image[i, j]
    return image_change

# 轉換成二進制
def Convert_to_binary(num):
    num_binary = bin(num)
    num_binary_array = [0, 0, 0, 0, 0, 0, 0, 0] 
    for i in range(len(num_binary)-1, 1, -1):
        num_binary_array[i + (8 - len(num_binary))] = int(num_binary[i])
    num_binary_array = np.array(num_binary_array)
    return num_binary_array

# 用mod函數取七個隨機數(已反轉)
def rand_num(seed, quantity):
    rand_array = []
    for i in range(quantity, 1, -1):
        rand_array.append(seed % i)
    rand_array = np.flipud(np.array(rand_array))
    return rand_array

# 二進制轉換成十進制
def binary_to_decimal(num_array):
    num = 0
    for i in range(len(num_array)):
        if(num_array[i] == 1):
            num += math.pow(2, 7-i)
    return num

# 定義彩色圖的Durstenfeld_Random_Permutation函數
def Durstenfeld_Reverse_Random_Permutation_True_Color(num, rand_array):
    num_array_blue = Convert_to_binary(num[0])
    num_array_green = Convert_to_binary(num[1])
    num_array_red = Convert_to_binary(num[2])
    num_array = np.concatenate((num_array_red, num_array_green, num_array_blue))
    for i in range(len(rand_array)):
        change_site = rand_array[i]
        change_num = num_array[change_site]
        num_array[change_site] = num_array[i+1]
        num_array[i+1] = change_num
    num_array_red_transform, num_array_green_transform, num_array_blue_transform = np.split(num_array, 3)
    num_array_red_transform = binary_to_decimal(num_array_red_transform)
    num_array_green_transform = binary_to_decimal(num_array_green_transform)
    num_array_blue_transform = binary_to_decimal(num_array_blue_transform)
    return np.array([num_array_blue_transform, num_array_green_transform, num_array_red_transform])

# 定義灰階圖的Durstenfeld_Random_Permutation函數
def Durstenfeld_Reverse_Random_Permutation_Gray(num, rand_array):
    num_array = Convert_to_binary(num)
    for i in range(len(rand_array)):
        change_site = rand_array[i]
        change_num = num_array[change_site]
        num_array[change_site] = num_array[i+1]
        num_array[i+1] = change_num
    num_transform =  binary_to_decimal(num_array)
    return num_transform

# 判斷圖片為彩色或灰階並做Durstenfeld_Reverse_Random_Permutation
def differentiate_image_color(image, seed):
    if len(image.shape) == 3:
        rand_array = rand_num(seed, 24)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = Durstenfeld_Reverse_Random_Permutation_True_Color(image[i][j], rand_array)
        image_transform = image
    else:
        rand_array = rand_num(seed, 8)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                image[i][j] = Durstenfeld_Reverse_Random_Permutation_Gray(image[i][j], rand_array)
        image_transform = image
    return image_transform

#圖片資料夾的路徑
folder_path = "encryp"

#資料夾中所有圖片的名字
image_files = os.listdir(folder_path)

for image_file in image_files:
    # 構建完整的文件路徑
    image_path = os.path.join(folder_path, image_file) 

    # 使用OpenCV讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  

    # 取隨機亂數用
    seed = 100

    # 執行Durstenfeld_Reverse_Random_Permutation
    image = differentiate_image_color(image, seed)

    # 2D-EAT反轉換的次數
    G = 10

    for i in range(G):
        # 執行2D-EAT反轉換
        encrypted_image = two_dimensional_equilateral_arnold_inverse_transform(image, 1, 1)
        image = encrypted_image

    # 轉換後圖片檔名
    image_new_name = image_file[:-8] + '_dec' + image_file[-4:]

    # 保存圖像
    cv2.imwrite('decryp/%s'%image_new_name, image)

    #顯示圖片
    cv2.imshow('%s'%image_new_name, image)
    cv2.waitKey(0)  # 等待用户按键
    cv2.destroyAllWindows()