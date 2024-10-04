import math
import numpy as np
import copy
import pandas as pd

# 檢查矩陣是否符合要求
def is_legal_matrix(M, N, a, b, c, d):
    p = math.gcd(M, N)
    l_1 = int(M/p)
    l_2 = int(N/p)
    return (b % l_1 == 0 or c % l_2 == 0) and math.gcd(a * d - b * c, p) == 1 and math.gcd(a, l_1) == 1 and math.gcd(d, l_2) == 1

# RT轉換
def rectangular_transform(image, i, j, k, l, period):
    A = np.array([[i, j], [k, l]])
    image_change = copy.deepcopy(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixcel_change = np.mod(np.dot(A, np.array([[j], [i]])), np.array([[image.shape[1]], [image.shape[0]]]))
            image_change[pixcel_change[1][0]][pixcel_change[0][0]] = image[i][j]
    period = period + 1
    return image_change, period

# 找轉換週期
def find_period(array_original, a, b, c, d, period):
    # 進行RT轉換
    array_transform, period= rectangular_transform(array_original, a, b, c, d, period)

    # 轉換到變回原來的圖或超過上限
    while array_transform.tolist() != array_original.tolist():
        if period > math.floor(M * N / 2):
            return 0
        array_transform, period= rectangular_transform(array_transform, a, b, c, d, period)
    
    return period

# 顯示在螢幕上
def print_to_screen(data):
    print("%4s %3s %3s %3s %3s %6s"%('NO.', 'a', 'b', 'c', 'd', 'period'))
    if len(data) != 0:
        for i in range(len(data)):
            print("%4d %3d %3d %3d %3d %6d"%(data.at[i, 'NO.'], data.at[i, 'a'], data.at[i, 'b'], data.at[i, 'c'], data.at[i, 'd'], data.at[i, 'period']))

# 使用者輸入
M, N = map(int, input('Input M N: ').split())
a1, a2 = map(int, input('Input range a1 a2: ').split())
b1, b2 = map(int, input('Input range b1 b2: ').split())
c1, c2 = map(int, input('Input range c1 c2: ').split())
d1, d2 = map(int, input('Input range d1 d2: ').split())

# 產生測試週期的矩陣
i_original = np.array(list(range(M * N))).reshape(N, M)

# 放period的dataframe
table = {
    "NO.": [],
    "a": [],
    "b": [],
    "c": [],
    "d": [],
    "period": []
}
period_table = pd.DataFrame(table, dtype=int)

for i in range(a1, a2+1):
    for j in range(b1, b2+1):
        for k in range(c1, c2+1):
            for l in range(d1, d2+1):
                
                # 檢查矩陣是否符合要求
                if not(is_legal_matrix(M, N, i, j, k, l)):
                    continue
                
                # 紀錄轉換次數
                period = 0

                # 找轉換週期
                period = find_period(i_original, i, j, k, l, period)

                # 紀錄到dataframe
                if period != 0:
                    period_table = pd.concat([period_table, pd.DataFrame({"NO.": [len(period_table)+1], "a": [i], "b": [j], "c": [k], "d": [l], "period": [period]})], ignore_index=True)

# 顯示在螢幕上
print_to_screen(period_table)

# 找不到符合的a b c d就印出訊息，否則輸出dataframe到csv
if len(period_table) == 0:
    print("No legal matrix within the range!")
else:
    period_table.to_csv("%d_%d_parameters.csv" %(M, N), index=False)
                

