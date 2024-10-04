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
def find_period(array_original, a, b, c, d):
    # 紀錄轉換次數
    period = 0

    # 進行RT轉換
    array_transform, period = rectangular_transform(array_original, a, b, c, d, period)

    # 轉換到變回原來的圖或超過上限
    while array_transform.tolist() != array_original.tolist():
        if period > math.floor(array_original.shape[0] * array_original.shape[1] / 2):
            return 0
        array_transform, period= rectangular_transform(array_transform, a, b, c, d, period)
    
    return period

# 記錄period到csv
def period_to_csv(M, N, a, b, c, d, period, csv):
    period_csv = pd.read_csv(csv)

    # 如果period比過去資料大就紀錄上去
    for i in range(len(period_csv)):
        if period_csv.at[i, 'M'] == M and period_csv.at[i, 'N'] == N:
            if np.isnan(period_csv.at[i, 'period']) or period_csv.at[i, 'period'] < period:
                period_csv.at[i, 'a'] = a
                period_csv.at[i, 'b'] = b
                period_csv.at[i, 'c'] = c
                period_csv.at[i, 'd'] = d
                period_csv.at[i, 'period'] = period
                period_csv = period_csv.astype(dtype = int, errors = 'ignore')
            break

    period_csv.to_csv(csv, index=False)

# 使用者輸入
M, N = map(int, input('Please input M N: ').split())
a, b, c, d = map(int, input('Please input a b c d: ').split())

# 產生測試週期的矩陣
i_original = np.array(list(range(M * N))).reshape(N, M)

# 檢查矩陣是否符合要求
if not(is_legal_matrix(M, N, a, b, c, d)):
    print('Invalid RT matrix!')
else:
    # 找轉換週期
    period = find_period(i_original, a, b, c, d)
    print('M = %d, N = %d, (a, b, c, d) = (%d, %d, %d, %d), Period = %d' %(M, N, a, b, c, d, period))

    # 記錄period到csv
    period_to_csv(M, N, a, b, c, d, period, '7112056007-ass04-result.csv')