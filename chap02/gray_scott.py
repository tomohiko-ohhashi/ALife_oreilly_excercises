# U: 食べ物｡V: 生き物｡V2つでU1つを消費して1つ増える｡P: Vが死んだやつ｡

import sys, os
sys.path.append(os.pardir)
import numpy as np
from alifebook_lib.visualizers import MatrixVisualizer

# visualizerの初期化(付録参照)
visualizer = MatrixVisualizer()

# シミュレーションを行う空間のサイズとパラメータ設定
SPACE_GRID_SIZE = 256
dx = 0.01 # 空間のグリッド1メモリあたりのモデル内の長さ
# dx * SPACE_GRID_SIZEがモデル空間の1ペンの長さ
# dxが大きいと大きな空間を粗くシミュレーションする
# dxが小さいと小さな空間を細かくシミュレーションする

dt = 1 # 1ステップごとのモデル内での時間の変化量
# dtを大きくすれば進みが大きくなるが時間に関して粗くなる
# dtが小さくなると正確な結果を得られるが時間が長くなる

VISUALIZATION_STEP = 8 # 何ステップごとに画面を更新するか

# 以下､Gray-Scottモデルのパラメータ設定

# DuとDvはuとvの拡散係数
Du = 2e-5
Dv = 1e-5

# feed kill を変えると様々に異なるふるまいが現れる
f, k = 0.022, 0.051 # stripe
# f, k = 0.035, 0.065  # spots
# f, k = 0.012, 0.05  # wandering bubbles
# f, k = 0.025, 0.05  # waves
# f, k = 0.05, 0.003 # 適当

# UとVの空間中の各点での濃度を表す変数u, v
# 初期化
u = np.ones((SPACE_GRID_SIZE, SPACE_GRID_SIZE)) # 要素すべて1の行列
v = np.zeros((SPACE_GRID_SIZE, SPACE_GRID_SIZE)) # 要素すべて0の行列


# 中央にSQUARE_SIZE四方の正方形を置く
SQUARE_SIZE = 20
u[SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2] = 0.5
v[SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2,
  SPACE_GRID_SIZE//2-SQUARE_SIZE//2:SPACE_GRID_SIZE//2+SQUARE_SIZE//2] = 0.25
# : はシーケンスのスライスを表す｡始点:終点｡要するに始点番目から終点番目までの要素を取得する｡

# : スライス インデックスというより､仕切りの間を指すらしい｡
li = ["あ", "い", "う", "え", "お"]
print(li[2:4]) # ['う', 'え']

# 初期状態をランダムにする｡対象性を壊すために､少しノイズを入れる
u += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE) * 0.1
v += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE) * 0.1



while visualizer: # ウィンドウが閉じられるとFalseを返す
    # visualizer.update(u) # 初期化パターンを表示できる
    
    for i in range(VISUALIZATION_STEP):
        # ラプラシアンの計算
        laplacian_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                    np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / (dx*dx)
        laplacian_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
                    np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / (dx*dx)
        
        # Gray-Scottモデル方程式
        dudt = Du*laplacian_u - u*v*v + f*(1.0-u)
        dvdt = Dv*laplacian_v + u*v*v - (f+k)*v
        u += dt * dudt
        v += dt * dvdt
    visualizer.update(u)