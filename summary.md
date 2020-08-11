# OpenCV

## RGB

赤，緑，青

|  Red  | Green | Blue  |
| :---: | :---: | :---: |
| 0-255 | 0-255 | 0-255 |

## HSV

色相，彩度，明度

|  Hue  | Saturation | Value  |
| :---: | :--------: | :----: |
| 0-359 |   0-100%   | 0-100% |

OpenCV では，HSV の範囲は上記とは異なり，以下の通りとなる

|  Hue  | Saturation | Value |
| :---: | :--------: | :---: |
| 0-179 |   0-255    | 0-255 |

## 座標軸の原点は左上

画像の原点は右下ではなく，左上となる．

## 画素の順番は\(B, G, R\)の順番

RGB ではなく，BGR となる．

## img.shape からは(Y, X, COLOR)

(X, Y) の順ではなく，(Y, X) となる

## 画像の読み込み・表示・書き込み

`cv2.imread('path/to/image')`で，画像の読み込みを行うことができる．  
`cv2.imread`では存在しないパスを指定しても，例外を送出せずに`None`を返すようになっている

`cv2.imshow('window_name', image_object)`で，新たなウィンドウを開き画像を表示する．

`cv2.imwrite('path/to/image', image_object)`で，指定したパスに画像を保存する．

```py
import cv2

# 読み込み
img = cv2.imread('image.jpg')

# ウィンドウを開いて画像を表示
cv2.imshow('window_name', img)
# キーが押されるまで待機
cv2.waitKey(0)
# 全ての開かれたウィンドウを閉じる
cv2.destroyAllWindows()

# 画像を保存
cv2.imwrite('output/test.jpg')
```

## 動画の読み込み・表示・書き込み

`fourcc`とは，４つの文字で構成されたコーデックを識別するための文字列．

`コーデック`とは，`Compression/DECompression`の略で，どういうプログラムを用いて`符号化(Encode)`，`復号(Decode)`を行うかを示す．  
動画自体が動画と音声とで容量が大きいため，符号化を行う必要がある．  
動画を再生するためには符号化したままだと再生できないため，復号をする必要がある．  
動画と音声の符号化と復号はそれぞれ扱う`コーデック`が異なる．

また，`コーデック`によって扱える`コンテナ(mp4やavi)`などの動画形式が変わる．  
そのため符号化と復号を行う際は，扱いたい`コンテナ`によって`動画コーデック`と`音声コーデック`の組み合わせを考えなくてはいけない．

`cv2.VideoCapture('path/to/video')`で，動画の読み込みを行うことができる．  
`VideoCapture #isOpened`で，動画の読み込みに成功しているかをチェックすることができる．

`VideoCapture #read`で，動画の次のフレームを取得することができる．このメソッドは 2 つの戻り値を戻し，フレームの読み込みに成功したかを示す真偽値と，画像を返す．

`cv2.VideoWriter_fourcc(*'XVID')`で，コーデックを取得する．

`cv2.VideoWriter('path/to/video', fourcc, 30.0, (w, h))`で，指定したパスに動画を保存するように設定する．  
引数は順に出力パス，コーデック，フレームレート，縦・横のタプル．

`VideoWriter #write`で，引数に与えた画像を動画ファイルに出力する．

```py
import cv2
import sys

# 動画の読み込み
cap = cv2.VideoCapture('video.mp4')
# 動画のオープンに成功しているか
if cap.isOpened() == False:
    sys.exit()

# 次のフレームを読み込み(一フレーム分を読み込む)
ret, frame = cap.read()
# 縦と横の長さを取得(shapeプロパティは[Y, X, COLOR])
h, w = frame.shape[:2]
# fourcc -> 4-character code of codec used to compress the frames
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# 今回は，XVID: MPEG-4コーデックで書き込み，AVI拡張子で出力する
dst = cv2.VideoWriter('output/test.avi', fourcc, 30.0, (w, h))
# 作成した動画ファイルのオープンに成功しているか
if dst.isOpened == False:
    sys.exit()

while True:
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imshow('frame', frame)
    # 一フレーム分を動画として書き込む
    dst.write(frame)
    # 30秒間，Escキーが押されるまで待機
    if cv2.waitKey(30) == 27:
        break

cv2.destroyAllWindows()
# 読み込んだ動画をクローズ
cap.release()
```

## ウィンドウの調整

```py
import cv2

img = cv2.imread('image.jpg')
# 画像サイズに合わせるようなウィンドウサイズにする
# リサイズ不可
cv2.namedWindow('auto_size', cv2.WINDOW_AUTOSIZE)
cv2.imshow('auto_size', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# リサイズ可能
cv2.namedWindow('normal_size', cv2.WINDOW_NORMAL)
cv2.imshow('normal_size', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 画像のリサイズ

`cv2.resize(img, size)`で，画像をリサイズすることができる．  
デフォルトでの補完方法(`interpolation`)は，`INTER_LINEAR`となっている．

`INTER_AREA`は画素面積を使ってリサンプリングしてリサイズを行う．モアレ(縞模様)が発生する．

```py
import cv2

img = cv2.imread('image.jpg')
size = (img.shape[1] // 2, img.shape[0] // 2)
img_resized = cv2.resize(img, size)
cv2.imshow('img_resized', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

img_area = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
cv2.imshow('img_area', img_area)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## HSV への変換・グレースケールへの変換

`cv2.cvtColor(img, CONSTANT)`で，変換処理を行うことができる．

```py
import cv2

img = cv2.imread('image.jpg')
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

## ヒストグラムの計算

```py
cv2.calcHist(
    [img],
    [channel],
    mask,
    [histogram_sizes_in_each_dimension],
    [boundary_in_each_dimension]
)
```

で，ヒストグラムの計算を行うことができる．

```py
import cv2

# BGR画像のヒストグラム
img = cv2.imread('image.jpg')
color_list = ['blue', 'green', 'red']
for i, j in enumerate(color_list):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=j)

# グレースケールのヒストグラム
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])
plt.plot(hist)
```

## ヒストグラムの均一化

`cv2.equalizeHist(img)`で，ヒストグラムの均一化を行う．  
ヒストグラムの均一化とは，一言でいうと明暗をよりはっきりにするための処理．

```py
import cv2

img = cv2.imread('image.jpg', 0)
img_eq = cv2.equalizeHist(img)

cv2.imshow('original_img', img)
cv2.imshow('equalized_img', img_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## γ 変換

γ(ガンマ)変換とは画像の明るさの変更方法を示す．
γ が 1 より小さいときは，暗くなる．
γ が 1 より大きいときは，明るくなる．

γ 変換の計算式は以下．  
`y = (x/255) ^ (1/γ)`

```py
import cv2
import numpy as np

gamma = 1.5
lookup_table = np.zeros((256, 1), dtype=np.uint8)
for i in range(256):
    lookup_table[i][0] = 255 * (float(i)/255) ** (1.0/gamma)

img = cv2.imread('image.jpg')
img_gamma = cv2.LUT(img, lookup_table)
cv2.imshow('img', img)
cv2.imshow('img_gamma', img_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## トラックバーの作成

下記のようにすることで，ウィンドウにトラックバーを作成できる．

```py
import cv2

def on_track_bar(position):
    global trackValue
    trackValue = position

trackValue = 100
cv2.namedWindow('window_name')
cv2.createTrackbar('track_bar', 'window_name', trackValue, 255, on_track_bar)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## マウスイベント

下記のようにすることで，マウスイベントを感知させることができる．

```py
import cv2
import numpy as np

def print_position(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDBCLK:
        print(x, y)

img = np.zeros((512, 512), dtype=np.uint8)
cv2.namedWindow('window_name')
cv2.setMouseCallback('window_name', print_position)
cv2.imshow('window_name', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 図形の描画

### 線の描画

`cv2.line(img, point1, point2, color, thickness)`

### 四角形の描画

`cv2.rectangle(img, point1, point2, color, thickness)`

`thickness`に`-1`などのネガティブな番号を入れると，図形を塗りつぶしてくれる．

### 円の描画

`cv2.circle(img, center, radius, color, thickness)`

### 楕円の描画

`cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)`

### 多角形の描画

`cv2.polylines(img, [points], isClosed, color, thickness)`

### 文字の描画

`cv2.putText(img, text, position, fontFace, fontScale, color, thickness, lineType)`
