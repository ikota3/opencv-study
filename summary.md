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

(X, Y) の順ではなく，(Y，X) となる

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
