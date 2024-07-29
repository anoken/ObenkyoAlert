# ObenkyoAlert


![top_obenkyoalert](https://github.com/user-attachments/assets/f59cd7f3-2098-4e93-8c32-f562bc59727e)
### ストーリー
　勉強中に「一人だとどうしてもサボっちゃう…」と感じたことはありませんか？そんなあなたにピッタリなのが「お勉強アラート」！<br>
この小さなボックス型ロボットが、あなたの手元をじっと見守り、ペンを置いた瞬間に優しく叱ってくれます。一人でお勉強するときの心強い味方として、あなたの学びをサポートします。<br>

### 実装
![アルゴリズム説明_3](https://github.com/user-attachments/assets/8bf923be-f755-425a-9f14-3a5939014e43)
　USBカメラからの画像をオブジェクト認識のNanoDetを使い、手の領域を抽出します。NanoDetを呼び出すために、ディープラーニングフレームワークのncnnを使います。<br>
 
 ![アルゴリズム説明_4](https://github.com/user-attachments/assets/e4216ffd-51ee-4d01-a54d-072a6d7bfb6e)
　手の領域の画像から、クラス分類の推論を行います。 クラス分類での特徴ベクトルを求めるためにCNNのMobileNetV1を使用します。また、特徴ベクトルからクラスを求める処理は、SEFRアルゴリズムを使用します。SEFRアルゴリズムはデバイスのリソースに制約がある場合にも使うことのできる分類器アルゴリズムです。<br>

![アルゴリズム説明_5](https://github.com/user-attachments/assets/f770f23c-043b-473a-b04a-796a27aafa33)
 　お勉強アラートでは、 Studyingクラス、 Not Studyingクラス、Smart Phoneクラス、の クラス分類を用意しました。事前に、各クラスの画像を20枚程度ずつ集め、CoreMP135内部のオフライン処理で学習を行います。


### 謝辞
- 制作にあたり、[@ciniml](https://protopedia.net/prototyper/ciniml)氏の[CoreMP135ｽﾀｯｸﾁｬﾝの3Dプリンタ部品](https://drive.google.com/file/d/1b-DyHLuQ_9KmNaHUxqd3j8Rp-5D0_t_P/view)を使用させて頂きました。<br>
-  制作にあたり、[@FeiGeChuanShu](https://github.com/FeiGeChuanShu/ncnn_nanodet_hand)氏の[hand detect model](https://github.com/FeiGeChuanShu/ncnn_nanodet_hand)を使用させて頂きました。

### 参考
M5Stack CoreMP135: https://docs.m5stack.com/en/core/M5CoreMP135<br>
ncnn: https://github.com/Tencent/ncnn <br>
SEFR: A Fast Linear-Time Classifier for Ultra-Low Power Devices: https://arxiv.org/abs/2006.04620 <br>
