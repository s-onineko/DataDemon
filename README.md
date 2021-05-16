# DataDemon
### https://share.streamlit.io/s-onineko/datademon/main/app.py
#
![スライド1](https://user-images.githubusercontent.com/70475483/118382393-dc429000-b62f-11eb-94e6-bb84e5cb15af.JPG)
#### このWebアプリではAWSが提供する機械学習フレームワークAutoGluonを利用してテストすることができます。
#### 必要なものは「機械学習モデル構築用のデータ（教師データ）」と「予測・分類を実施するデータ（テストデータ）」の2つのみです。
#### 複数の機械学習モデルの中から最適なモデルを自動で選び、テストデータに対して予測・分類した結果を出力します。
#
![スライド2](https://user-images.githubusercontent.com/70475483/118382397-e2387100-b62f-11eb-8262-baa58da660e1.JPG)
![スライド3](https://user-images.githubusercontent.com/70475483/118382399-e5336180-b62f-11eb-96a3-1e33d5ebb393.JPG)
![スライド4](https://user-images.githubusercontent.com/70475483/118382400-e795bb80-b62f-11eb-8910-53f260828566.JPG)
#
#### 実行される内容
##### 1. データの前処理(各カラムを数値・カテゴリ・テキストに分類)
##### 2. ラベルカラムから推論タスクを決定(分類・回帰)
##### 3. データの分離(eg.disjoint training/validation sets, k-fold split)
##### 4. 各モデルを個々に学習(Random Forest, KNN, LightGBM, NNetc.)
##### 5. 最適化されたアンサンブルを作成
##### 6. 作成したモデルをテストデータに適用、予測結果を出力
#
#### 【注意】本アプリを使用したことによって、利用者に不利益や損害が発生する場合においても、アプリ制作者はその責任を一切負わないものとします。各自の責任で使用してください。
