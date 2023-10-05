# TechnoproFoodPackageImageClassification
Signate主催のパッケージの画像分類(一般向け)のコード
# コンペティションの概要
パッケージ画像を入力として、そのパッケージ画像が飲料か食料かを分類する
# 試した手法
転移学習ですべての層の重みを再学習した(timmを使用)  
水増しなどをは実施していない  
またCLIPやgoogle の TeachableMachineでも試した  
# codes
local -> local環境で実行  
colab -> Google Colabで実行(無料枠のGPUで実行可能)  