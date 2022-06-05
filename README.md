# StarGANv2-VC
このリポジトリは筆者が卒業研究で扱ったStarGANv2-VC[^1]および当手法で使用するJDCNet[^2]の**非公式**実装を含む就職活動用ポートフォリオである.
公開に際して[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)をデータセットとして学習および推論を行うデモの整備を行った. 

## 使用方法

### 1. HiFi-GAN[^3]の学習

[HiFi-GANの公式実装](https://github.com/jik876/hifi-gan)に従ってボコーダの作成を行う. 
このときのconfigは本リポジトリのHiFiGAN/config_v1.jsonを推奨する. 
これと異なる設定を行う場合もJVS corpusに合わせて`"sampling_rate": 24000`とする必要がある. 

### 2. CNNConformer_ASRの学習

`%path_jvs%`に[JVS corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)のルートディレクトリ(jvs_ver1)のパスを指定し, 以下を実行する. 

```
python CNNConformer_ASR/main.py --path_jvs %path_jvs%
```

### 3. JDCNet[^2]の学習

```
python JDCNet/main.py --path_jvs %path_jvs% 
```

### 4. StarGANv2-VC[^1]の学習

`%package_HiFiGAN%`に[HiFi-GAN](https://github.com/jik876/hifi-gan)のルートディレクトリ名(ex. hifi-gan-master)を指定する.
動的インポートを行うため, `%package_HiFiGAN%`はsys.pathに存在するパスに置かれている必要がある. なお本リポジトリのルートディレクトリはsys.pathに追加される.

`%checkpoint_HiFiGAN%`に1.で作成したGeneratorのcheckpoint(ex. g_01000000)のパスを指定する. 

```
python StarGANv2VC/main.py --path_jvs %path_jvs% --package_HiFiGAN %package_HiFiGAN% --checkpoint_HiFiGAN %checkpoint_HiFiGAN%
```

## 参照

- CNNConformer_ASR/common/conformer: https://github.com/sooftware/conformer
- CNNConformer_ASR/common/conformer/cnn.py
- JDCNet/common/model/jdcnet.py
- StarGANv2VC/common/model/models.py
- StarGANv2VC/train/tool/losses.py<br>
: https://github.com/yl4579/StarGANv2-VC

[^1]: StarGANv2-VC<br>
  paper: https://arxiv.org/abs/2107.10394#<br>
  official implementation: https://github.com/yl4579/StarGANv2-VC

[^2]: JDCNet<br>
  paper: https://www.mdpi.com/2076-3417/9/7/1324<br>
  official implementation: https://github.com/keums/melodyExtraction_JDC

[^3]: HiFi-GAN<br>
  paper: https://arxiv.org/abs/2010.05646<br>
  official implementation: https://github.com/jik876/hifi-gan
