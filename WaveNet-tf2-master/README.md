# WaveNet Tensorflow v2

[WaveNet](https://arxiv.org/abs/1609.03499) with TensorFlow 2.0

## Train

### Library

```
tensorflow
numpy
librosa
scipy
```


```bash=
   ls wavs/*.wav | tail -n+10 > train_files.txt
   ls wavs/*.wav | head -n10 > test_files.txt
   python preprocess.py
   python train.py
```

## Eval

```bash=
   python synthesize.py [input_path] [output_path] [weight_path]
```

## Result

The 1-day training model synsthesis result is below:
<img width="782" alt="0001" src="https://user-images.githubusercontent.com/33972190/76222682-45758900-625e-11ea-843a-c9957e44ee57.png">

pre trained model link is still a little way off.

## Book

![4a948f86-b96f-42ea-927f-14232d57589c_base_resized](https://user-images.githubusercontent.com/33972190/76142918-7feff200-60b5-11ea-9569-0423f8bb3fe9.jpg)

https://techbookfest.org/product/5743005264773120  
https://otakuassembly.booth.pm/items/1834753

## References

[WaveNet](https://arxiv.org/abs/1609.03499)

論文。

[r9y9/wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder)  
[LESS IS MORE/WaveNet vocoder をやってみましたので、その記録です](https://r9y9.github.io/blog/2018/01/28/wavenet_vocoder/)  

いくつもの論文で使われている実装。PyTorch。

[Rayhane-mamah/Tacotron-2](https://github.com/Rayhane-mamah/Tacotron-2)

Tactron2 + WaveNetのDeepMindの人の実装。TensorFlow v1。

[Monthly Hacker's Blog/VQ-VAEの追試で得たWaveNetのノウハウをまとめてみた。](https://www.monthly-hack.com/entry/2018/02/23/203208)

WaveNetに関する知見が纏められている。

[Synthesize Human Speech with WaveNet](https://chainer-colab-notebook.readthedocs.io/ja/latest/notebook/official_example/wavenet.html)

Colabを用いた解説。Chainer。

[The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

英語の単一話者のデータセット。

[JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)

日本語の単一話者のデータセット。
