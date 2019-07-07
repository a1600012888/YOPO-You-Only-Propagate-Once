# YOPO (You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle)
Code for our [paper](https://arxiv.org/abs/1905.00877): "You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle" by [Dinghuai Zhang](https://scholar.google.com/citations?view_op=list_works&hl=zh-CN&user=8W-MW9sAAAAJ), [Tianyuan Zhang](http://tianyuanzhang.com), [Yiping Lu](https://web.stanford.edu/~yplu/), [Zhanxing Zhu](https://sites.google.com/view/zhanxingzhu/), [Bin Dong](http://bicmr.pku.edu.cn/~dongbin/).

![The Pipeline of YOPO](/images/pipeline.jpg)


## Prerequisites
* Pytorch==1.0.1, torchvision
* Python 3.5
* tensorboardX
* easydict
* tqdm

## Intall
```bash
git clone https://github.com/a1600012888/YOPO-You-Only-Propagate-Once.git
cd YOPO-You-Only-Propagate-Once
pip3 install -r requirements.txt --user
```

## How to run our code

### Natural training and PGD training 
* normal training: `experiments/CIFAR10/wide34.natural`
* PGD adversarial training: `experiments/CIFAR10/wide34.pgd10`
run `python train.py -d <whcih_gpu>`

You can change all the hyper-parameters in `config.py`. And change network in `network.py`
Actually code in above mentioned director is very **flexible** and can be easiliy modified. It can be used as a **template**. 

### YOPO training
Go to directory `experiments/CIFAR10/wide34.yopo-5-3`
run `python train.py -d <whcih_gpu>`

You can change all the hyper-parameters in `config.py`. And change network in `network.py`
Runing this code for the first time will dowload the dataset in `./experiments/CIFAR10/data/`, you can modify the path in `dataset.py`

<!--
## Experiment results

<center class="half">
    <img src="https://s2.ax1x.com/2019/05/16/EbamrT.jpg" width="300"/><img src="https://s2.ax1x.com/2019/05/16/EbatsK.jpg" width="300"/>
</center>
-->

## Miscellaneous
The mainbody of `experiments/CIFAR10-TRADES/baseline.res-pre18.TRADES.10step` is written according to 
[TRADES official repo](https://github.com/yaodongyu/TRADES).

A tensorflow implementation provided by [Runtian Zhai](http://www.runtianz.cn/) is provided
 [here](https://colab.research.google.com/drive/1hglbkT4Tzf8BOkvX185jFmAND9M67zoZ#scrollTo=OMyffsWl1b4y).
The implemetation of the ["For Free"](https://arxiv.org/abs/1904.12843) paper is also included. It turns out that our 
YOPO is faster than "For Free" (detailed results will come soon). 
Thanks for Runtian's help!

## Cite
```
@article{zhang2019you,
  title={You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle},
  author={Zhang, Dinghuai and Zhang, Tianyuan and Lu, Yiping and Zhu, Zhanxing and Dong, Bin},
  journal={arXiv preprint arXiv:1905.00877},
  year={2019}
}
```
