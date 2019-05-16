# YOPO(You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle)
Code for our [paper](https://arxiv.org/abs/1905.00877): "You Only Propagate Once: Accelerating Adversarial Training via Maximal Principle" by Dinghuai Zhang, [Tianyuan Zhang](http://tianyuanzhang.com), [Yiping Lu](https://web.stanford.edu/~yplu/), Zhanxing Zhu, Bin Dong.

![The Pipeline of YOPO](/pipeline.jpg)


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


## Miscellaneous
A tensorflow implementation provided by [Runtian Zhai](http://www.runtianz.cn/) is provided
 [here](https://colab.research.google.com/drive/1hglbkT4Tzf8BOkvX185jFmAND9M67zoZ#scrollTo=OMyffsWl1b4y).
The implemetation of the ["For Free"](https://arxiv.org/abs/1904.12843) paper is also included. It turns out that our 
YOPO is faster than "For Free" (detailed results will come soon). 
Thanks for Runtian's help!
