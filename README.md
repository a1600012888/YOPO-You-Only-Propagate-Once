# YOPO(You Only Propagate Once: Painless Adversarial Training Using Maximal Principle)
Code for our paper: "You Only Propagate Once: Painless Adversarial Training Using Maximal Principle" by Dinghuai Zhang, [Tianyuan Zhang](http://tianyuanzhang.com), [Yiping Lu](https://web.stanford.edu/~yplu/), Zhanxing Zhu, Bin Dong.


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
normal training
Go to directory `./experiments/CIFAR10/wide34.natural`, run `python train.py -d <whcih_gpu>`

PGD adversarial training
Go to directory `./experiments/CIFAR10/wide34.pgd10`, run `python train.py -d <whcih_gpu>`

You can change all the hyper-parameters in config.py. And change network in `network.py`
Actually code in above mentioned director is very flexible and can be easiliy modified. It can be used as a template. 

### YOPO training
Go to directory `./experiments/CIFAR10/wide34.yopo-5-3`, run `python train.py -d <whcih_gpu>`
You can change all the hyper-parameters in config.py. And change network in `network.py`


Runing this code for the first time will dowload the dataset in `./experiments/CIFAR10/data/`, you can modify the path in `dataset.py`

