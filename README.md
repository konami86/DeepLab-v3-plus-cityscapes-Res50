# DeepLab V3plus
The implementation of [Deeplab_v3plus](https://arxiv.org/abs/1802.02611). This repository is based on the dataset of cityscapes and the mIOU is 80.12.

I am working with python3.5 and pytorch1.0.0 built from source. Other environments are not tested, but you need at least pytorch1.0 since I use torch.distributed to manipulate my gpus. I use two 1080ti to train my model, so you also need two gpus each of which should have at least 9G memory.


## Dataset
The experiment is done with the dataset of [CityScapes](https://www.cityscapes-dataset.com/). You need to register on the website and download the dataset images and annotations. Then you create a `data` directory and then decompress.
```
    $ cd DeepLabv3plus
    $ mkdir -p data
    $ mv /path/to/leftImg8bit_trainvaltest.zip data
    $ mv /path/to/gtFine_trainvaltest.zip data
    $ cd data
    $ unzip leftImg8bit_trainvaltest.zip
    $ unzip gtFine_trainvaltest.zip
```


## Train && Eval
After creating the dataset, you can train on the cityscapes train set and evaluate on the validation set.  
Train: 
```
    $ cd DeepLabv3plus
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
```
This will take around 13 hours on two 1080ti gpus. After training, the model will be evaluated on the val set automatically, and you will see a mIOU of 80.12%(backbone res101, evaluated with mutiple scale and flip). 78.477%(backbone res101, single scale, no flip) after 40000 iters

Eval:
If you want to evaluate a trained model, you can also do this: 
```
    $ python evaluate.py
```
or if you want to evaluate on multi-gpus, you can also do this: 
```
    $ CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 evaluate.py
```

## Configurations
* If you want to use your own dataset, you may implement you `dataset` file as does with my `cityscapes.py`. 

* As for the hyper-parameters, you may change them in the configuration file `configs/configurations.py`.


### Outreach

We re-trained the model and got a mIoU of 78.457% after ~60000 iters(backbone res101, evaluated with single scale and no flip), but we got an unbelievable mIoU of 78.398 after ~60000 iters when using res50 as backbone (evaluated with single scale and no flip).

我们使用了别人(coin)实现的pytorch版本的deeplabv3+(https://github.com/CoinCheung/DeepLab-v3-plus-cityscapes)  在cityscape上进行训练，结果和coin几乎一致. 但是我们将代码的backbone换成了res50，其余所有不变，精度几乎达到了res101的精度，单尺度evl集达到了惊人的78.398(只比deeplabv3+使用Res101的backbone低了0.06%，比同样使用res50 backbone的PSPNet精度高了1个点 https://github.com/hszhao/semseg).
更让人意外的是deeplabv3+原作tensorflow (https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md) 中使用xception65 backbone的结果只有78.79%，xception71为80.31% 

所以是代码的原因还是说deeplabv3+在cityscape上其实backbone使用res50的就可以达到接近于res101、xception65 的精度？？？？？

相关记录见slurm-11413.out(60000次迭代版本)


