#倾斜检测

## yanmei 2020.05--改动说明
- 这次的样本用陈涛给的7700张样本（两个数据集合并），随机抽500做验证集，7200做训练集
- 其中，500张做旋转增加到2000张做测试集，每类样本500张，样本均衡
- 增加了切图，把每一张大图切成几十张小图，再训练
- train_batch=6，抽取6张大图，随机抽取48张小图，然后全部转正，再统一旋转，保证样本均衡，进行训练
- 验证的时候每一张大图切出小图，将全部小图喂给模型输出一堆标签，取众数做当前大图的标签，验证100张大图，计算一次正确率
  ## 2020.05.15训练结果如下（标签对应[0,1,2,3]=[0,90,180,270]）：
- accuracy=0.978(小图的正确率)
- recall=0.99（小图中标签为0的正确率）
- F1=1（大图的正确率）
  ## 2000张测试集，对比赵毅模型、老模型（创哥训练的大图）、新模型（切小图）三个模型的正确率：
- 赵毅模型：0.976
- 老模型：0.90
- 新模型：0.9975（2000张预测错了5张）

  ## 2020.05.29因左右标签跟生产上的不一致，对换以后重新训练[0,1,2,3]=[0,270,180,90]，结果如下：
- accuracy=0.973(小图的正确率)
- recall=0.984（小图中标签为0的正确率）
- F1=1（大图的正确率）
- 新模型的正确率accuracy: 0.996（2000张预测错了8张）


## piginzoo 2019.7

### 目录说明
raw.train和raw.validate应该没什么用了，可以忽略
origin目录存放着所有的图，用来 里面一张图片会按照角度，旋转4次，生成4个样本，到train目录下。
background里面放着各种的背景图，用于合成新的图片，这里面的背景是多样化的。


### 这次的改动
这次是因为涛那边给的反馈是效果不好，所以要做二次训练
所以岩来做新的样本生成，重点是加上背景，特别是黑背景
这次的验证集不用自己生成了，用淘反馈回来的220多张的真实样本（都是我们没有识别出来或者识别错的）

岩的做法是：
1、从raw.train里面所有的文件，做shuffle，然后取出2000张
2、对每一张，做4、8张变形，分别对应0、90、180、270度的旋转，并且伴随着做透射，然后贴到随机的一张背景图上
3、把生成的每一张，保存到 data/second目录下，并且将其标签写入到second.txt中："xxxxx_1.png   1”，1表示90度旋转，例如。
4、手工把second目录下的文件拷贝到train中，切记，要提前先备份一下train，原因是有可能再生成second的文件
5、手工合并second.txt到train.txt，使用cat train.txt second.txt > train.new.txt

细节的样本生成方法：
1.原图做transkform,梯形，背景是黑色，得到前景图
2.前景图做二值化，得到mask图
3.背景图.past(pos, 前景图，mask=mask图)


## piginzoo 2019.5
This is a pretty simple project. It will detect whether the bill picture is inclined.

It use VGG as backbone, then send the tensor to a very simple full connection layer, which is 256 dimensions, the last layer is a 4 classes output, represents 4 directions: no lean,  or lean 90' clockwise, or 180' or 270'.

Train.py is in charge of training work, you can run it by bin/train.sh, you can adjust the parameters in the script for those hyper-parameters.

Pred.py can help you predict the picture direction, it will load the model, restore all weights of the network, create a session then predict the picture.

There is also a web server, who did almost same function with pred.py, but more is, it will expose the restful api to other application. You can use it to combine with your application to do more things.

And in data_generators, you can find the code who help to create train fake data and labels, I did not upload the raw data, but if you need, i can share some. Just free to contact me by email: piginzoo@gmail.com.


