# eg
**粗体**
-------------------------
*斜体*
## 欢迎来到[梵居闹市](http://blog.leanote.com/freewalk)
<font color=red size=6 face ="黑体">红色</font>



# Github 

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

#####
在这个任务中，主要是一个简单的二分类分割任务，输入的图像是png格式，mask是通过rel编码转换过来的灰度图像。
数据集方面：这里采用的是交叉验证的方法，每次可以选择正负样本的比例
模型方面选择的是se_resnet-unet
训练过程中先是训练出一个初始模型，然后后面的阶段再根据原来的模型进行微调，每个阶段都去改变正负样本的比例

在目前阶段最好的训练方法是：负样本比正样本多1200个，这样的话就不用使用focalloss了，直接选用dice和bce，学习率1e-4，最小1e-6，用余温退火的学习率更新策略，结果显示在20个epoch后
就差不多收敛完成啦。
最终在测试的时候有两种方法，一种是双重阈值的方法（0.75，1000，0.5）来先判断大于0.75的像素点是否大于1000，如果不大于则表示没有气胸，如果大于了则表示有气胸，然后再用0.5的阈值来
生成mask，个人觉得这个不是特别好
另一种方法是直接用阈值先生成一个二值图像，然后用连通域标记图像，保留连通域大于min_size的地方，这样的话能去除掉较多的噪声干扰，而具体的阈值和size需要根据具体情况去不断的调整尝试。

