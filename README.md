# 《Python深度学习》代码及笔记
&emsp;&emsp;Francois Chollet的《Python深度学习》介绍了用Python和Keras进行深度学习的探索实践，涉及计算机视觉、自然语言处理、生成式模型等应用。书中包含30多个代码示例，步骤讲解详细透彻。  

## 使用说明
1. 本笔记是搭配《Python深度学习》一书来阅读。
2. 关于本笔记中的代码，本书中的源码是基于keras的，但是后端用的是tensorflow1.X的引擎，所以在tensorflow2上面会有很多报错，比如第8章的代码中，就有很多Tensorflow的广播方法报错。  
3. 本笔记中的有些代码采用了TensorFlow2重写（代码目录：tensorflow_V2_src），主要还是动态图的问题，导致有些代码重写难度大。
4. 关于python包的版本问题，请详见requirements.txt文件，笔者采用的tensorflow-gpu==2.0.0，可以使用cpu版本，但是运行会特别慢。
5. 相关数据集见百度网盘的下载地址：链接：https://pan.baidu.com/s/1XdkibXpL-UNG0dXk1w5fNw 提取码：u8pn
6. keras模型与数据下载地址：链接：https://pan.baidu.com/s/1Rt6KYWUAQ8MWKY9UVVDtmQ 提取码：wedp  
7. 由于笔者的电脑配置不行，推荐大家租用GPU服务器来运行示例代码，租用方式链接：https://mp.weixin.qq.com/s?__biz=MzU0NjczNTg2NQ==&mid=2247486813&idx=2&sn=93e72cdf73675df69bcbe32de057f6dc&chksm=fb585ecbcc2fd7ddf3858cff26be550e37b578fb522446e7fcf1b71f2f0788231f730cc9f03c&token=1693266535&lang=zh_CN#rd

## 选用的《Python深度学习》版本
<img src="./resources/deep-learning-with-python-book.png" width="336" height= "393">


> 书名：Python深度学习<br/>
> 作者：Francois Chollet<br/>
> 译者：张亮<br/>
> 出版社：人民邮电出版社<br/>
> 版次：2018年8月第1版<br/>

## 主要贡献者（按首字母排名）
 [@胡锐锋-天国之影-Relph](https://github.com/Relph1119)

## LICENSE
[GNU General Public License v3.0](https://github.com/relph1119/deep-learning-with-python-notebooks/blob/master/LICENSE)