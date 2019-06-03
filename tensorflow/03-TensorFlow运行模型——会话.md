# TensorFlow运行模型——会话

## TensorFlow会话的概念

TensorFlow中的会话（session）是用来执行定义好的运算。

会话拥有并管理 TensorFlow 程序运行时的所有资源。所有计算完成之后需要关闭会话来帮助系统回收资源,否则就可能出现资源泄漏的问题。

## TensorFlow会话的使用

TensorFlow 中使用会话的模式一般有两种。

第一种模式需要明确调用会话生成函数和关闭会话函数,这种模式的代码流程如下。

```python3
# 创建一个会话。
sess = tf.Session()
# 使用这个创建好的会话来得到关心的运算的结果。比如可以调用 sess.run (result) 
# 来得到 3.1 节样例中张量 result 的取值。
sess.run(...)
# 关闭会话使得本次运行巾使用到的资源可以被释放。
sess.close()
```

这种模式，在执行所有计算之后，需要调用Session.close函数来关闭会话并释放资源。

但是，如果程序存在异常导致程序终止时，关闭会话的函数可能就不会被执行从而导致资源泄露。为了避免这种**异常退出导致的资源释放问题**，TensorFlow可以通过python的上下文管理器来使用会话。具体代码如下所示

```
#创建一个会话,并通过 Python 中的上下文管理器来管理这个会话。
with tf. Session() as sess :
    #使用创建好的会话来计算关心的结果。
    sess. run ( ... )
#不需要再调用“ Session.cl。 se ()”函数来关闭会话,
#当上下文退出时会话关闭和资源释放也自动完成了。
```

通过 Python 上下文管理器的机制,**只要将所有的计算放在 “ with ”的内部就可以** 。 当上下文管理器退出时候会自动释放所有资源。这样既解决了因为**异常退出**时资源释放的问题,同时也解决了**忘记调用** Session.close 函数而产生的资源泄漏。

第二种模式通过设定默认会话计算张量的取值。

```
# 通过设置默认会话计算张量的取值
sess = tf.Session()
with sess.as_default():
    print(result.eval())
```

以下代码也可以完成相同的功能。

```
# 与上面代码功能相同
sess = tf. Session()
#以下两个命令有相同的功能。
pr 工 nt(sess .run(result))
print(result.eval (session=sess))
```

在交互式环境下(比如 Python 脚本或者 Jupyter 的编辑器下),通过设置默认会话的方式来获取张量 的取值更加方便 。 所以 TensorFlow 提供了 一种在交互式环境下直接构建默认会话的函数。这个函数就是 tf.lnteractiveSession 。使用这个函数会自动将生成 的 会话注册为默认会话。以下代码展示了 tf.InteractiveSession 函数的用法。

```
# 交互式环境下构建默认会话的函数
sess = tf.InteractiveSes sion()
prir > t (result. eval () )
sess. close ()
```

通过 tf.InteractiveSession 函数可以省去将产生的会话注册为默认会话的过程。无论使
用哪种方法都可以通过 ConfigProto Protocol BufferCD来配置需要生成的会话 。 下面给出了通
过 ConfigProto 配置会话的方法 :

```
config = tf.ConfigProto(al low soft placement=True,
log_device_placem ent=True)
sessl = tf.InteractiveSes sion(config=conf ig)
sess2 = tf.Session(config =config)
```

通过 ConfigProto 可以配置类似并行的线程数、 GPU 分配策略、运算超时时间等参数。在这些参数中,最常使用的有两个。第 一个是 allow_so位_placement ,这是一个布尔型的参数,当它为 True 时, 在以下任意一个条件成立时, GPU 上的运算可 以放到 CPU 上进行 :
1.运算无法在 GPU 上执行 。

2.没有 GPU 资源(比如运算被指定在第 二个 GPU 上运行 ,但是机器只有一个 GPU ) 。

3.运算输入包含对 CPU 计算结果的引用 。
这个参数的默认值为 False ,但是为了使得代码的可移植性更强,在有 GPU 的环境下这个参数一般会被设置为 True 。不同的 GPU 驱动版本可能对计算的支持有略微的区别,通过将 allow_ soft _placement 参数设为 True , 当某些运算无法被当前 GPU 支持时,可 以自动调整到 CPU 上,而不是报错。类似地,通过将这个参数设置为 True ,可以让程序在拥有不同数量的 GPU 机器上顺利运行。

第二个使用得比较多的配置参数是log_device_placement。 这也是一个布尔型的参数,当它为 True 时日志中将会记录每个节点被安排在 哪个设备上以方便调试 。而在生产环境中将这个参数设置为 False 可以减少日志量。