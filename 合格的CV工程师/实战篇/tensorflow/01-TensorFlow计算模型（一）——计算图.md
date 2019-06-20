# TensorFlow计算模型——计算图

## 关于TensorFlow

​	TensorFlow是一个采用数据流图（data flow graphs），用于数值计算的开源软件库。节点（Nodes）在图中表示数学操作，图中的线（edges）则表示在节点间相互联系的多维数据数组，即张量（tensor）。它灵活的架构让你可以在多种平台上展开计算，例如台式计算机中的一个或多个CPU（或GPU），服务器，移动设备等等。TensorFlow 最初由Google大脑小组（隶属于Google机器智能研究机构）的研究员和工程师们开发出来，用于机器学习和深度神经网络方面的研究，但这个系统的通用性使其也可广泛用于其他计算领域

​        计算图是TensorFlow中最基本的一个概念，TensorFlow中的所有计算都会被转化为计算图上的节点。

## 计算图的概念

​        TensorFlow从其名称可以获取的信息是，其包含了两个最重要的概念——Tensor和Flow。

- [ ] Tensor的概念

​        Tensor就是张量。张量的概念在数学或物理学中有不同的解释，但在TensorFlow中并不强调它本身的含义，而是将其简单地理解为**多维数组**。

- [ ] Flow的概念

​        如果说TensorFlow中的Tensor表明了它的**数据结构**，那么Flow则体现了它的**计算模型**。中文将Flow翻译为“流”，它直观的表达了**张量之间通过计算相互转化的过程**。

- [ ] TensorFlow中的计算图

​        TensorFlow是通过计算图的形式来表述计算的编程系统。TensorFlow中的每个计算都是计算图上的一个节点，而节点之间的边描述了计算之间的依赖关系。

![](../image/20171221100429378.gif)



## 计算图的使用

TensorFlow程序的一般流程

第一，定义计算图中的所有的计算

第二，执行计算

- [ ] 计算定义阶段

  ```
  import tensorflow as tf
  a = tf.constant([1.0, 2.0], name="a")
  b = tf.constant([2.0, 3.0], name="b")
  result = a + b
  ```

​        在python中一般会采用“import tensorflow as tf”的形式来载入TensorFlow，这样子使用“tf”来代替“tensorflow”作为模块名称，使得整个程序更加简洁。

- [ ] 获取默认计算图以及如何查看一个运算所属的计算图

  ```
  # 通过a.graph可以查看张量所属的计算图。因为没有特意指定，所以这个计算图应该等于
  # 当前默认的计算图。所以下面这个操作输出值为True。
  print(a.graph is tf.get_default_graph())
  ```

​        除了使用默认的计算图，TensorFlow支持通过tf.graph函数来生成新的计算图。不同计算图上的张量和运算都不会共享。以下代码示意了如何在不同计算图上定义和使用变量。

```
import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
    # 在计算图“g1”中定义变量“v”，并设置初始值为0。
    v = tf.get_variable("v",initializer=tf.zeros_initializer()(shape=[1]))
    
g2 = tf.Graph()
with g2.as_default():
    # 在计算图“g1”中定义变量“v”，并设置初始值为1。
    v = tf.get_variable("v",initializer=tf.ones_initializer()(shape=[1]))
    
# 在计算图g1中读取变量“v”的取值。
with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中，变量“v”的取值一个为0，所以下面这行会输出[0.]。
        print(sess.run(tf.get_variable("v")))
        
# 在计算图g2中读取变量“v”的取值。
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("",reuse=True):
        # 在计算图g1中，变量“v”的取值一个为0，所以下面这行会输出[1.]。
        print(sess.run(tf.get_variable("v")))
```

​        以上代码产生了两个计算图，每个计算图中定义了一个名字为“v”的变量。在计算图g1中，将v初始化为0；在计算图g2中，将v初始化为1。可以看出来当运行不同计算图时，变量v的值也是不一样的。**TensorFlow中的计算图不仅仅可以用来隔离张量和计算，它还提供了管理张量和计算的机制**。计算图可以通过tf.Graph.device函数来制定运行计算的设备。这为TensorFlow使用GPU提供了机制。以下程序可以将加法计算泡在GPU上。

```
g = tf.Graph()
# 指定计算运行的设备。
with g.device('/gpu:0'):
    result = a+b
    print(result)
    
```



![点关注不迷路，我们一起上高速](../image/AI_study.jpg)





