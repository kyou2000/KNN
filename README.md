KNN算法

KNN又称K近邻算法，是一种常用的分类算法，采用测试数据特征与数据集中距离最近的K个数据的标签作为测试数据的类别。

我的程序是模拟了KNN算法中的距离计算以及按照数据集的标签对测试数据的类别推测。

程序使用python编写，我使用的数据集已经放到文件中，需要的库为numpy。

程序中的距离计算采用的是空间中的点到点的距离公式计算，即欧式距离。在选出的K个邻居中，这些邻居的标签可能会不相同，应采取类别数量最多的标签作为测试数据的类别。

KNN算法的文件为knn.py，测试类文件为main.py，在测试时已经选取了不同的k值。使用的数据集为电影数据集。


