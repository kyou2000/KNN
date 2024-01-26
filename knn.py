import numpy as np
#KNN主要计算

#t为测试样本，s为原始类别样本
#样本间的距离计算，输入的必须为向量
def distance(t,s):
    t_shape=t.shape
    s_shape=s.shape
    assert t_shape == s_shape,'输入的形状不一致!'
    dis= np.sqrt(np.sum(pow(t-s,2)))
    return dis

def is_exist(list,x):
    f=False
    for i in list:
        if x == i:
            f=True
    return f

def get_index(list,x):
    length=len(list)
    index=-1
    for i in range(length):
        if list[i] == x:
            index=i
    return index

#x,y是列表，x中是向量，y中是数字标签,k为相邻的k个邻居,test为测试样例
def KNN_Classify(x,y,k,test):
    len_x=len(x)
    len_y=len(y)
    test_arr = np.array(test)
    test_shape=test_arr.shape
    train_x = toArray(x)
    x_shape=train_x[0].shape
    assert len_x != 0,'列表不能为空!'
    assert len_y == len_x,'数据集与标签的数量必须一致!'
    assert test_shape == x_shape,'测试样例与数据集中的形状必须一致!'
    assert k != 0,'k!=0'

    #保存距离
    dis=[]
    for i in range(len_x):
        dis_x=distance(train_x[i],test_arr)
        dis.append(dis_x)
    #对距离进行排序
    for i in range(len_x):
        for j in range(len_x-1-i):
            if dis[j]>dis[j+1]:
                #将距离表排好顺序的同时将标签表也排序
                sw=y[j]
                temp=dis[j]
                dis[j]=dis[j+1]
                y[j]=y[j+1]
                dis[j+1]=temp
                y[j+1]=sw
    #选中前k个邻居,加入列表中
    labels=[] #记录K个邻居的标签
    num=[] #每个标签的个数
    for i in range(k):
        if is_exist(labels,y[i]) == False:
            labels.append(y[i])
            num.append(1)
        else:
            index=get_index(labels,y[i])
            num[index]=num[index]+1
    #从统计好的数据当中选择出类别种类最多的
    temp1=num[0]
    index1=-1
    for i in range(len(num)):
        if temp1 <= num[i]:
            temp1=num[i]
            index1=i
    pre_label=-1
    if index1 != -1:
        pre_label=labels[index1]

    return pre_label

def toArray(x):
    new_list = []
    for i in x:
        arr = np.array(i)
        new_list.append(arr)
    return new_list
