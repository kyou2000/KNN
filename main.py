import knn

if __name__ == '__main__':
    class_name = ['喜剧片','动作片','爱情片']
    #class_id = [0,1,2]
    X = [[39,0,31],[3,2,65],[2,3,55],[9,38,2],[8,34,17],[5,2,57],[21,17,5],[45,2,9]]
    Y = [0,1,2,2,2,1,0,0]
    #需要推测的数据
    #唐人街探案,23,3,17
    test = [23,3,17]
    #进行推测
    for k in range(1,6):
        res = knn.KNN_Classify(X,Y,k,test)
        print(f'k={k}')
        print(class_name[res])
