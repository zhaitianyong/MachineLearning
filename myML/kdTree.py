import numpy as np
import math


class Node:
    def __init__(self, data, parent=None, left=None, right=None):
        self.data = data
        self.parent = parent
        self.feature = None
        self.split = None
        self.left = left
        self.right = right


class kdTree:
    def __init__(self):
        self.rootNode = None

    def build(self, dataSet, pNode=None, depth=0):
        if (len(dataSet) > 1):
            m, n = np.shape(dataSet)  # rows cols
            ## 计算每列的方差或者标准差，
            std_array = [np.std(dataSet[:, i]) for i in range(n)]
            ## 选择最大的一列，然后排序
            max_index = np.argsort(std_array)[n - 1]
            # print(max_index)
            ## 以第std_max_indexes[0]列为节点 进行排序
            middle_indexes = np.argsort(dataSet[:, max_index])
            middle_index = middle_indexes[m // 2 - 1]
            # print(middle_index)
            # print(dataSet[middle_index, max_index])
            node = Node(dataSet[middle_index, :], pNode)
            node.split = max_index
            node.feature = dataSet[middle_index, max_index]
            if (self.rootNode is None):
                self.rootNode = node
            leftDataSet = dataSet[middle_indexes[:m // 2 - 1]]
            rightDataSet = dataSet[middle_indexes[m // 2:]]
            if (len(leftDataSet) == 1):
                node.left = Node(leftDataSet[0], node)
            else:
                node.left = self.build(leftDataSet, node, depth + 1)
            if (len(rightDataSet) == 1):
                node.right = Node(rightDataSet[0], node)
            else:
                node.right = self.build(rightDataSet, node, depth + 1)
            #print("depth = ", depth)
            return node

    def printKdTree(self, node):
        if node:
            print(node.data)
            print("feature = ",node.feature)
            print("split = ", node.split)
            if node.left:
                self.printKdTree(node.left)
            if node.right:
                self.printKdTree(node.right)

    def serach(self, x, k=1):
        self.x = x
        self.best_node = None
        searchNode =self.rootNode;
        while(searchNode):
            if(searchNode.split is None):
                break;
            if self.x[searchNode.split] >= searchNode.feature:
                if(searchNode.right is None):
                    break;
                searchNode = searchNode.right
            else:
                if (searchNode.left is None):
                    break;
                searchNode = searchNode.left
        nearlist=[]
        np = searchNode;
        nearlist.append(np)
        while np.parent:
            if np.parent:
                nearlist.append(np.parent)
            np = np.parent

        for n in nearlist:
            print(n.data)


    def distance(self, X1, X2):
        return math.sqrt(np.sum((X1 - X2) ** 2))



if __name__ == '__main__':
    dataSet = [[2, 3, 8],
               [5, 4, 2],
               [9, 6, 4],
               [4, 7, 9],
               [8, 1, 7],
               [7, 2, 6]]
    kdTree = kdTree()
    kdTree.build(np.array(dataSet))
    #kdTree.printKdTree(kdTree.rootNode)
    kdTree.serach([5, 3, 10])
