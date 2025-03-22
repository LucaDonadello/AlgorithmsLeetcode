class Tree:
    def __init__(self, val = 0, left = None, right = None):
        self.left = left
        self.right = right
        self.val = val

class Solution:
    def createTree(val):
        root = Tree(val)
        root.left = Tree(10)
        root.right = Tree(20)

        return root

test = Solution.createTree(15)
print(test.left.val)
