# class Tree:
#     def __init__(self, val = 0, left = None, right = None):
#         self.left = left
#         self.right = right
#         self.val = val

# class Solution:
#     def createTree(val):
#         root = Tree(val)
#         root.left = Tree(10)
#         root.right = Tree(20)

#         return root

# test = Solution.createTree(15)
# print(test.left.val)


# str = "Hello Worldhhhhh"
# str = str.lower()
# hash = {}
# maxVal = 0
# curChar = ""

# for i in str:
#     if i in hash:
#         hash[i] += 1
#     else:
#         hash[i] = 1

# for i in hash:
#     if hash[i] > maxVal:
#         maxVal = hash[i]
#         curChar = i

# for i in range(len(str)):
#     if curChar == str[i]:
#         print(i)
#         break