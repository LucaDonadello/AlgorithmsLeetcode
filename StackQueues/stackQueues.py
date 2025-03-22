from ast import List
from collections import deque

# Stack and Queue problems

# 20. Valid Parentheses
'''
Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1) Open brackets must be closed by the same type of brackets.
2) Open brackets must be closed in the correct order.
3) Every close bracket has a corresponding open bracket of the same type.
'''

def isValid(self, s: str) -> bool:
    '''
    This algorithm was inspired by the stack data structure. The idea is to use a stack to store the open brackets.
    The algorithm iterates through the string in the input and checks if the character is an open bracket. If it is, we add the character to the stack.
    If the character is a close bracket, we check if the stack is empty. If it is, we return False. If it is not, we pop the last element from the stack and check if the open bracket is the same as the close bracket.
    If it is not, we return False. Finally we check if the stack is empty. If it is, we return True. If it is not, we return False.
    Time complexity is O(n) where n is the length of the string.
    Space complexity is O(n) where n is the length of the string.
    '''

    stack = []

    for i in s:
        if i in ["(", "[", "{"]:    # Remember this is constant time since it is a limited set of characters (3)
            stack.append(i)
        elif len(stack) != 0:  
            check = stack.pop()
            if check == "(" and i != ")" or check == "[" and i != "]" or check == "{" and i != "}":
                return False
        else:
            return False
    
    return len(stack) == 0

# 739. Daily Temperatures

'''
Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i]
is the number of days you have to wait after the ith day to get a warmer temperature.
If there is no future day for which this is possible, keep answer[i] == 0 instead.
'''

def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
    '''
    I did not code the naive way to calculate this since it is not optimal and it takes O(n^2) time complexity.
    This instead is the optimal solution which uses a stack to store the indices of the temperatures.
    The algorithm iterates through the array in reverse and checks if the temperature is greater than the temperature at the top of the stack.
    If it is, we pop the last element from the stack and calculate the difference between the current index and the index at the top of the stack.
    If it is not, we add the index to the stack. Finally we return the result.
    Time complexity is O(n) where n is the length of the temperatures.
    Space complexity is O(n) where n is the length of the temperatures
    '''
    stack = []
    res = [0] * len(temperatures)

    for i in range(len(temperatures) - 1, -1, -1):
        while stack and temperatures[i] >= temperatures[stack[-1]]:
            stack.pop()
        if stack:
            res[i] = stack[-1] - i
        stack.append(i)
    return res

# 155. Min Stack

'''
Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

MinStack() initializes the stack object.
void push(int val) pushes the element val onto the stack.
void pop() removes the element on the top of the stack.
int top() gets the top element of the stack.
int getMin() retrieves the minimum element in the stack.

You must implement a solution with O(1) time complexity for each function.
'''

class MinStack:

    '''
    The idea of this algorithm is to use a stack to store the minimum value of the stack.
    The algorithm uses a tuple to store the value and the minimum value of the stack. The algorithm initializes the stack as an empty list.
    The algorithm pushes the value and the minimum value to the stack. If the value is less than the minimum value, we update the minimum value.
    The algorithm pops the last element from the stack. The algorithm returns the top element of the stack. The algorithm returns the minimum value of the stack.
    The time complexity is O(1) for each function.
    The space complexity is O(n) where n is the number of elements in the stack.
    '''

    def __init__(self):
        self.minStack = []

    def push(self, val: int) -> None:
        if not self.minStack:
            self.minStack.append((val, val))
        else:
            if self.minStack[-1][1] > val:
                self.minStack.append((val, val))
            else:
                self.minStack.append((val, self.minStack[-1][1]))

    def pop(self) -> None:
        self.minStack = self.minStack[:len(self.minStack)-1]

    def top(self) -> int:
        return self.minStack[-1][0]

    def getMin(self) -> int:
        return self.minStack[-1][1]
    
# 496. Next Greater Element I

'''
The next greater element of some element x in an array is the first greater element that is to the right of x in the same array.
You are given two distinct 0-indexed integer arrays nums1 and nums2, where nums1 is a subset of nums2.
For each 0 <= i < nums1.length, find the index j such that nums1[i] == nums2[j] and determine the next greater element of nums2[j] in nums2.
If there is no next greater element, then the answer for this query is -1.
Return an array ans of length nums1.length such that ans[i] is the next greater element as described above.'
'''

def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
    # I would avoid the naive since it is just a double loop which will run in O(n^2)
    '''
    This algorithm uses a stack to store the indices of the numbers in the array. The algorithm iterates through the array in reverse and checks if the number is greater than the number at the top of the stack.
    If it is, we pop the last element from the stack and check if the number is greater than the number at the top of the stack. If it is, we pop the last element from the stack.
    If it is not, we append the index to the stack. Finally we return the result.
    Time complexity is O(n) where n is the length of the array.
    Space complexity is O(n) where n is the length of the array
    -- This is very similar to the daily temperatures problem
    '''
    # stack = []
    # minArr = [-1] * len(nums2)
    # res = []

    # for i in range(len(nums2)-1,-1,-1):
    #     while stack:
    #         if nums2[stack[-1]] < nums2[i]:
    #             stack.pop()
    #         else:
    #             minArr[i] = nums2[stack[-1]]
    #             break
    #     stack.append(i)
    
    # for i in nums1:
    #     res.append(minArr[nums2.index(i)])
    
    # return res

    '''
    This algorithm is the same as the previous one but it uses a dictionary to store the indices of the numbers in the array.
    The algorithm iterates through the array in reverse and checks if the number is greater than the number at the top of the stack.
    If it is, we pop the last element from the stack and check if the number is greater than the number at the top of the stack. If it is, we pop the last element from the stack.
    If it is not, we append the index to the stack. Finally we return the result.
    Time complexity is O(n) where n is the length of the array.
    Space complexity is O(n) where n is the length of the array
    '''
    
    stack = []
    minMap = {}
    res = []

    for i in range(len(nums2)-1,-1,-1):
        while stack and nums2[stack[-1]] < nums2[i]:
            stack.pop()

        minMap[nums2[i]] = nums2[stack[-1]] if stack else -1
        stack.append(i)
    
    for i in nums1:
        res.append(minMap[i])
    
    return res

# 239. Sliding Window Maximum

'''
You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window.
Each time the sliding window moves right by one position.
Return the max sliding window.
'''

def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
    '''
    The naive solution is to have a nested loop with a pointer at the start of the array.
    The outer loop will iterate through the array and the inner loop will iterate through the window.
    The algorithm will calculate the maximum of the window and append it to the result. This is not the optimal solution since it takes O(n^2) time complexity.
    '''

    '''
    This is the optimal implementation of the algorithm. The idea is to use a deque to store the indices of the numbers in the array.
    The algorithm iterates through the array and checks if the number is greater than the number at the top of the deque.
    If it is, we pop the last element from the deque. Then we append the index to the deque. If the index is greater than k, we check if the number at the top of the deque is the number at the index k.
    If it is, we pop the last element from the deque. Finally we append the number at the top of the deque to the result.
    Time complexity is O(n) where n is the length of the array.
    Space complexity is O(n) where n is the length of the array.
    '''

    # This is how to declare the queue
    queue = deque()
    res = []

    # I iterate through the array using enumerate to get the index and the number in the array
    for i, num in enumerate(nums):
        # I check if the number is greater than the number at the top of the queue
        while queue and queue[-1] < num:
            # If it is, I pop the last element from the queue
            queue.pop()

        # Append the value to the queue
        queue.append(num)

        # check if the sliding window is greater than k 
        if i >= k and nums[i-k] == queue[0]:
            queue.popleft()
        
        # Append the value at the top of the queue to the result
        if i >= k - 1:
            res.append(queue[0])
        
        '''
        Example of how the algorithm works
        nums = [1,3,-1,-3,5,3,6,7]
        k = 3

        queue = [1]
        queue = [3, 1]
        queue = [3, 1, -1]
        res = [3]
        queue = [3, 1, -1, -3] --> Since - 3 is not greater than the last element
        res = [3, 3]
        queue = [5, 3, 1, -1, -3]
        ...
        This creates a decreasing queue where the first element is the maximum of the window
        If the first element is out of the window, we remove it from the queue
        '''
    
    return res

# 232. Implement Queue using Stacks

'''
Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).
Implement the MyQueue class:
void push(int x) Pushes element x to the back of the queue.
int pop() Removes the element from the front of the queue and returns it.
int peek() Returns the element at the front of the queue.
boolean empty() Returns true if the queue is empty, false otherwise.

'''

class MyQueue:
    def __init__(self):
        self.lst = []

    def push(self, x: int) -> None:
        self.lst.append(x)
        
    def pop(self) -> int:
        val = self.lst[0]
        self.lst = self.lst[1:]
        return val

    def peek(self) -> int:
        return self.lst[0]

    def empty(self) -> bool:
        if len(self.lst) > 0:
            return False
        else:
            return True
