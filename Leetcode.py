from ast import List
from collections import deque
from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

## Leetcode Problems

# Strings and Arrays problems

# 1. Two Sum
'''
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
'''


def twoSum(self, nums: List[int], target: int) -> List[int]:
    '''
    The Idea I had initially is to have a nested loop with a pointer to the start of the array.
    Each time we iterate through the array, we check if the sum of the start and the current index is equal to the target.
    If it is, we return the indices of the two numbers. Othewrise, we increment the start pointer and repeat the process.
    Of Course this is not the optimal solution since it takes O(n^2) time complexity.
    '''
    # start = 0

    # while start < len(nums):
    #     for i in range(start+1,len(nums)):
    #         if nums[start] + nums[i] == target:
    #             return [start,i]
    #     start += 1

    # return []

    '''
    Then an idea came to my mind. I could use the subtraction of the target and the current number to check if the difference is in the array.
    This solution can be ideal but I have used a non optimal way to find the indices. This indeed is an improvement but it takes still O(n^2) time complexity.

    '''

    # for i in range(len(nums)):
    #     check = target - nums[i]
    #     if check in nums and i != nums.index(check):
    #         return [i,nums.index(check)]
    # return []

    '''
    Finally, I understood that I could use a dictionary to store the indices of the numbers in the array. This is particularly good because it only takes O(1) to find the value in the dictionary.
    The main idea is to iterate through the array and check if the difference between the target and the current number is in the dictionary. If is not I add the number to the dictionary. 
    If it is, I return the indices of the two numbers. This is the optimal solution since it takes O(n) time complexity.
    '''
    
    dp = {}

    for i, num in enumerate(nums):
        if target - num in dp:
            return [i, dp[target-num]]
        dp[num] = i
    return []

# 2. Best Time to Buy and Sell Stock

'''
You are given an array prices where prices[i] is the price of a given stock on the ith day.
You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
'''
def maxProfit(self, prices: List[int]) -> int:
    '''
    This is the first implementation that came to my mind. What it does it to iterate through the array and check the difference between the current number and the rest of the numbers.
    If the difference is greater than the maxProfit, we update the maxProfit. This is not the optimal solution since it takes O(n^2) time complexity.
    '''

    # maxProfit = 0
    # start = 0

    # while start < len(prices):
    #     for i in range(start+1,len(prices)):
    #         if prices[i] - prices[start] > maxProfit:
    #             maxProfit = prices[i] - prices[start]

    #     start += 1
    
    # return maxProfit

    '''
    This is the optimal solution. The idea behind this algorithm is to keep track of the current profit and the minimum value of the stock.
    The algorithm iterates through the array and checks if the currect profit is maximum. If it is not, we update the profit.
    Then the algorith checks if the current value is less than the minimum value. If it is, we update the minimum value.
    '''
    profit = 0
    currVal = prices[0]
    for i in prices[1:]:
        profit = max(profit, i - currVal)
        if currVal > i:
            currVal = i
    
    return profit

# 238. Product of Array Except Self

'''
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
You must write an algorithm that runs in O(n) time and without using the division operation.
'''

def productExceptSelf(self, nums: List[int]) -> List[int]:
    '''
    The idea I had is to split the product in two parts. The first part is the product of the numbers to the left of the current number.
    The second part is the product of the numbers to the right of the current number. Then we multiply the two parts to get the final product.
    This will have time complexity of O(n) but it will have space complexity of O(n) as well. Remember Prefix and Suffix is the key for this problem.
    '''
    l = [1] * len(nums)
    r = [1] * len(nums)

    curProd = 1

    for i in range(len(nums)):
        if i != 0:
            l[i] = curProd
        curProd *= nums[i]
        
    curProd = 1

    for i in range(len(nums)-1,-1,-1):
        if i != len(nums)-1:
            r[i] = curProd
        curProd *= nums[i]
        l[i] *= r[i]    

    return l

# 53. Maximum Subarray

'''
Given an integer array nums, find the with the largest sum, and return its sum.
'''

def maxSubArray(self, nums: List[int]) -> int:
    
    '''
    The idea I had initially was to have a nested loop with a pointer at the start of the array.
    This will check all the combination of the subarrays and return the maximum sum. This is not the optimal solution since it takes O(n^2) time complexity.
    '''
    # start = 0
    # maxSub = float(-inf)

    # while start < len(nums):
    #     counter = 0
    #     for i in range(start,len(nums)):
    #         counter += nums[i]
    #         maxSub = max(maxSub, counter)
    #     start += 1
    
    # return maxSub

    '''
    This is the optimal solution. The idea is to keep track of the current sum, check if the current sum is greater than the actual value at i, if so update the sum with the ith value.
    If the ith value is not greater than the sum, we add the ith value to the sum. Then we check if the sum is greater than the maxSum, if so we update the maxSum.
    kadane's algorithm O(n) time complexity.
    '''

    current = nums[0]
    maxSum = current

    for i in range(1, len(nums)):
        if nums[i] > current + nums[i]:
            current = nums[i]
        else:
            current += nums[i]
        
        if current > maxSum:
            maxSum = current
    
    return maxSum

# 3. Longest Substring Without Repeating Characters
'''
Given a string s, find the length of the longest without duplicate characters.
'''
def lengthOfLongestSubstring(self, s: str) -> int:
    '''
    The idea I had initially was to have a nested loop with a pointer at the start of the array.
    This will check all the combination of the subarrays and return the maximum sum. This is not the optimal solution since it takes O(n^2) time complexity.
    The solution here uses a dictionary to store the indices of the characters in the string. The algorithm iterates through the string and checks if the character is in the dictionary.
    If it is, we update the left pointer to the index of the character + 1, This is important because we already calculated the max so far. Then we update the character in the dictionary with the current index.
    Finally we calculate the max length by taking the difference between the right and left pointer, +1 to include the current character.
    '''
    cache = {}
    maxLen = 0
    left = 0

    for right in range(len(s)):
        if s[right] in cache and cache[s[right]] >= left:
            left = cache[s[right]] + 1
        
        cache[s[right]] = right

        maxLen = max(maxLen, right - left + 1)
    
    return maxLen


# 322. Coin Change

'''
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.
Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.
You may assume that you have an infinite number of each kind of coin.
'''


def coinChange(self, coins: List[int], amount: int) -> int:
    '''
    The main idea of this algorithm is to use dynamic programming to solve the problem. The algorithm iterates through the amount and checks the minimum number of coins needed to make up the amount.
    The algorithm uses a dp array to store the minimum number of coins needed to make up the amount. The algorithm iterates through the coins and checks the difference between the amount and the coin.
    If the difference is less than 0, we break the loop. If the difference is greater than 0, we update the minimum number of coins needed to make up the amount.
    Time complexity is O(n*m) where n is the amount and m is the number of coins.
    Space complexity is O(n) where n is the amount.
    '''

    coins.sort()
    dp = [0] * (amount + 1)

    for i in range(1,amount+1):
        minn = float('inf')

        for coin in coins:
            dif = i - coin
            if dif < 0:
                break
            minn = min(dp[dif] + 1, minn)

        dp[i] = minn
    
    if dp[amount] < float('inf'):
        return dp[amount]
    else:
            return -1

# 242. Valid Anagram
'''
Given two strings s and t, return true if t is an of s, and false otherwise.
'''

def isAnagram(self, s: str, t: str) -> bool:
    '''
    The idea of this algorithm is to use a dictionary to store the frequency of the characters in the string.
    The algorithm iterates through the string and checks if the character is in the dictionary. If it is not, we add the character to the dictionary.
    If it is, we increment the frequency of the character. Finally we check if the two dictionaries are equal.
    '''

    anagramS = {}
    anagramT = {}

    for i in s:
        if i not in anagramS:
            anagramS[i] = 1
        else:
            anagramS[i] += 1

    for i in t:
        if i not in anagramT:
            anagramT[i] = 1
        else:
            anagramT[i] += 1
    
    return anagramT == anagramS


# 49. Group Anagrams
'''
Given an array of strings strs, group the together. You can return the answer in any order.
'''
def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
    '''
    The idea I had is to use a dictionary to store the sorted string as the key and the original string as the value.
    The algorithm iterates through the strings and sorts the string. Then we check if the sorted string is in the dictionary.
    If it is not, we add the sorted string as the key and the original string as the value. If it is, we append the original string to the value.
    Finally we return the values of the dictionary.
    Time complexity is O(n*mlogm) where n is the number of strings and m is the length of the string.
    Space complexity is O(n) where n is the number of strings.
    '''
    
    cache = {}
    res = []

    for i in strs:
        temp = i
        i = str(sorted(i))
        if i not in cache:
            cache[i] = [temp]
        else:
            cache[i].append(temp)
    
    for i in cache.values():
        res.append(i)
    
    return res

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

# Linked Lists problems
    
# 206. Reverse Linked List

'''
Given the head of a singly linked list, reverse the list, and return the reversed list.
'''
def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        '''
        The optimal solution is to use a recursive function to reverse the linked list.

        The main idea is to return the newHead of the linked list. The algorithm checks if the head is None or the next element is None.
        If it is, we return the head. Otherwise, we call the function recursively with the next element of the head.
        Then we update the next element of the next element of the head to the head. Then we update the next element of the head to None.
        Finally we return the newHead.

        Time complexity is O(n) where n is the length of the linked list.
        Space complexity is O(n) where n is the length of the linked list.

        '''
        if head == None or head.next == None:
            return head
        
        newHead = self.reverseList(head.next)
        head.next.next = head
        head.next = None

        '''

        Illustration:
        1 -> 2 -> 3 -> 4 -> 5 -> None
        When you reach the end it will return the second last element.
        This element can be used to reverse the linked list.
        4.next = 5 and 4.next.next is the next element after the 5 which is None.
        4.next = None so we remove the link between 4 and 5 in the other direction.
        since are in the body of the recursive call and we done with the last call we now are going to deal with the previous call.
        3.next.next = 4 and 3.next = None
        then 
        2.next.next = 3 and 2.next = None
        then
        1.next.next = 2 and 1.next = None
        Then we return the newHead which is the last element of the linked list.
        5 -> 4 -> 3 -> 2 -> 1 -> None

        '''
        return newHead

# 21. Merge Two Sorted Lists

'''
You are given the heads of two sorted linked lists list1 and list2.
Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.
Return the head of the merged linked list.
'''

def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
    '''
    This algorithm is a recursive function to merge two sorted linked lists.
    If the first list is None, we return the second list. If the second list is None, we return the first list.
    If the value of the first list is greater than the value of the second list, we swap the lists.
    Then we call the function recursively with the next element of the first list and the second list.
    Finally we return the first list.
    '''
    if not list1 or not list2:
        return list1 if list1 else list2

    if list1.val > list2.val:
        list2, list1 = list1, list2

    list1.next = self.mergeTwoLists(list1.next, list2)

    '''
    The idea uses the following example:
    list1 = 1 -> 2 -> 4 -> None
    list2 = 1 -> 3 -> 4 -> None
    '''

    return list1

# 141. Linked List Cycle

'''
Given head, the head of a linked list, determine if the linked list has a cycle in it.
There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer.
Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.
Return true if there is a cycle in the linked list. Otherwise, return false.
'''

def hasCycle(self, head: Optional[ListNode]) -> bool:
    # the naive interpretation is to use an hash table and record every entry. If the new is already in means it is a cycle

    '''
    This algorithm uses the fast and slow pointer to check if the linked list has a cycle.
    The algorithm initializes the fast and slow pointers as the head of the linked list.
    The algorithm iterates through the linked list and checks if the fast pointer is None or the next element of the fast pointer is None.
    If it is, we return False. If it is not, we update the fast pointer to the next element of the next element of the fast pointer.
    Then we update the slow pointer to the next element of the slow pointer.
    If the fast pointer is equal to the slow pointer, we return True. If it is not, we return False.
    Time complexity is O(n) where n is the length of the linked list.
    Space complexity is O(1).
    '''
    fast = head
    slow = head

    while fast and fast.next:
        '''
        The update of the fast pointer is twice the speed of the slow pointer.
        This is particularly crucial because otherwise the cycle will never be detected.
        It has to be twice the speed of the slow pointer to avoid the slow pointer to catch up with the fast pointer.
        '''
        fast = fast.next.next
        slow = slow.next

        if fast == slow:
            return True
    
    return False

# 2. Add Two Numbers

'''
You are given two non-empty linked lists representing two non-negative integers.
The digits are stored in reverse order, and each of their nodes contains a single digit.
Add the two numbers and return the sum as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.
'''
def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
    '''
    The recursion is the first idea that came to my mind. This work by adding the two numbers and passing the carry to the next element.
    The algorithm checks if the two linked lists are None and the carry is 0. If it is, we return None.
    If it is not, we calculate the sum of the two numbers and the carry. Then we update the carry to the sum divided by 10.
    Then we create a new node with the value of the sum modulo 10. Then we update the next element of the new node to the recursive call of the next element of the first linked list and the next element of the second linked list.
    Finally we return the new node.
    Time complexity is O(n) where n is the length of the linked list.
    Space complexity is O(n) where n is the length of the linked list. --> Since we are using the stack to store the values.
    '''
    def addTwoNumbersHelper(l1,l2,carry):
        if not l1 and not l2 and carry == 0:
            return

        currSum = carry

        if l1:
            currSum += l1.val
            l1 = l1.next
        
        if l2:
            currSum += l2.val
            l2 = l2.next
        
        node = ListNode(currSum % 10)
        node.next = addTwoNumbersHelper(l1,l2,int(currSum / 10))
        return node

    return addTwoNumbersHelper(l1,l2,0)

# 19. Remove Nth Node From End of List

'''
Given the head of a linked list, remove the nth node from the end of the list and return its head.
'''
def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
    '''
    The idea of this algorithm is to use a recursive function to remove the nth node from the end of the linked list.
    The algorithm uses a helper function to remove the nth node from the end of the linked list. 
    The algorithm initializes the head of the linked list as the next element of the head.
    The algorithm iterates through the linked list and checks if the index is equal to n. If it is, we update the next element of the head to the next element of the next element of the head.
    Finally we return the head of the linked list.
    Time complexity is O(n) where n is the length of the linked list.
    Space complexity is O(n). --> Since we are using the stack to store the values.
    '''
    def removeHelper(node, n):
        if not node:
            return 0

        i = removeHelper(node.next,n)

        if i == n:
            node.next = node.next.next
        return i + 1
        
    return head.next if removeHelper(head,n) == n else head

# 234. Palindrome Linked List

'''
Given the head of a singly linked list, return true if it is a palindrome.
'''

def isPalindrome(self, head: Optional[ListNode]) -> bool:


    # This of course is not the optimal solution to the problem which should involve recursion.
    # array = []
    # end = -1
    # start = 0

    # while head:
    #     array.append(head.val)
    #     head = head.next
    #     end += 1

    # while start < end:
    #     if array[start] != array[end]:
    #         return False
    #     start += 1
    #     end -= 1
    
    # return True

    # Try to use recursion

    '''
    This algorithm uses a recursive function to check if the linked list is a palindrome.
    The algorithm uses a global variable to store the head of the linked list.
    The algorithm initializes the head of the linked list as the current head.
    The algorithm checks if the head is None. If it is, we return True.
    If it is not, we call the function recursively with the next element of the head.
    Then we check if the value of the head is equal to the value of the current head.
    If it is not, we return False. If it is, we update the current head to the next element of the current head.
    Finally we return True.
    Time complexity is O(n) where n is the length of the linked list.
    Space complexity is O(1)
    '''

    self.curHead = head

    def pal(head):
        if not head:
            return True
        
        ans = pal(head.next) and head.val == self.curHead.val
        self.curHead = self.curHead.next
        
        return ans

    return pal(head)
