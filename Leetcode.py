from ast import List

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