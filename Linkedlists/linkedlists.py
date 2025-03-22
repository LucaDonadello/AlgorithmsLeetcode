from typing import Optional

# Linked Lists problems

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
    
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