####################################### LAMBDA #######################################
# 1. Write a Python program to create a lambda function that adds 15 to a given number passed in as an argument, also create a lambda function that multiplies argument x with argument y and print the result. Go to the editor
# Sample Output:
# 25
# 48
from curses.ascii import isdigit
from dataclasses import replace


add15 = lambda x: x+15
multiply_xy = lambda x,y: x*y

print(add15(10))
print(multiply_xy(6,8))

# 2. Write a Python program to create a function that takes one argument, and that argument will be multiplied with an unknown given number. Go to the editor
# Sample Output:
# Double the number of 15 = 30
# Triple the number of 15 = 45
# Quadruple the number of 15 = 60
# Quintuple the number 15 = 75
def func_compute(n):
 return lambda x,z : (x+z) * n
result = func_compute(3)
print("Double the number of 15 =", result(1,6))

# 6. Write a Python program to square and cube every number in a given list of integers using Lambda. Go to the editor
# Original list of integers:
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Square every number of the said list:
# [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
# Cube every number of the said list:
# [1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]

l = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

sq = list(map(lambda x: x**3,l))

print(sq)

# 7. Write a Python program to find if a given string starts with a given character using Lambda. Go to the editor
# Sample Output:
# True
# False

s = "mystring"

f = lambda t,c: True if t[0] == c else False
f_same = lambda t,c: True if t.startswith(c) else False
f(s,'m')
f_same(s,'m')

# 9. Write a Python program to check whether a given string is number or not using Lambda. Go to the editor
# Sample Output:
# True
# True
# False
# True
# False
# True
# Print checking numbers:
# True
# True

check_if_numb = lambda n: True if (n.startswith('-') and n[1:].isdigit()) or n.isdigit() == 'int' else False  
check_if_numb('-778')



###################################### DATETIME #####################################################
import time
import datetime as dt

current = dt.datetime.now()

print(current)
type(current)


############################## PANDAS ###########################################


############################## NUMPY ############################################



############################## SKLEARN ##########################################




############################## CHALLENGE ########################################
# 1. Write a Python program to check if a given positive integer is a power of two.
import numpy as np

pow2 = lambda x: True if (np.log(x) / np.log(2)).is_integer() else False

pow3 = lambda x: True if (np.log(x) / np.log(3)).is_integer() else False


pow2(36)
pow3(21)
#alternative
def is_Power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0
def is_Power_of_three(n):
    while (n % 3 == 0):
         n /= 3;         
    return n == 1

# 4. Write a Python program to check if a number is a perfect square. Go to the editor
def is_perfect_square(n):
    x = n // 2
    y = set([x])
    while x * x != n:
        x = (x + (n // x)) // 2
        if x in y: return False
        y.add(x)
    return True

#  14. Write a Python program to find the single element appears once in a list where every element appears four times except for one. 
s = set[2,5,69,777]

s