# Linear Data Structures

# Introduction

Linear Data Structures(LDS): 

- are ADTs that refer to sequential collections
- store items in a certain sequence so that any given item’s relative location will be known and remembered
- items are thus ordered in the way they were added

Examples of LDS:

| ADT | Insert/Deletion | Example |
| --- | --- | --- |
| Stacks | LIFO → Add and delete items only at one side | A stack of dirty dishes that need to be washed |
| Queues | FIFO → Add items at one side, delete from the other side | A queue of customers at the grocery |
| Deques | Add or delete items from either side | Building a tower with lego bricks by adding/removing bricks from either end |
| List | Add or delete items anywhere in the list | Announcement of winners in a lucky draw contest in alphabetical order |

# OOP In Python

- Python is a fully OOP language, containing **objects** and **classes**
    - Classes are a way of combining properties(variables) together with their associated behaviour(functions)
    - Classes are made when multiple objects share common sets of variables and functions

# Encapsulation

- Effective form of information hiding where the data of an object can only be accessed indirectly through access functions(getters and setters)
- this means the class properties are considered private, and the class behaviour public
- promote a class variable to private by adding two underscore characters in front of the name (e.g. self.__name)
- attempting to access a private class variable outside the class will cause an attribute error
- add getter and setter methods for accessing a private variable outside the class

# Inheritance

![image.png](image.png)

![image.png](image%201.png)

![image.png](image%202.png)

# Abstract Classes

- Is a class that you cannot(or should not) instantiate objects from
- purpose is to implement common properties and behaviour of future child classes
- `def sample(): raise NotImplementedError('Subclass must implement abstract method')`
- all classes in python will be derived from parent abstract class *Object*

## str

Default implementation of `__str__` is to return the memory address of the object when you print the object using print()

override it to return something else