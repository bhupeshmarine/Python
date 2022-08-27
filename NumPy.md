# NumPy Essential

### Introduction to NumPy


```python
print("Hello NumPy")
```

    Hello NumPy



```python
2+8*5
```




    42




```python
import numpy as np # Importing NumPy
```


```python
integers = np.array([10,20,30,40,50])
```


```python
print(integers)
```

    [10 20 30 40 50]



```python
integers[0]
```




    10




```python
integers[0]=20
integers
```




    array([20, 20, 30, 40, 50])




```python
integers.dtype # Using .dtype to find the type of array like string, integer etc. 
```




    dtype('int64')




```python
smallerIntegers=np.array(integers, dtype=np.int8)
smallerIntegers
```




    array([20, 20, 30, 40, 50], dtype=int8)




```python
integers.nbytes
```




    40




```python
smallerIntegers.nbytes
```




    5



# NumPy Array Types and Creating NumPy Array

### Multidimenssional Array


```python
nums = np.array([[1,2,3,4,5],[6,7,8,9,10]])
nums
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10]])




```python
nums[0]
```




    array([1, 2, 3, 4, 5])




```python
nums[0,0]
```




    1




```python
nums[1,0]
```




    6




```python
nums.ndim
```




    2




```python
multi_arr=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,13]]])
multi_arr
```




    array([[[ 1,  2,  3],
            [ 4,  5,  6]],
    
           [[ 7,  8,  9],
            [10, 11, 13]]])




```python
multi_arr[0]
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
multi_arr[0,1,1]
```




    5



### Creating arrays from List and other Python structure


```python

```


```python
first_list=[1,2,3,4,5,6,7,8,9,10]
```


```python
first_list
```




    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]




```python
first_array=np.array(first_list)
```


```python
first_array
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10])



### Numpy array converts all the element to most common type of element


```python
second_list=[1,2,3,-1.23,50,128000.56,4.56]
second_array=np.array(second_list)
second_array
```




    array([ 1.0000000e+00,  2.0000000e+00,  3.0000000e+00, -1.2300000e+00,
            5.0000000e+01,  1.2800056e+05,  4.5600000e+00])




```python
second_array.dtype
```




    dtype('float64')




```python
third_list=['Ann',1111,'Peter',11112,'Susan',11114]
third_list
```




    ['Ann', 1111, 'Peter', 11112, 'Susan', 11114]




```python
third_array=np.array(third_list)
third_array
```




    array(['Ann', '1111', 'Peter', '11112', 'Susan', '11114'], dtype='<U21')



### Numpy Arange


```python
integers_array=np.arange(10) #Array of integers starting from 0
integers_array
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
integers_second_array=np.arange(100,300) # gives arrray of integers between 100 to 300
integers_second_array
```




    array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
           113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
           126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138,
           139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
           152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
           165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177,
           178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
           191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203,
           204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,
           217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
           230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
           243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
           256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268,
           269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
           282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294,
           295, 296, 297, 298, 299])




```python
integers_third_array=np.arange(100,151,2)
integers_third_array
```




    array([100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124,
           126, 128, 130, 132, 134, 136, 138, 140, 142, 144, 146, 148, 150])



### Linspace in Numpy


```python
first_float_arr=np.linspace(10,20) #By default it gives 50 values
first_float_arr
```




    array([10.        , 10.20408163, 10.40816327, 10.6122449 , 10.81632653,
           11.02040816, 11.2244898 , 11.42857143, 11.63265306, 11.83673469,
           12.04081633, 12.24489796, 12.44897959, 12.65306122, 12.85714286,
           13.06122449, 13.26530612, 13.46938776, 13.67346939, 13.87755102,
           14.08163265, 14.28571429, 14.48979592, 14.69387755, 14.89795918,
           15.10204082, 15.30612245, 15.51020408, 15.71428571, 15.91836735,
           16.12244898, 16.32653061, 16.53061224, 16.73469388, 16.93877551,
           17.14285714, 17.34693878, 17.55102041, 17.75510204, 17.95918367,
           18.16326531, 18.36734694, 18.57142857, 18.7755102 , 18.97959184,
           19.18367347, 19.3877551 , 19.59183673, 19.79591837, 20.        ])




```python
second_float_arr=np.linspace(10,20,30) #Linspace, the third element tells no of element in the array, However
                                    #In arange()third element produce a nubers with that difference
second_float_arr
```




    array([10.        , 10.34482759, 10.68965517, 11.03448276, 11.37931034,
           11.72413793, 12.06896552, 12.4137931 , 12.75862069, 13.10344828,
           13.44827586, 13.79310345, 14.13793103, 14.48275862, 14.82758621,
           15.17241379, 15.51724138, 15.86206897, 16.20689655, 16.55172414,
           16.89655172, 17.24137931, 17.5862069 , 17.93103448, 18.27586207,
           18.62068966, 18.96551724, 19.31034483, 19.65517241, 20.        ])




```python
first_rand_arr = np.random.rand(40) #single dimension array of random number
first_rand_arr
```




    array([0.30877296, 0.23814092, 0.96523755, 0.01550433, 0.4872231 ,
           0.82359345, 0.97604711, 0.87395524, 0.36314777, 0.17891277,
           0.29218157, 0.35144253, 0.57133763, 0.81026835, 0.83568915,
           0.46345512, 0.53936955, 0.73652826, 0.09286166, 0.92710118,
           0.76972591, 0.27603292, 0.16914659, 0.44166385, 0.29817509,
           0.12100934, 0.72890507, 0.41003627, 0.55106773, 0.96843619,
           0.07433568, 0.08419971, 0.66163481, 0.12361991, 0.57391881,
           0.47255427, 0.74976043, 0.18256984, 0.6116124 , 0.79069786])




```python
second_rand_arr=np.random.rand(2,4) # two dimensional array of random numbers
second_rand_arr
```




    array([[0.1672827 , 0.85006854, 0.31603429, 0.21183894],
           [0.31121014, 0.06200682, 0.42324381, 0.16964573]])




```python
third_rand_arr=np.random.randint(0,100,20)
third_rand_arr
```




    array([70, 57, 53, 76,  3, 76, 14, 29, 13, 87, 13, 68, 35, 95, 53, 69, 22,
           81, 10, 99])



### Creating arrays filled with constant values


```python
first_z_arr=np.zeros(5) # Array of zeros
first_z_arr
```




    array([0., 0., 0., 0., 0.])




```python
second_z_arr=np.zeros((14,9))
second_z_arr

```




    array([[0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0.]])




```python
third_ones_arr=np.ones(6)
third_ones_arr
```




    array([1., 1., 1., 1., 1., 1.])




```python
second_ones_arr=np.ones((19,8))
second_ones_arr
```




    array([[1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1.]])




```python
third_ones_arr=np.ones((7,8),dtype=int)
third_ones_arr
```




    array([[1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]])




```python
first_fill_arr=np.empty(10,dtype=int)
first_fill_arr.fill(12)
first_fill_arr
```




    array([12, 12, 12, 12, 12, 12, 12, 12, 12, 12])




```python
first_full_arr=np.full(5,10) # One dimensional array with full function
first_full_arr
```




    array([10, 10, 10, 10, 10])




```python
second_full_arr=np.full((4,7),10)
second_full_arr
```




    array([[10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10],
           [10, 10, 10, 10, 10, 10, 10]])



### Finding shape and size of an array


```python
first_arr=np.arange(20)
first_arr
```




    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19])




```python
second_arr=np.linspace((1,2),(10,20),10) # will produce 6 vectros smallest is [1,2] and largest is [10,20]
second_arr
```




    array([[ 1.,  2.],
           [ 2.,  4.],
           [ 3.,  6.],
           [ 4.,  8.],
           [ 5., 10.],
           [ 6., 12.],
           [ 7., 14.],
           [ 8., 16.],
           [ 9., 18.],
           [10., 20.]])




```python
third_arr=np.full((2,3,4),10)
third_arr
```




    array([[[10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10]],
    
           [[10, 10, 10, 10],
            [10, 10, 10, 10],
            [10, 10, 10, 10]]])




```python
np.shape(first_arr)
```




    (20,)




```python
np.shape(second_arr)
```




    (10, 2)




```python
np.shape(third_arr)
```




    (2, 3, 4)




```python
np.size(first_arr)
```




    20




```python
np.size(second_arr)
```




    20




```python
np.size(third_arr)
```




    24



# Manipulation of NumPy Array

### Adding, removing and sorting elements


```python
first_arr=np.array([1,2,3,5])
first_arr
```




    array([1, 2, 3, 5])




```python
new_first_arr=np.insert(first_arr,3,4) # It will insert 4 at position 3 
new_first_arr
```




    array([1, 2, 3, 4, 5])




```python
second_arr=np.array([1,2,3,4])
second_arr
```




    array([1, 2, 3, 4])




```python
new_second_arr=np.append(second_arr,5)
new_second_arr
```




    array([1, 2, 3, 4, 5])




```python
third_arr=np.array([1,2,3,4,5])
third_arr
```




    array([1, 2, 3, 4, 5])




```python
del_arr=np.delete(third_arr,4) #It will delete the value at index 4
del_arr
```




    array([1, 2, 3, 4])




```python
integer_arr=np.random.randint(0,50,20)
integer_arr
```




    array([19,  0, 47, 17, 44, 27, 34, 31,  2, 31, 17, 20, 49, 30, 46, 32,  1,
           17, 20, 17])




```python
print(np.sort(integer_arr))
```

    [ 0  1  2 17 17 17 17 19 20 20 27 30 31 31 32 34 44 46 47 49]



```python
integer_2dim_arr=np.array([[3,2,5,7,4],[12,45,23,78,65]])
integer_2dim_arr
```




    array([[ 3,  2,  5,  7,  4],
           [12, 45, 23, 78, 65]])




```python
print(np.sort(integer_2dim_arr))
```

    [[ 2  3  4  5  7]
     [12 23 45 65 78]]



```python
colors=np.array(['orange','green','blue','purple','white','black'])
colors
```




    array(['orange', 'green', 'blue', 'purple', 'white', 'black'], dtype='<U6')




```python
print(np.sort(colors))
```

    ['black' 'blue' 'green' 'orange' 'purple' 'white']


### Coppies and View


```python
students_ids=np.array([111,2222,3333,4444,5555,6666,7777,8888])
students_ids
```




    array([ 111, 2222, 3333, 4444, 5555, 6666, 7777, 8888])




```python
students_ids2=students_ids # Assignment doesnot change the id it points to same memory
print('id of students_ids',id(students_ids))
print('id of students_ids2', id(students_ids2))
```

    id of students_ids 140178495124464
    id of students_ids2 140178495124464



```python
students_ids[1]=1212
print(students_ids)
print(students_ids2)
```

    [ 111 1212 3333 4444 5555 6666 7777 8888]
    [ 111 1212 3333 4444 5555 6666 7777 8888]



```python
students_ids_copy=students_ids.copy()
print(students_ids_copy)
```

    [ 111 1212 3333 4444 5555 6666 7777 8888]



```python
print('id of students_ids',id(students_ids))
print('id of students_ids2', id(students_ids_copy)) # copy() function will create a new memory location, 
                                                    # id will change

```

    id of students_ids 140178495124464
    id of students_ids2 140178741362288



```python
students_ids[2]=1313
print('origional :', students_ids)
print('copy :', students_ids_copy)
```

    origional : [ 111 1212 1313 4444 5555 6666 7777 8888]
    copy : [ 111 1212 3333 4444 5555 6666 7777 8888]


### Reshape in NumPy



```python
first_arr=np.arange(1,13)
first_arr
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])




```python
second_arr=np.reshape(first_arr,(3,4))
second_arr
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
third_arr=np.reshape(first_arr,(6,2))
third_arr
```




    array([[ 1,  2],
           [ 3,  4],
           [ 5,  6],
           [ 7,  8],
           [ 9, 10],
           [11, 12]])




```python
fourth_arr=np.reshape(first_arr,(3,2,2))
fourth_arr
print('Dimenssion of fourth array is', fourth_arr.ndim)
```

    Dimenssion of fourth array is 3



```python
fifth_arr=np.array([[1,2],[3,4],[5,6]]) # creating a 3x2 matrix
fifth_arr
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
sixth_arr_flat=np.reshape(fifth_arr,-1)
sixth_arr_flat
```




    array([1, 2, 3, 4, 5, 6])



### Indexing and Slicing in NumPy


```python
two_dim_arr=np.reshape(np.arange(12),(3,4))
two_dim_arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
threedim_arr=np.reshape(np.arange(3*4*5),(3,4,5))
threedim_arr
```




    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    
           [[20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39]],
    
           [[40, 41, 42, 43, 44],
            [45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54],
            [55, 56, 57, 58, 59]]])




```python
threedim_arr[2,-1,-1]
```




    59




```python
# Creating one dimensional array
onedim_arr=np.arange(10)
onedim_arr
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
#for first 3 elements
onedim_arr[:3]
```




    array([0, 1, 2])




```python
# for last 3 elements
onedim_arr[-3:]
```




    array([7, 8, 9])




```python
#for every other element
onedim_arr[::2]
```




    array([0, 2, 4, 6, 8])




```python
# Creating two dimenssional array
twodim_arr=np.reshape(np.arange(12),(3,4))
twodim_arr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
#slicing from two dimensional array
twodim_arr[1:,1:]   # [1: - all rows from 1st idex, 1: all columns from index 1]

```




    array([[ 5,  6,  7],
           [ 9, 10, 11]])




```python
# slice after fist row and include all columns
twodim_arr[1:,]
```




    array([[ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
# Slice and give first row
twodim_arr[1,:]
```




    array([4, 5, 6, 7])




```python
# Slice to give 3rd column
twodim_arr[:,2]
```




    array([ 2,  6, 10])



### Function and Joining Array


```python
first_arr= np.arange(1,11)
second_arr=np.arange(11,21)
print('first_arr :', first_arr)
print('second_arr :', second_arr)
```

    first_arr : [ 1  2  3  4  5  6  7  8  9 10]
    second_arr : [11 12 13 14 15 16 17 18 19 20]



```python
con_arr=np.concatenate((first_arr,second_arr))
con_arr
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20])




```python
third_2darr=np.array([[1,2,3,4,5],[6,7,8,9,10]])
third_2darr
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10]])




```python
fourth_2darr=np.array([[11,12,13,14,15],[16,17,18,19,20]])
fourth_2darr
```




    array([[11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20]])




```python
con2d_arr=np.concatenate((third_2darr,fourth_2darr)) # vertical concatenation
con2d_arr
```




    array([[ 1,  2,  3,  4,  5],
           [ 6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15],
           [16, 17, 18, 19, 20]])




```python
con2d_arr.ndim
```




    2




```python
con2d_arr2=np.concatenate((third_2darr,fourth_2darr), axis=1) # horizontal concatenationn
con2d_arr2
```




    array([[ 1,  2,  3,  4,  5, 11, 12, 13, 14, 15],
           [ 6,  7,  8,  9, 10, 16, 17, 18, 19, 20]])



### vstack and hstack


```python
horstack=np.hstack((first_arr,second_arr))
horstack
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20])




```python
ver_stack=np.vstack((first_arr, second_arr))
ver_stack
```




    array([[ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])



# Functions of NumPy


```python

```

### split, array_split, hsplit, vsplit


```python
#It will split only when the no. elements is divisible by no.of split >> resulting array must have same shape

fifth_arr=np.arange(1,13)
fifth_arr
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])




```python
splt_arr=np.array_split(fifth_arr,4)
splt_arr
```




    [array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9]), array([10, 11, 12])]




```python
print(splt_arr[1])
```

    [4 5 6]



```python
sp2_arr=np.array_split(fifth_arr,8)
sp2_arr
```




    [array([1, 2]),
     array([3, 4]),
     array([5, 6]),
     array([7, 8]),
     array([9]),
     array([10]),
     array([11]),
     array([12])]




```python
# Spliting 2-d array horizontally
hsp_arr=np.hsplit(third_2darr,5)
hsp_arr
```




    [array([[1],
            [6]]),
     array([[2],
            [7]]),
     array([[3],
            [8]]),
     array([[4],
            [9]]),
     array([[ 5],
            [10]])]




```python
# Vertical Spliting 2-d array

vsp_arr=np.vsplit(third_2darr, 2)
vsp_arr
```




    [array([[1, 2, 3, 4, 5]]), array([[ 6,  7,  8,  9, 10]])]



# Functions and Operations

### Arithmetic Operations and functions


```python
a=np.arange(1,11)
b=np.arange(21,31)
print('a :',a)
print('b :',b)
```

    a : [ 1  2  3  4  5  6  7  8  9 10]
    b : [21 22 23 24 25 26 27 28 29 30]



```python
a+b # addition is element by element operation
```




    array([22, 24, 26, 28, 30, 32, 34, 36, 38, 40])




```python
b-a
```




    array([20, 20, 20, 20, 20, 20, 20, 20, 20, 20])




```python
a*b
```




    array([ 21,  44,  69,  96, 125, 156, 189, 224, 261, 300])




```python
b/a
```




    array([21.        , 11.        ,  7.66666667,  6.        ,  5.        ,
            4.33333333,  3.85714286,  3.5       ,  3.22222222,  3.        ])




```python
c=np.arange(2,12)
print('c :',c)
```

    c : [ 2  3  4  5  6  7  8  9 10 11]



```python
a**c
```




    array([           1,            8,           81,         1024,
                  15625,       279936,      5764801,    134217728,
             3486784401, 100000000000])




```python
a*2 # Each element will be multiplied by 2
```




    array([ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20])




```python
np.add(a,b)
```




    array([22, 24, 26, 28, 30, 32, 34, 36, 38, 40])




```python
np.subtract(a,b)
```




    array([-20, -20, -20, -20, -20, -20, -20, -20, -20, -20])




```python
np.multiply(a,b)
```




    array([ 21,  44,  69,  96, 125, 156, 189, 224, 261, 300])




```python
np.divide(b,a)
```




    array([21.        , 11.        ,  7.66666667,  6.        ,  5.        ,
            4.33333333,  3.85714286,  3.5       ,  3.22222222,  3.        ])




```python
np.mod(b,a) # mod function gives remainder element by element operarion
```




    array([0, 0, 2, 0, 0, 2, 6, 4, 2, 0])




```python
np.power(b,a)
```




    array([             21,             484,           12167,          331776,
                   9765625,       308915776,     10460353203,    377801998336,
            14507145975869, 590490000000000])




```python
np.sqrt(a)
```




    array([1.        , 1.41421356, 1.73205081, 2.        , 2.23606798,
           2.44948974, 2.64575131, 2.82842712, 3.        , 3.16227766])



### Broadcasting


```python
import numpy as np
a=np.arange(1,10).reshape(3,3)
a
```




    array([[1, 2, 3],
           [4, 5, 6],
           [7, 8, 9]])




```python
b=np.arange(1,4)
b
```




    array([1, 2, 3])




```python
a+b # array a has dimension 3x3 and array b has dimenssion 1x3 here last dimenssion for both array is 3
```




    array([[ 2,  4,  6],
           [ 5,  7,  9],
           [ 8, 10, 12]])




```python
c=np.arange(1,3)
c
```




    array([1, 2])




```python
# a+c is not prossible as array a has dimenssion 3x3, array c has dimesnssion 1x2
```


```python
d=np.arange(24).reshape(2,3,4)
d
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
e=np.arange(4)
e
```




    array([0, 1, 2, 3])




```python
d-e
```




    array([[[ 0,  0,  0,  0],
            [ 4,  4,  4,  4],
            [ 8,  8,  8,  8]],
    
           [[12, 12, 12, 12],
            [16, 16, 16, 16],
            [20, 20, 20, 20]]])



### Agggregate fucntion


```python
first_arr=np.arange(10,110,10)
first_arr
```




    array([ 10,  20,  30,  40,  50,  60,  70,  80,  90, 100])




```python
second_arr=np.arange(10,100,10).reshape(3,3)
second_arr
```




    array([[10, 20, 30],
           [40, 50, 60],
           [70, 80, 90]])




```python
third_arr=np.arange(10,110,10).reshape(2,5)
third_arr
```




    array([[ 10,  20,  30,  40,  50],
           [ 60,  70,  80,  90, 100]])




```python
first_arr.sum()
```




    550




```python
second_arr.sum()
```




    450




```python
third_arr.sum()
```




    550




```python
second_arr.sum(axis=0)
```




    array([120, 150, 180])




```python
second_arr.sum(axis=1)
```




    array([ 60, 150, 240])




```python
third_arr.prod()
```




    36288000000000000




```python
third_arr.prod(axis=0)
```




    array([ 600, 1400, 2400, 3600, 5000])




```python
third_arr.prod(axis=1)
```




    array([  12000000, 3024000000])




```python
np.average(first_arr)
```




    55.0




```python
np.min(first_arr)
```




    10




```python
np.max(first_arr)
```




    100




```python
np.median(first_arr)
```




    55.0




```python
np.mean(first_arr)
```




    55.0




```python
np.std(first_arr)
```




    28.722813232690143



### How to get unique items and count


```python
first_arr=np.array([1,2,3,4,5,6,1,2,7,10,7,8])
```


```python
np.unique(first_arr)
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8, 10])




```python
second_arr=np.array([[1,1,2,1], [3,1,2,1], [1,1,2,1], [7,1,1,1]])
second_arr
```




    array([[1, 1, 2, 1],
           [3, 1, 2, 1],
           [1, 1, 2, 1],
           [7, 1, 1, 1]])




```python
np.unique(second_arr)
```




    array([1, 2, 3, 7])




```python
np.unique(second_arr, axis=0) # third row is not printed because it is duplicate of first 1st row.
```




    array([[1, 1, 2, 1],
           [3, 1, 2, 1],
           [7, 1, 1, 1]])




```python
np.unique(second_arr, axis=1) # 4th column is not printed because it is duplicate of second column
```




    array([[1, 1, 2],
           [1, 3, 2],
           [1, 1, 2],
           [1, 7, 1]])




```python
# For returning index no of elements we use function return_index
np.unique(first_arr, return_index=True)
```




    (array([ 1,  2,  3,  4,  5,  6,  7,  8, 10]),
     array([ 0,  1,  2,  3,  4,  5,  8, 11,  9]))




```python
# Returing the freuqency of unique elemnts 
np.unique(second_arr, return_counts=True)
```




    (array([1, 2, 3, 7]), array([11,  3,  1,  1]))



### Transpose lilke operation


```python
# it changes the row to column and vice versa
first_2dimarr=np.arange(12).reshape(3,4)
first_2dimarr
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
y=np.transpose(first_2dimarr)
y
```




    array([[ 0,  4,  8],
           [ 1,  5,  9],
           [ 2,  6, 10],
           [ 3,  7, 11]])




```python
y.shape
```




    (4, 3)




```python
second_2dimarr=np.arange(6).reshape(3,2)
second_2dimarr
```




    array([[0, 1],
           [2, 3],
           [4, 5]])




```python
np.transpose(second_2dimarr)
```




    array([[0, 2, 4],
           [1, 3, 5]])




```python
# Move axis funcion
```


```python
first_3dimarr=np.arange(2*3*4).reshape(2,3,4)
first_3dimarr
```




    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])




```python
np.moveaxis(first_3dimarr,0,-1) # shape 2,3,4 will become 3,4,2 (2 having idex 0 initially will take index -1)
```




    array([[[ 0, 12],
            [ 1, 13],
            [ 2, 14],
            [ 3, 15]],
    
           [[ 4, 16],
            [ 5, 17],
            [ 6, 18],
            [ 7, 19]],
    
           [[ 8, 20],
            [ 9, 21],
            [10, 22],
            [11, 23]]])




```python
# Swap axis interchanges two axis of an array
np.swapaxes(first_3dimarr,0,2)
```




    array([[[ 0, 12],
            [ 4, 16],
            [ 8, 20]],
    
           [[ 1, 13],
            [ 5, 17],
            [ 9, 21]],
    
           [[ 2, 14],
            [ 6, 18],
            [10, 22]],
    
           [[ 3, 15],
            [ 7, 19],
            [11, 23]]])



### Reversing an array


```python
arr_1dim=[10,1,9,2,8,3,7,4,6,5]
arr_1dim
```




    [10, 1, 9, 2, 8, 3, 7, 4, 6, 5]




```python
arr_1dim[::-1]
```




    [5, 6, 4, 7, 3, 8, 2, 9, 1, 10]




```python
np.flip(arr_1dim)
```




    array([ 5,  6,  4,  7,  3,  8,  2,  9,  1, 10])




```python
arr_2dim=np.arange(9).reshape(3,3)
arr_2dim
```




    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])




```python
np.flip(arr_2dim)
```




    array([[8, 7, 6],
           [5, 4, 3],
           [2, 1, 0]])


