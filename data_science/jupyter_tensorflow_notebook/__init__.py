def reverse(number):
    arr = []
    for char in str(number):
        arr.append(char)

    arr = reversed(arr)
    return  int("".join(arr))


print(reverse(199))

