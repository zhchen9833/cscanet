def blelloch_scan(arr):
    arr = list(arr)
    n = len(arr)

    print("Initial array:", arr)

    step = 1
    print("\nUpsweep Phase:")
    while step < n:
        for i in range(0, n, 2 * step):
            if i + step < n:
                arr[i + 2 * step - 1] += arr[i + step - 1]
        print(f"After step size {step}, array: {arr}")
        step *= 2

    arr[-1] = 0
    print("\nSetting root to 0, array:", arr)

    step = n // 2
    print("\nDownsweep Phase:")
    while step > 0:
        for i in range(0, n, 2 * step):
            if i + step < n:
                temp = arr[i + step - 1]
                arr[i + step - 1] = arr[i + 2 * step - 1]
                arr[i + 2 * step - 1] += temp
        print(f"After step size {step}, array: {arr}")
        step //= 2

    return arr

input_array = [1,2,0,1,2,0,1,2]
arr = blelloch_scan(input_array)
prefix_sum_array = [input_array[i] + arr[i] for i in range(len(input_array))]

print(f"\nFinal PrefixSum array: {prefix_sum_array}")