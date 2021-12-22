# use to draw out the input to see if it match the given result

import mnist_loader

print("Input the index of training input to test, as an integer")
input_index = int(input())

c = 0
for i in mnist_loader.load_data_wrapper()[0]:
    if c == input_index:
        for j in range(0,784,28):
            for k in range(j,j+28):
                if i[0][k] > 0.7:
                    print(1, end="")
                else:
                    print(" ", end="")
            print("\n", end="")

        print("The expected result is: ", end="")
        for h in range(0,10):
            if i[1][h] == 1:
                print(h);
                break
            
        break
    c += 1