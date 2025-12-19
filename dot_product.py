inputs = [1,2,3,4,5,6]
weights = [0.89, 0.45, 0.34, 0.55, 0.78, 0.72]

dot_product = 0
for i, w in zip(inputs, weights):
    dot_product = dot_product + i * w
print(f'Dot Product of {inputs} and {weights}: ', dot_product)
