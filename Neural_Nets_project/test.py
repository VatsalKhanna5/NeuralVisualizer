import numpy as np

# Without a seed
print("First run without seed:")
print(np.random.rand(3))
print(np.random.rand(3))

print("\nSecond run without seed:")
print(np.random.rand(3))
print(np.random.rand(3))

# With a seed
seed_value = 42
np.random.seed(seed_value)
print("\nFirst run with seed 42:")
print(np.random.rand(3))
print(np.random.rand(3))

np.random.seed(seed_value) # Reset the seed
print("\nSecond run with seed 42:")
print(np.random.rand(3))
print(np.random.rand(3))
