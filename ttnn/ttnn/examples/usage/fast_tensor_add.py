import ttnn
import torch

# Open device with id 0
device_id = 0
tt_device = ttnn.open_device(device_id = 0)

# Create torch tensors with element data type torch.float32
torch_tensor_a = torch.Tensor([[2, 3, 4], [5, 6, 7]]).to(dtype=torch.float32)
torch_tensor_b = torch.Tensor([[6, 7, 8], [3, 2, 1]]).to(dtype=torch.float32)

# Convert torch tensors to ttnn tensors
# Note: that the resultant ttnn tensors are still in the host's memory
# Note: binary element wise operations require the tensor operands to be in TILE_LAYOUT layout
tensor_a = ttnn.from_torch(torch_tensor_a, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
tensor_b = ttnn.from_torch(torch_tensor_b, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)

print('Torch: ')
print('\ttensor_a: ' + str(torch_tensor_a))
print('\ttensor_b: ' + str(torch_tensor_b))

# Now copy the ttnn tensors to tt device's L1 memory
tensor_device_a = ttnn.to_device(tensor_a, tt_device, memory_config=ttnn.L1_MEMORY_CONFIG)
tensor_device_b = ttnn.to_device(tensor_b, tt_device, memory_config=ttnn.L1_MEMORY_CONFIG)


print('ttnn: ')
print('\ttensor_a: ' + str(tensor_a))
print('\ttensor_b: ' + str(tensor_b))
print('\ttensor_device_a: ' + str(tensor_device_a))
print('\ttensor_device_b: ' + str(tensor_device_b))

# Perform addition
tensor_result_1 = ttnn.add(tensor_device_a, tensor_device_b)

tensor_result_2 = torch.Tensor.add(torch_tensor_a, torch_tensor_b)


print('torch addition: ' + str(tensor_result_2))
print('ttnn addition: ' + str(tensor_result_1))

# Fast Tensor Add Example
fta_tensor_result = ttnn.fast_tensor_add(tensor_device_a, tensor_device_b)

print('fta_tensor_result: ' + str(fta_tensor_result))

# Close the tt device which we have opened
ttnn.close_device(tt_device)