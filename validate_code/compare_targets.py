import torch
import numpy


accl_target_u_0 = torch.load(
    "./sph_les/validate_code/validate_code_u=0.pt", map_location=torch.device("cpu")
)
accl_target_u_not_0 = torch.load(
    "./sph_les/validate_code/validate_code_u!=0.pt", map_location=torch.device("cpu")
)
accl_target_lb = numpy.load("./sph_les/validate_code/val_lb_target.npy")

torch.tensor(accl_target_lb, dtype=torch.float32)

print(accl_target_u_0)
print(accl_target_u_not_0)
print(accl_target_lb)

if torch.all(accl_target_u_0 == accl_target_lb):
    print("The LB are the same!")

if torch.all(accl_target_u_0 == accl_target_u_not_0):
    print("The targets are the same!")
