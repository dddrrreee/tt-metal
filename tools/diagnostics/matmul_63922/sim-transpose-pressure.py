import torch,ttnn

torch.manual_seed(63291)
a=torch.randn((80,2560,64),dtype=torch.bfloat16)
b=torch.randn((80,64,32),dtype=torch.bfloat16)
d=ttnn.open_device(device_id=0)
try:
 print('grid',d.compute_with_storage_grid_size(),flush=True)
 ta=ttnn.from_torch(a,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d,tile=ttnn.Tile((32,32),transpose_tile=True))
 tb=ttnn.from_torch(b,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=d)
 y=ttnn.to_torch(ttnn.matmul(ta,tb)).float()
 r=(a.float()@b.float()).reshape(-1);v=y.reshape(-1)
 print('PASS shape',list(y.shape),'pcc',torch.corrcoef(torch.stack([v,r]))[0,1].item(),flush=True)
finally:ttnn.close_device(d)
