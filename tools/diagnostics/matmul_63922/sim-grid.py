import json,os
from pathlib import Path
import numpy as np
import torch
import ttnn

torch.manual_seed(63291)
a = torch.randn((8,64,128),dtype=torch.bfloat16)
b = torch.randn((8,128,64),dtype=torch.bfloat16)
d = ttnn.open_device(device_id=0)
try:
 ta,tb=[ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=d) for x in (a,b)]
 c=ttnn.init_device_compute_kernel_config(d.arch(),math_fidelity=ttnn.MathFidelity.HiFi2,math_approx_mode=False,packer_l1_acc=True)
 y=ttnn.to_torch(ttnn.matmul(ta,tb,compute_kernel_config=c,core_grid=ttnn.CoreGrid(x=2,y=2))).contiguous()
 out=Path(os.environ['VALID_OUT']);np.save(out/'result.npy',y.view(torch.uint16).numpy(),allow_pickle=False)
 ref=(a.double()@b.double()).reshape(-1);v=y.double().reshape(-1);pcc=torch.corrcoef(torch.stack([v,ref]))[0,1].item()
 error=(v-ref).abs();result=dict(pcc=pcc,max_abs=error.max().item(),l2=float(torch.linalg.vector_norm(v-ref)),shape=list(y.shape));(out/'numerics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
finally:ttnn.close_device(d)
