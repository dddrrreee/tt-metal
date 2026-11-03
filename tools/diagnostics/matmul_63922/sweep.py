"""Default matmul kernel-time and exact-output capture. Device job only."""
import argparse,gc,hashlib,json,math,os,statistics,time,traceback
from pathlib import Path
import numpy as np
import torch
import ttnn
r=Path(__file__).resolve().parent;out=Path(os.environ['VALID_OUT']);side=os.environ['VALID_SIDE']
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def tensor_bytes(x):return x.contiguous().view(torch.uint8).numpy().tobytes()
def digest(x):return hashlib.sha256(tensor_bytes(x)).hexdigest()
def finite_number(x):return float(x) if math.isfinite(float(x)) else str(float(x))
def metrics(actual,ref):
 x=actual.double().reshape(-1);y=ref.double().reshape(-1);err=x-y;good=torch.isfinite(x)&torch.isfinite(y)
 if bool(good.all()):
  xc=x-x.mean();yc=y-y.mean();den=torch.linalg.vector_norm(xc)*torch.linalg.vector_norm(yc);pcc=float(xc.dot(yc)/den) if den else (1.0 if torch.equal(x,y) else 0.0)
  l2=float(torch.linalg.vector_norm(err));rl=l2/max(float(torch.linalg.vector_norm(y)),1e-300)
 else:pcc=l2=rl=float('nan')
 return dict(pcc=finite_number(pcc),max_abs=finite_number(err.abs().max()),l2=finite_number(l2),relative_l2=finite_number(rl),nan=int(torch.isnan(x).sum()),inf=int(torch.isinf(x).sum()),ref_nan=int(torch.isnan(y).sum()),ref_inf=int(torch.isinf(y).sum()))
def memory(name,shape):
 if name in ['DRAM','L1']:return getattr(ttnn,name+'_MEMORY_CONFIG')
 return ttnn.create_sharded_memory_config(shape=(math.prod(shape[:-1]),shape[-1]),core_grid=ttnn.CoreGrid(y=2,x=2),strategy=getattr(ttnn.ShardStrategy,name),orientation=ttnn.ShardOrientation.ROW_MAJOR)
def generated(shape,kind):
 if kind=='uniform':return torch.rand(shape,dtype=torch.bfloat16)
 if kind=='signed_zeros':return (torch.zeros(shape,dtype=torch.float32)*torch.randn(shape).sign()).bfloat16()
 if kind=='subnormal':return (torch.randn(shape)*2.0**-129).bfloat16()
 if kind=='nan_input':
  x=torch.randn(shape,dtype=torch.bfloat16);x.reshape(-1)[0]=float('nan');return x
 if kind=='inf_input':
  x=torch.randn(shape,dtype=torch.bfloat16);x.reshape(-1)[0]=float('inf');return x
 if kind=='zeros':return torch.zeros(shape,dtype=torch.bfloat16)
 if kind=='identity':return torch.eye(shape[-2],shape[-1],dtype=torch.bfloat16).expand(shape).contiguous()
 x=torch.randn(shape,dtype=torch.float32)
 if kind=='exponents':x=x.sign()*torch.pow(2.0,torch.randint(-12,13,shape)).float()
 if kind=='large_finite':x=x.sign()*torch.pow(2.0,torch.randint(45,56,shape)).float()
 if kind=='cancellation':x=x.sign()*torch.pow(2.0,torch.randint(-2,10,shape)).float()
 return x.bfloat16()
def perf(device):
 ttnn.synchronize_device(device);ttnn.ReadDeviceProfiler(device)
 programs=[p for group in ttnn.get_latest_programs_perf_data().values() for p in group]
 data=[dict(uid=repr(p.program_execution_uid),cores=p.core_count,available=p.num_available_cores,analyses={k:dict(start=int(v.start_timestamp),end=int(v.end_timestamp),duration=int(v.duration)) for k,v in p.program_analyses_results.items()}) for p in programs]
 durations=[x['analyses']['DEVICE KERNEL DURATION [ns]']['duration'] for x in data]
 assert durations and all(x>0 for x in durations),data
 return dict(kernel_ns=sum(durations),programs=data)
def actual_ref(aa,bb,d,bias):
 x=aa.double();y=bb.double()
 if d.get('transpose_a'):x=x.transpose(-1,-2)
 if d.get('transpose_b'):y=y.transpose(-1,-2)
 z=x@y
 if bias is not None:z=z+bias.double()
 act=d.get('activation')
 if act=='relu':z=torch.relu(z)
 if act=='gelu':z=torch.nn.functional.gelu(z)
 if act=='silu':z=torch.nn.functional.silu(z)
 return z

def case(d,device):
 folder=out/'outputs'/d['id'];folder.mkdir();start=time.monotonic();selector=folder/'selector.log';os.environ['TK_VALID_SELECTOR_LOG']=str(selector)
 rec=dict(case=d,id=d['id'],side=side,status='running',methods={},configs=[],elapsed_s=None)
 allocated=[]
 try:
  torch.manual_seed(d['seed']);b,m,k,n=d['shape'];sa=(*d['batch_a'],m,k);sb=(*d['batch_b'],k,n)
  if d.get('transpose_a'):sa=(*d['batch_a'],k,m)
  if d.get('transpose_b'):sb=(*d['batch_b'],n,k)
  a=generated(sa,d['distribution']);bt=generated(sb,d['distribution'])
  if d['distribution']=='cancellation' and not d.get('transpose_b') and k%2==0:
   a[...,1::2]=a[...,::2];bt[...,1::2,:]=-bt[...,::2,:];bt[...,-1,:]+=torch.tensor(0.25,dtype=torch.bfloat16)
  tensors=[]
  for x,which in [(a,'a'),(bt,'b')]:
   kw=dict(dtype=getattr(ttnn,d['dtype_'+which]),device=device,layout=getattr(ttnn,d['layout']+'_LAYOUT'),memory_config=memory(d['memory_'+which],x.shape))
   if d['tile_'+which]!=[32,32]:kw['tile']=ttnn.Tile(d['tile_'+which])
   t=ttnn.from_torch(x,**kw);tensors.append(t);allocated.append(t)
  ta,tb=tensors
  # References use the values actually stored on device, including BFP quantization.
  aa=ttnn.to_torch(ta).float();bb=ttnn.to_torch(tb).float();rec['inputs']=[dict(shape=list(x.shape),sha256=digest(x)) for x in [aa,bb]]
  kwargs=dict(dtype=getattr(ttnn,d['dtype_out']),memory_config=memory(d['memory_out'],(*torch.broadcast_shapes(d['batch_a'],d['batch_b']),m,n)))
  for flag in ['transpose_a','transpose_b']:
   if d.get(flag):kwargs[flag]=True
  if d.get('activation'):kwargs['activation']=d['activation']
  if d.get('compute'):
   c=dict(d['compute']);c['math_fidelity']=getattr(ttnn.MathFidelity,c['math_fidelity']);kwargs['compute_kernel_config']=ttnn.init_device_compute_kernel_config(device.arch(),**c)
  if d.get('core_grid'):kwargs['core_grid']=ttnn.CoreGrid(x=d['core_grid'][0],y=d['core_grid'][1])
  if d.get('explicit_reuse'):
   grid=device.compute_with_storage_grid_size();kwargs['program_config']=ttnn.MatmulMultiCoreReuseProgramConfig(compute_with_storage_grid_size=(grid.x,grid.y),in0_block_w=1,out_subblock_h=2,out_subblock_w=2,per_core_M=2,per_core_N=2)
  if d.get('output_tile'):kwargs['output_tile']=ttnn.Tile(d['output_tile'])
  if d.get('optional_output'):
   ot=ttnn.from_torch(torch.zeros((*d['batch_a'],m,n),dtype=torch.bfloat16),device=device,layout=ttnn.TILE_LAYOUT,tile=ttnn.Tile(d['output_tile']));kwargs['optional_output_tensor']=ot;allocated.append(ot)
  bias=None
  if d.get('bias'):
   shape={'row':(1,n),'matrix':(m,n),'batched':(*d['batch_a'],m,n)}[d['bias']];bias=torch.randn(shape,dtype=torch.bfloat16)
   t_bias=ttnn.from_torch(bias,device=device,layout=ttnn.TILE_LAYOUT);allocated.append(t_bias);kwargs['bias']=t_bias
   rec['inputs'].append(dict(shape=list(bias.shape),sha256=digest(bias)))
  ref=actual_ref(aa,bb,d,bias);rec['reference_sha256']=digest(ref)
  if side.startswith('main') and not out.name.endswith('A2'):np.save(folder/'reference.npy',ref.numpy(),allow_pickle=False)
  rec['kwargs']={k:repr(v) for k,v in kwargs.items()};ttnn.synchronize_device(device);ttnn.ReadDeviceProfiler(device)
  for method in ['default']+(['moreh'] if d.get('moreh') and side.startswith('pr-') else []):
   fun=ttnn.moreh_bmm if method=='moreh' else getattr(ttnn,d['op']);opts={} if method=='moreh' else kwargs;samples=[]
   for iteration in range(7):
    started=time.monotonic_ns();result=fun(ta,tb,**opts);sample=perf(device);sample['host_bound_ns']=time.monotonic_ns()-started;sample['clock_span_valid']=sample['kernel_ns']<=sample['host_bound_ns']*1.1+1_000_000
    if iteration>=2:samples.append(sample)
    if iteration==6:
     actual=ttnn.to_torch(result).contiguous();bits=actual.view(torch.uint16).numpy() if actual.dtype==torch.bfloat16 else actual.float().numpy().view(np.uint32)
     np.save(folder/(method+'.npy'),bits,allow_pickle=False)
     rec['methods'][method]=dict(samples=samples,median_ns=statistics.median(x['kernel_ns'] for x in samples),checks=metrics(actual,ref),output_shape=list(actual.shape),output_dtype=str(actual.dtype),sha256=hashlib.sha256(bits.tobytes()).hexdigest())
    if not d.get('optional_output'):ttnn.deallocate(result)
   rec['configs']=list(dict.fromkeys(selector.read_text().splitlines())) if selector.exists() else []
  rec['status']='passed'
 except Exception as e:
  rec['status']='error';rec['error']=repr(e);rec['traceback']=traceback.format_exc()
 finally:
  for t in reversed(allocated):
   try:ttnn.deallocate(t)
   except Exception:pass
  rec['elapsed_s']=time.monotonic()-start;dump(folder/'record.json',rec);print(json.dumps({k:rec[k] for k in ['id','status','elapsed_s']}),flush=True);gc.collect()
 return rec

def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',default='plan.json');p.add_argument('--shard',type=int);p.add_argument('--id',action='append');p.add_argument('--first',type=int);p.add_argument('--last',type=int);args=p.parse_args();plan=json.loads((r/args.plan).read_text());selected=[x for x in plan if (args.shard is None or x['shard']==args.shard) and (not args.id or x['id'] in args.id) and (args.first is None or x['ordinal']>=args.first) and (args.last is None or x['ordinal']<=args.last)]
 maps=[line for line in Path('/proc/self/maps').read_text().splitlines() if any(x in line for x in ['_ttnn','libtt_metal'])];dump(out/'loaded-libraries.json',maps)
 expected=json.loads((Path(os.environ.get('VALID_LIBRARIES',str(r/'libraries')))/side/'hashes.json').read_text());paths={line.split()[-1] for line in maps}
 for name in ['_ttnncpp.so','_ttnn.so','libtt_metal.so']:
  file=next(Path(x) for x in paths if x.endswith('/'+name));assert hashlib.sha256(file.read_bytes()).hexdigest()==expected[name],(name,file)
 device=ttnn.open_device(device_id=0);device.enable_program_cache();grid=device.compute_with_storage_grid_size();dump(out/'device.json',dict(arch=str(device.arch()),grid=[grid.x,grid.y],visible=os.environ.get('TT_VISIBLE_DEVICES'),nodes=[p.name for p in Path('/dev/tenstorrent').iterdir()]))
 try:records=[case(d,device) for d in selected]
 finally:ttnn.close_device(device)
 dump(out/'summary.json',[{k:x[k] for k in ['id','status','elapsed_s']} for x in records])
if __name__=='__main__':main()
