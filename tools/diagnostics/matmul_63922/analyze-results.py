"""Replay A/B/A and A/C/A from retained profiler and bit-pattern receipts."""
from pathlib import Path
import collections,hashlib,json,math,statistics
import numpy as np
r=Path(__file__).resolve().parent;plan=json.loads((r/'plan.json').read_text())
def output(label,rec):
 p=r/'runs'/label/'outputs'/rec['id']/'default.npy';v=np.load(p,allow_pickle=False);assert hashlib.sha256(v.tobytes()).hexdigest()==rec['methods']['default']['sha256'];return v

def errors(a,b,dtype):
 bits=16 if dtype=='torch.bfloat16' else 32;mask=(1<<bits)-1;sign=1<<(bits-1)
 def order(z):
  x=z.astype(np.uint64).reshape(-1);return np.where(x&sign,(~x)&mask,x|sign).astype(np.int64)
 d=np.abs(order(a)-order(b));q=np.percentile(d,[50,95,99,100])
 def values(x):
  raw=(x.astype(np.uint32)<<16) if bits==16 else x.astype(np.uint32)
  return raw.view(np.float32).astype(np.float64).reshape(-1)
 x,y=values(a),values(b);good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
 pcc=float(np.corrcoef(x,y)[0,1]) if x.size>1 and np.std(x)>0 and np.std(y)>0 else None
 return dict(bitwise_equal=bool(np.array_equal(a,b)),changed_lanes=int(np.count_nonzero(a!=b)),total_lanes=int(a.size),finite_lanes=int(x.size),pcc=pcc,ulp_p50=float(q[0]),ulp_p95=float(q[1]),ulp_p99=float(q[2]),ulp_max=int(q[3]))
rows=[];missing=[]
for c in plan:
 prefix='q'+str(10 if c['shard']==0 else c['shard']);labels=[prefix+x for x in ['A','B','A2','C','A3']];paths=[r/'runs'/l/'outputs'/c['id']/'record.json' for l in labels]
 if not all(p.exists() for p in paths):missing.append(c['id']);continue
 recs=[json.loads(p.read_text()) for p in paths];item=dict(id=c['id'],case=c,labels=labels,statuses=[x['status'] for x in recs],errors=[x.get('error') for x in recs])
 if all(x['status']=='passed' for x in recs):
  assert all(x['case']==c for x in recs);assert all(x['inputs']==recs[0]['inputs'] for x in recs);assert all(x['reference_sha256']==recs[0]['reference_sha256'] for x in recs)
  arrays=[output(l,x) for l,x in zip(labels,recs)];dtype=recs[0]['methods']['default']['output_dtype'];base_stable=all(np.array_equal(arrays[0],arrays[i]) for i in [2,4]);times=[x['methods']['default']['median_ns']/1000 for x in recs];pr_base=statistics.median([times[0],times[2]]);ours_base=statistics.median([times[2],times[4]])
  item.update(times_us=times,pr_baseline_us=pr_base,ours_baseline_us=ours_base,pr_delta_pct=100*(times[1]/pr_base-1),ours_delta_pct=100*(times[3]/ours_base-1),base_bitwise_stable=base_stable,pr_output=errors(arrays[0],arrays[1],dtype),ours_output=errors(arrays[2],arrays[3],dtype),configs=[x['configs'] for x in recs],torch_checks=[x['methods']['default']['checks'] for x in recs],core_counts=[[p['cores'] for p in x['methods']['default']['samples'][0]['programs']] for x in recs])
  if 'moreh' in recs[1]['methods']:item['moreh_us']=recs[1]['methods']['moreh']['median_ns']/1000
 rows.append(item)
result=dict(cases=len(rows),missing=missing,rows=rows);(r/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(cases=len(rows),missing=len(missing),unequal=sum(not x.get('pr_output',{}).get('bitwise_equal',True) for x in rows),failures=sum(any(s!='passed' for s in x['statuses']) for x in rows)),indent=2))
