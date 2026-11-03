import json
from pathlib import Path
r=Path(__file__).resolve().parent
rows=json.loads((r/'analysis.json').read_text())['rows']
issue=[x for x in rows if x['case']['group']=='issue']
print('| Shape [B,M,K,N] | Main µs | PR µs | Alternative µs | moreh_bmm µs | PR bitwise = main |')
print('|---|---:|---:|---:|---:|---|')
for x in issue:
 t=x['times_us'];o=x['pr_output']; shape=x['case']['shape']
 print('| '+str(shape)+' | '+f"{x['pr_baseline_us']:.3f} | {t[1]:.3f} | {t[3]:.3f} | {x['moreh_us']:.3f} | "+('yes' if o['bitwise_equal'] else f"no; PCC {o['pcc']:.5f}, ULP p99/max {o['ulp_p99']:.0f}/{o['ulp_max']}")+' |')
print('PR bitwise',sum(x['pr_output']['bitwise_equal'] for x in issue),'of',len(issue))
print('alternative bitwise',sum(x['ours_output']['bitwise_equal'] for x in issue),'of',len(issue))
for side,index in [('main',0),('PR',1)]:
 values=[x['torch_checks'][index]['pcc'] for x in issue];print(side,'Torch PCC range',min(values),max(values))
for shape in ([8,64,4096,128],[32,64,4096,64]):
 x=next(y for y in issue if y['case']['shape']==shape)
 print(shape,'main/PR Torch PCC',[x['torch_checks'][i]['pcc'] for i in (0,1)],'relative L2',[x['torch_checks'][i]['relative_l2'] for i in (0,1)])
for x in rows:
 if x['id']=='explicit-af363adac030':print('explicit grid PR/main',x['pr_output'],'Torch PCC',[x['torch_checks'][i]['pcc'] for i in (0,1)],'alternative/main',x['ours_output'])
