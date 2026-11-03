import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
print('| Shape [B,M,K,N] | Main µs | PR µs | Alternative µs | PR slower |')
print('|---|---:|---:|---:|---:|')
for item in json.loads((r/'plan.json').read_text()):
 if item['shard'] not in (4,5):continue
 label='q'+str(item['shard']);recs=[json.loads((r/'runs'/(label+s)/'outputs'/item['id']/'record.json').read_text()) for s in ('A','B','A2','C','A3')]
 assert all(x['status']=='passed' for x in recs)
 times=[x['methods']['default']['median_ns']/1000 for x in recs];main=statistics.median((times[0],times[2]));slow=100*(times[1]/main-1)
 print('| '+str(item['shape'])+f' | {main:.3f} | {times[1]:.3f} | {times[3]:.3f} | {slow:.2f}% |')
