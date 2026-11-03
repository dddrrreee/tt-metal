"""Replay each measured program's kernel span from raw clock receipts."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,csv,json,re,socket

def verify(path):
 offsets=list(csv.reader((path/'clock-offsets.csv').open()));offset_values={int(x[5]) for x in offsets};groups=collections.defaultdict(set);lines=0
 for row in csv.reader((path/'clock-markers.csv').open()):
  uid,chip,x,y,risc,name,kind,raw,offset,corrected=row;raw,offset,corrected=map(int,(raw,offset,corrected));assert raw-offset==corrected,row;assert offset in offset_values,row
  groups[int(uid)].add((int(chip),int(x),int(y),int(risc),name,int(kind),raw,offset,corrected));lines+=1
 count=0;per_core=0;raw_spans=[];corrected_spans=[];ratios=[];errors=[]
 for file in sorted((path/'outputs').glob('*/record.json')):
  d=json.loads(file.read_text())
  for method,obs in d['methods'].items():
   for sample in obs['samples']:
    for program in sample['programs']:
     uid=int(re.search(r'runtime_id=(\d+)',program['uid'])[1]);events=groups[uid];starts=[x for x in events if x[5]==0];ends=[x for x in events if x[5]==1]
     assert starts and ends,(file,uid)
     s=min(x[8] for x in starts);e=max(x[8] for x in ends);a=program['analyses']['DEVICE KERNEL DURATION [ns]'];assert (s,e)==(a['start'],a['end']),(file,method,uid,(s,e),a)
     raw_span=max(x[6] for x in ends)-min(x[6] for x in starts);raw_spans.append(raw_span);corrected_spans.append(e-s);ratios.append((e-s)/a['duration']);count+=1
     local=collections.defaultdict(list)
     for event in events:local[event[:5]].append(event)
     for key,es in local.items():
      ss=[x for x in es if x[5]==0];ee=[x for x in es if x[5]==1]
      assert ss and ee,(uid,key)
      assert max(x[6] for x in ee)-min(x[6] for x in ss)==max(x[8] for x in ee)-min(x[8] for x in ss);per_core+=1
 assert count>0
 return dict(label=path.name,raw_marker_lines=lines,distinct_programs=len(groups),measured_programs_replayed=count,per_core_risc_duration_checks=per_core,calibrated_worker_cores=len(offsets),offset_range_cycles=[min(offset_values),max(offset_values)],max_raw_span_cycles=max(raw_spans),max_corrected_span_cycles=max(corrected_spans),cycles_per_ns_observed_range=[min(ratios),max(ratios)],result='passed')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('labels',nargs='+');a=p.parse_args();r=Path(__file__).resolve().parent;start=datetime.now(timezone.utc).isoformat();rows=[verify(r/'runs'/x) for x in a.labels]
 rec=dict(start_utc=start,end_utc=datetime.now(timezone.utc).isoformat(),host=socket.gethostname(),chips=[],variant=json.loads((r/'pins.json').read_text()),command=['verify_clocks.py',*a.labels],exit_code=0,log_path=None,artifact_paths=[str(r/'runs'/x) for x in a.labels],results=rows)
 out=r/('clock-replay-'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json');out.write_text(json.dumps(rec,indent=2)+'\n');print(json.dumps(rows,indent=2))
