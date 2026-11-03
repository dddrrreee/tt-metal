"""Collect or run selected existing tests in bounded, paired shards."""
from pathlib import Path
import argparse,json,os,subprocess,sys
r=Path(__file__).resolve().parent;out=Path(os.environ['VALID_OUT'])
p=argparse.ArgumentParser();p.add_argument('--collect',action='store_true');p.add_argument('--shard',type=int,default=0);a=p.parse_args()
selectors=json.loads((r/'existing-selectors.json').read_text()) if a.collect else json.loads((r/'existing-plan.json').read_text())[a.shard]['ids']
cmd=[sys.executable,'-m','pytest','-p','runtime_provenance','-q','--timeout=75','-o','log_cli=false','-o','cache_dir='+str(out/'cache'),'--junitxml='+str(out/'pytest.xml')]
if a.collect:cmd+=['--collect-only']
cmd+=selectors
(out/'pytest-command.json').write_text(json.dumps(cmd,indent=2)+'\n');raise SystemExit(subprocess.run(cmd).returncode)
