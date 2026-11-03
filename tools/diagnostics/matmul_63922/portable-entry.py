import argparse,os,runpy,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--kind',required=True);p.add_argument('--side',required=True);p.add_argument('--mode',required=True);p.add_argument('--label',required=True);p.add_argument('--shard',type=int,default=0);a=p.parse_args()
r=Path(__file__).resolve().parent;os.environ['VALID_SIDE']=a.side+'-'+a.mode
if a.kind=='perf': script='sweep.py';args=['--shard',str(a.shard)]
elif a.kind=='subdevice_repro':script='repro-subdevice.py';args=[]
elif a.kind=='subdevice_explicit':script='repro-subdevice.py';args=['--explicit-id']
elif a.kind=='transpose_repro':script='sim-transpose-pressure.py';args=[]
elif a.kind=='python':script='pytest_runner.py';args=['tests/ttnn/unit_tests/operations/matmul/test_matmul_batch_parallel.py','--timeout=75']
elif a.kind=='prtests':script='pytest_runner.py';args=['tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_split_by_batch','--timeout=75']
elif a.kind=='existing':script='existing.py';args=['--shard',str(a.shard)]
elif a.kind=='gtest':os.execv('build/test/ttnn/unit_tests_ttnn',['build/test/ttnn/unit_tests_ttnn','--gtest_filter=MatmulSmoke.*','--gtest_output=xml:'+os.environ['VALID_OUT']+'/gtest.xml'])
else:raise SystemExit('Unknown kind: '+a.kind)
sys.argv=[str(r/script),*args];runpy.run_path(str(r/script),run_name='__main__')
