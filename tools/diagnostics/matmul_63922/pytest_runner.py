from pathlib import Path
import os,subprocess,sys,json
out=Path(os.environ['VALID_OUT'])
cmd=[sys.executable,'-m','pytest','-p','runtime_provenance','-q','-o','log_cli=false','-o','cache_dir='+str(out/'cache'),'--junitxml='+str(out/'pytest.xml'),*sys.argv[1:]]
(out/'pytest-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
raise SystemExit(subprocess.run(cmd,env=dict(os.environ,TK_VALID_SELECTOR_LOG=str(out/'selector.log'))).returncode)
