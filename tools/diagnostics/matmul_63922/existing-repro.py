import json,subprocess,sys,xml.etree.ElementTree as ET
from pathlib import Path

selectors = ['tests/ttnn/unit_tests/operations/matmul/test_custom_grids.py', 'tests/ttnn/unit_tests/operations/matmul/test_experimental.py::test_ttnn_matmul', 'tests/ttnn/unit_tests/operations/matmul/test_linear.py', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_alternating_dst_sync_mode_matmul', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_interleaved_input_sharded_output_matmul', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_linear_broadcast_bias_into_optional_output', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_linear_with_non_tile_aligned_bias', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_linear_with_optional_output_tensor', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_1d_mcast_sharded_output_anchors_at_worker_bbox_start', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_activation_with_sharded_input', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_block_float_ktile_padding', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_block_float_ktile_padding_fp32_dest_acc', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_block_float_ktile_padding_preserves_shared_byte', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_block_float_ktile_padding_transpose_a', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_block_sharded_input_with_padding', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_column_wise_bfp_tilize_via_transpose_b', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_default_block_sharded_single_core', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_default_sharded_non_origin_single_core', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_does_dot_product', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_ktile_padding_non_block_float', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_reuse_config_partial_batch_blocks', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_same_shape_and_valid', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_same_shape_but_invalid', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_with_core_grid', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_with_matched_width_height_4D', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_with_transpose_a_or_b', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_matmul_with_transpose_and_configs', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_optional_output_argument', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_optional_output_argument_with_tiny_tiles', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_optional_output_shape_must_match', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_pytorch_2_0_failed_cases', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_sharded_matmul', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_small_matmul_pcc', 'tests/ttnn/unit_tests/operations/matmul/test_matmul.py::test_tutorial_matmul_inputs_and_output_in_l1_memory', 'tests/ttnn/unit_tests/operations/matmul/test_matmul_batch_mismatch.py', 'tests/ttnn/unit_tests/operations/matmul/test_matmul_program_cache.py', 'tests/ttnn/unit_tests/operations/matmul/test_matmul_fp32_partials_reload.py', 'tests/ttnn/unit_tests/operations/matmul/test_sparse_matmul.py']
data=Path(__file__).resolve().parent
mode=sys.argv[1]
if mode == 'collect':
    cmd=[sys.executable,'-m','pytest','--collect-only','-q','-o','addopts=',*selectors]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True)
    nodes=[line.strip() for line in p.stdout.splitlines() if line.startswith('tests/') and '::' in line]
    (data/'nodeids.json').write_text(json.dumps(nodes,indent=2)+'\n')
    print('collected',len(nodes))
elif mode == 'run':
    nodes=json.loads((data/'nodeids.json').read_text());totals=[0,0,0,0]
    for i,start in enumerate(range(0,len(nodes),35)):
        xml=data/f'existing-{i}.xml'
        cmd=[sys.executable,'-m','pytest','--timeout=75','-q','-o','log_cli=false','--junitxml='+str(xml),*nodes[start:start+35]]
        subprocess.run(cmd)
        root=ET.parse(xml).getroot()
        for suite in root.iter('testsuite'):
            vals=[int(suite.get(k,'0')) for k in ('tests','failures','errors','skipped')]
            totals=[a+b for a,b in zip(totals,vals)]
    print('tests passed failures errors skipped:',totals[0],totals[0]-sum(totals[1:]),*totals[1:])
else:
    raise SystemExit('use collect or run')
