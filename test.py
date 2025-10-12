import os
workspace_root = '/home/hbw/improve'
data_root = f'{workspace_root}/data'
output_root = f'{workspace_root}/output'
os.makedirs(output_root, exist_ok=True)

paramList = [
    ['bicycle', 3000000],
    ['flowers', 1500000],
    ['garden', 3000000],
    ['stump', 3000000],
    ['treehill', 1500000],

    ['bonsai', 1000000],
    ['counter', 1000000],
    ['kitchen', 1000000],
    ['room', 1000000],
]

for data, budget in paramList:
    scene_data_path = f'{data_root}/{data}'
    for run_id in range(1, 4):
        output = f'{output_root}/{data}_run{run_id}'

        one_cmd = f'python -u train.py -s "{scene_data_path}" -m "{output}" --budget {budget} '
        os.system(one_cmd)
        two_cmd = f'python -u render.py -m "{output}"'
        os.system(two_cmd)
        three_cmd = f'python -u metrics.py -m "{output}"'
        os.system(three_cmd)
        four_cmd = f'python -u {workspace_root}/metrics-train.py -m "{output}"'
        os.system(four_cmd)
