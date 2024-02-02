import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=str, required=True, help="The gpu id(s) to run.")
parser.add_argument("--target", type=str, required=True, help="The target file to read tasks from.")
args = parser.parse_args()

executed_cmd = []
while True:
    with open(args.target, 'r') as f:
        lines = f.readlines()

    if len(lines) == 0:
        break
    
    cmd = lines[0].strip()
    lines = lines[1:]

    with open(args.target, 'w') as f:
        f.writelines(lines)

    if len(cmd) == 0:
        break

    print(cmd)
    os.system(f'{cmd} CUDA_VISIBLE_DEVICES={args.gpus}')
