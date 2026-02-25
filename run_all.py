import os

def gen_command(batch_size, population_size):
    num_gpus = 1 if population_size < 65536 else 4
    command = f"sbatch --gpus={num_gpus} --job-name=m_{batch_size}_{population_size} --output=slurm/outs/multinode_{batch_size}_{population_size}.out slurm/multinode.batch {batch_size} {population_size}"
    return command

batch_sizes = [4, 16, 64, 256, 1024]
pop_sizes = [2, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]
# batch_sizes = [4]
# pop_sizes = [2]
for bs in batch_sizes:
    for ps in pop_sizes:
        command = gen_command(bs, ps)
        print(command)
        os.system(command)
