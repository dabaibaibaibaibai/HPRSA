import math 
import random
import os
import json

# 创建保存数据集的目录
def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def generate_fft_dataset(p, ccr=1.0):
    """
    生成FFT任务的DAG数据集
    Args:
        p: 输入参数p，任务数N = 2*p - 1 + p * log2(p)
        ccr: 通信计算比
    """
    # 基本执行时间范围
    base_time_range = (10, 35)
    tasks = []
    total_comp = 0  # 用于计算通信量
    task_id = 1
    
    # 计算任务数 N = 2*p - 1 + p * log2(p)
    N = int(2 * p - 1 + p * math.log2(p))
    
    # 生成每个任务的属性
    for _ in range(N):
        # 生成基本执行时间
        base_time = random.randint(*base_time_range)
        
        # CPU、FPGA和GPU的执行时间
        software_execution_time = base_time
        fpga_execution_time = int(base_time * random.uniform(0.5, 0.8))
        gpu_execution_time = int(base_time * random.uniform(0.6, 0.9))
        
        # 计算最小执行时间用于后续CCR计算
        min_execution_time = min(software_execution_time, fpga_execution_time, gpu_execution_time)
        total_comp += min_execution_time
        
        # 资源需求(1-3)
        fpga_resource = random.randint(1, 3)
        gpu_resource = random.randint(1, 3)
        
        # 时间窗口限制
        release_time = random.randint(0, 20)
        deadline = release_time + base_time * 3
        
        task = {
            'id': task_id,
            'release_time': release_time,
            'deadline': deadline,
            'software_execution_time': software_execution_time,
            'fpga_execution_time': fpga_execution_time,
            'gpu_execution_time': gpu_execution_time,
            'fpga_resource_requirement': fpga_resource,
            'gpu_resource_requirement': gpu_resource
        }
        tasks.append(task)
        task_id += 1

    # 根据FFT算法结构生成任务依赖关系
    edges = []
    total_comm = 0
    
    # 计算目标通信大小
    avg_comp = total_comp / N
    target_comm = avg_comp * ccr
    data_size = max(1, int(target_comm / N))
    
    # FFT算法的依赖关系生成
    stages = int(math.log2(p))  # FFT的阶段数
    
    # 第1阶段: 初始化阶段，生成2p-1个节点的完全二叉树
    current_id = 1
    for level in range(p-1):  # 生成完全二叉树的边
        nodes_in_level = min(2**level, p)
        for node in range(nodes_in_level):
            if current_id * 2 <= N:
                edges.append((current_id, current_id * 2, data_size))
                total_comm += data_size
            if current_id * 2 + 1 <= N:
                edges.append((current_id, current_id * 2 + 1, data_size))
                total_comm += data_size
            current_id += 1
            
    # 第2阶段: FFT蝶形运算阶段
    base_id = 2 * p  # 从完全二叉树后的节点开始
    for stage in range(stages):
        stage_size = p // (2 ** stage)  # 每个阶段的分组大小
        for group in range(2 ** stage):
            group_start = group * stage_size
            for i in range(stage_size // 2):
                # 每个蝶形单元包含两个输入和两个输出
                if base_id <= N:
                    # 添加蝶形依赖关系
                    input1 = group_start + i
                    input2 = group_start + i + stage_size//2
                    
                    # 确保源任务ID在有效范围内
                    if input1 + 1 <= N and input2 + 1 <= N and base_id + 1 <= N:
                        edges.append((input1 + 1, base_id, data_size))
                        edges.append((input2 + 1, base_id, data_size))
                        total_comm += 2 * data_size
                    base_id += 1

    return tasks, edges, total_comm

def save_dataset(tasks, edges, total_comm, filename):
    """保存数据集到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入基本信息
        f.write("DAG Type: FFT\n")
        f.write(f"Number of Tasks: {len(tasks)}\n")
        f.write(f"CCR Value: {total_comm/sum(min(t['software_execution_time'], t['fpga_execution_time'], t['gpu_execution_time']) for t in tasks):.1f}\n\n")
        
        # 写入任务信息
        for task in tasks:
            f.write(f"Task {task['id']} Attributes:\n")
            f.write(f"  release_time: {task['release_time']}\n")
            f.write(f"  deadline: {task['deadline']}\n")
            f.write(f"  software_execution_time: {task['software_execution_time']}\n")
            f.write(f"  fpga_execution_time: {task['fpga_execution_time']}\n")
            f.write(f"  gpu_execution_time: {task['gpu_execution_time']}\n")
            f.write(f"  fpga_resource_requirement: {task['fpga_resource_requirement']}\n")
            f.write(f"  gpu_resource_requirement: {task['gpu_resource_requirement']}\n\n")
        
        # 写入边信息
        f.write("DAG Edges:\n")
        for edge in edges:
            src, dst, size = edge
            f.write(f"{src} -> {dst} (Data Size: {size})\n")

if __name__ == "__main__":
    # 设置参数
    p = 32  # 可以手动调整，任务数N = 2*p - 1 + p * log2(p)
    ccr_values = [0.1, 1.0]  # CCR值固定为0.1和1.0
    
    for ccr in ccr_values:
        # 计算任务数
        N = int(2 * p - 1 + p * math.log2(p))
        
        # 创建数据集目录
        folder = f"/root/123/FFT/dataset/ccr={ccr}"
        create_dir(folder)
        
        # 生成数据集
        tasks, edges, total_comm = generate_fft_dataset(p, ccr)
        filename = f"{folder}/dag_fft_{N}tasks.txt"
        save_dataset(tasks, edges, total_comm, filename)
        print(f"Generated {filename}, tasks count: {N}")