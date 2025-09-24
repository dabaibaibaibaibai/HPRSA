import random
import os

def generate_gaussian_dataset(p=5, ccr=1.0):
    """
    生成高斯消元法任务的DAG数据集
    Args:
        p: 参数p,实际矩阵大小为p*p
        ccr: 通信计算比
    """
    # 基本执行时间范围
    base_time_range = (10, 35)
    tasks = []
    total_comp = 0  # 用于计算通信量
    task_id = 1
    
    # 计算任务数N = (p^2+p-2)/2
    N = int((p * p + p - 2) / 2)
    
    # 生成每个任务的属性
    for _ in range(N):  # 生成N个任务节点
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

    # 修改：先构建完整的任务依赖矩阵
    edges = []
    total_comm = 0
    
    # 计算目标通信大小
    avg_comp = total_comp / N
    target_comm = avg_comp * ccr
    data_size = max(1, int(target_comm / N))
    
    # 按照高斯消元算法生成依赖关系
    task_matrix = [[0] * p for _ in range(p)]  # 用于跟踪每个位置对应的任务ID
    task_id = 1
    
    # 填充任务矩阵
    for k in range(p-1):  # 每一轮消元
        for i in range(k+1, p):  # 被消元的行
            if task_id <= N:
                task_matrix[i][k] = task_id
                task_id += 1
    
    # 生成边关系
    for k in range(p-1):
        for i in range(k+1, p):
            current_task = task_matrix[i][k]
            if current_task == 0 or current_task > N:
                continue
                
            # 添加行内依赖
            if k > 0:
                prev_task = task_matrix[i][k-1]
                if prev_task > 0 and prev_task <= N:
                    edges.append((prev_task, current_task, data_size))
                    total_comm += data_size
            
            # 添加列内依赖
            if i > k+1:
                prev_task = task_matrix[i-1][k]
                if prev_task > 0 and prev_task <= N:
                    edges.append((prev_task, current_task, data_size))
                    total_comm += data_size
    
    # 确保所有任务都有依赖关系（添加到前一个任务的依赖）
    tasks_with_edges = set([edge[0] for edge in edges] + [edge[1] for edge in edges])
    for task_id in range(1, N+1):
        if task_id not in tasks_with_edges:
            if task_id > 1:
                edges.append((task_id-1, task_id, data_size))
                total_comm += data_size
            tasks_with_edges.add(task_id)
    
    return tasks, edges, total_comm

def save_dataset(tasks, edges, total_comm, filename):
    """保存数据集到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        # 写入调度环境信息
        f.write("================ 调度环境信息 ================\n")
        f.write(f"总任务数: {len(tasks)}\n")
        f.write("FPGA资源上限: 10\n")
        f.write("GPU资源上限: 6\n") 
        f.write("数据传输速率: 2\n\n")
        
        # 写入任务信息
        f.write("================ 任务信息 ================\n")
        for task in tasks:
            f.write(f"Task {task['id']}:\n")
            f.write(f"  release_time: {task['release_time']}\n")
            f.write(f"  deadline: {task['deadline']}\n")
            f.write(f"  software_execution_time: {task['software_execution_time']}\n")
            f.write(f"  fpga_execution_time: {task['fpga_execution_time']}\n")
            f.write(f"  gpu_execution_time: {task['gpu_execution_time']}\n")
            f.write(f"  fpga_resource_requirement: {task['fpga_resource_requirement']}\n")
            f.write(f"  gpu_resource_requirement: {task['gpu_resource_requirement']}\n\n")
            
        # 写入任务依赖关系，修改为符合main.py要求的格式
        f.write("DAG Edges:\n")  # 改为 "DAG Edges:" 标记
        for edge in edges:
            src, dst, size = edge
            f.write(f"{src} -> {dst} (Data Size: {size})\n")

if __name__ == "__main__":
    # 设置参数
    p = 5  # 可以手动调整,p为矩阵大小
    ccr = 1.0  # 可以手动调整通信计算比
    
    # 计算任务数N = (p^2+p-2)/2
    N = int((p * p + p - 2) / 2)
    
    # 创建数据集目录
    folder = f"/root/123/gausse/dataset/ccr=1.0/N={N}"
    os.makedirs(folder, exist_ok=True)
    
    tasks, edges, total_comm = generate_gaussian_dataset(p, ccr)
    filename = f"{folder}/dag_gausse_{N}tasks.txt"
    save_dataset(tasks, edges, total_comm, filename)
    print(f"Generated {filename}, tasks count: {N}")