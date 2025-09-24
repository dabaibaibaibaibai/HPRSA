import networkx as nx
import random
import numpy as np
import pulp
import copy

# 使用与main.py相同的常量
FPGA_TOTAL_RESOURCE = 10
GPU_TOTAL_RESOURCE = 6
DATA_TRANSFER_RATE = 2

def read_dataset(file_path):
    """读取数据集文件，解析任务属性和DAG边信息"""
    task_attributes = []
    dag = nx.DiGraph()
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        task_index = -1
        
        for line in lines:
            if line.startswith("Task "):
                task_index += 1
                task_attributes.append({})
            elif ": " in line:
                attr, value = line.strip().split(": ")
                if attr in ['release_time', 'deadline', 'software_execution_time', 
                          'fpga_execution_time', 'gpu_execution_time', 
                          'fpga_resource_requirement', 'gpu_resource_requirement']:
                    task_attributes[task_index][attr] = int(value)
            elif line.startswith("DAG Edges:"):
                for edge_line in lines[lines.index(line) + 1:]:
                    if " -> " in edge_line:
                        parts = edge_line.strip().split(" (Data Size: ")
                        source, target = map(int, parts[0].split(" -> "))
                        data_size = int(parts[1].rstrip(")"))
                        dag.add_edge(source, target, data_size=data_size)
                        
    print(f"读取到的任务数量: {len(task_attributes)}")
    print(f"读取到的边数量: {len(dag.edges())}")
    
    return dag, task_attributes

class Chromosome:
    def __init__(self, num_tasks):
        self.processor_allocation = []  # 处理器分配
        self.task_sequence = []        # 任务执行顺序
        self.fitness = float('inf')    # 适应度(makespan)
        
    def initialize(self, dag, num_tasks):
        # 随机初始化处理器分配
        self.processor_allocation = [random.choice(["CPU", "FPGA", "GPU"]) 
                                   for _ in range(num_tasks)]
        # 使用拓扑排序初始化任务序列
        self.task_sequence = list(nx.topological_sort(dag))
        self.task_sequence = [x-1 for x in self.task_sequence]  # 转换为0基索引

def evaluate_chromosome(chromosome, dag, task_attributes):
    """评估染色体的适应度(计算makespan)"""
    num_tasks = len(task_attributes)
    finish_times = [0] * num_tasks
    cpu_busy = [0] * 2  # 2个CPU核心
    fpga_resources = [0] * FPGA_TOTAL_RESOURCE
    gpu_resources = [0] * GPU_TOTAL_RESOURCE
    
    for task_id in chromosome.task_sequence:
        # 计算最早开始时间(考虑前驱任务)
        est = 0
        for pred in dag.predecessors(task_id + 1):
            pred_id = pred - 1
            comm_time = dag[pred][task_id + 1]['data_size'] / DATA_TRANSFER_RATE
            est = max(est, finish_times[pred_id] + comm_time)
            
        # 根据分配的处理器计算执行时间
        processor = chromosome.processor_allocation[task_id]
        if processor == "CPU":
            # 找到最早可用的CPU核心
            cpu_id = cpu_busy.index(min(cpu_busy))
            start_time = max(est, cpu_busy[cpu_id])
            exec_time = task_attributes[task_id]['software_execution_time']
            finish_times[task_id] = start_time + exec_time
            cpu_busy[cpu_id] = finish_times[task_id]
            
        elif processor == "FPGA":
            # 检查FPGA资源可用性
            resource_req = task_attributes[task_id]['fpga_resource_requirement']
            start_time = est
            while True:
                available = True
                for r in range(resource_req):
                    if fpga_resources[r] > start_time:
                        start_time = max(start_time, fpga_resources[r])
                        available = False
                if available:
                    break
                    
            exec_time = task_attributes[task_id]['fpga_execution_time']
            finish_times[task_id] = start_time + exec_time
            for r in range(resource_req):
                fpga_resources[r] = finish_times[task_id]
                
        else:  # GPU
            # 检查GPU资源可用性
            resource_req = task_attributes[task_id]['gpu_resource_requirement']
            start_time = est
            while True:
                available = True
                for r in range(resource_req):
                    if gpu_resources[r] > start_time:
                        start_time = max(start_time, gpu_resources[r])
                        available = False
                if available:
                    break
                    
            exec_time = task_attributes[task_id]['gpu_execution_time']
            finish_times[task_id] = start_time + exec_time
            for r in range(resource_req):
                gpu_resources[r] = finish_times[task_id]
                
    return max(finish_times)

def crossover(parent1, parent2):
    """交叉操作"""
    child = copy.deepcopy(parent1)
    # 随机选择交叉点
    cross_point = random.randint(0, len(parent1.processor_allocation)-1)
    # 处理器分配交叉
    child.processor_allocation[cross_point:] = parent2.processor_allocation[cross_point:]
    return child

def mutate(chromosome, mutation_rate=0.1):
    """突变操作"""
    for i in range(len(chromosome.processor_allocation)):
        if random.random() < mutation_rate:
            chromosome.processor_allocation[i] = random.choice(["CPU", "FPGA", "GPU"])
    return chromosome

def mgaa_scheduling(dag, task_attributes, pop_size=50, max_generations=100):
    """MGAA主算法"""
    num_tasks = len(task_attributes)
    population = []
    
    # 初始化种群
    for _ in range(pop_size):
        chromosome = Chromosome(num_tasks)
        chromosome.initialize(dag, num_tasks)
        population.append(chromosome)
    
    best_solution = None
    best_makespan = float('inf')
    
    # 主循环
    for generation in range(max_generations):
        # 评估种群
        for chromosome in population:
            makespan = evaluate_chromosome(chromosome, dag, task_attributes)
            chromosome.fitness = makespan
            if makespan < best_makespan:
                best_makespan = makespan
                best_solution = copy.deepcopy(chromosome)
        
        # 选择
        population.sort(key=lambda x: x.fitness)
        new_population = population[:pop_size//2]  # 保留最好的一半
        
        # 生成新的染色体
        while len(new_population) < pop_size:
            parent1 = random.choice(population[:pop_size//2])
            parent2 = random.choice(population[:pop_size//2])
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)
            
        population = new_population
        
        if generation % 10 == 0:
            print(f"Generation {generation}, Best makespan: {best_makespan}")
            
    return best_solution, best_makespan

if __name__ == "__main__":
    try:
        # 1. 读取数据集
        file_path = "/root/123/FFT/dataset/ccr=1.0/dag_fft_5tasks.txt"
        print(f"正在读取数据集：{file_path}")
        dag, task_attributes = read_dataset(file_path)
        
        if not task_attributes:
            raise ValueError("未读取到任何任务信息")
            
        print(f"成功读取 {len(task_attributes)} 个任务")
        
        # 2. 执行MGAA调度
        print("\n开始执行MGAA调度...")
        best_solution, best_makespan = mgaa_scheduling(dag, task_attributes)
        
        # 3. 输出结果到文件
        output_file = "/root/123/FFT/result/ccr=1.0/MGAA/dag_fft_5tasks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入基本信息
            f.write("================ 调度环境信息 ================\n")
            f.write(f"总任务数: {len(task_attributes)}\n")
            f.write(f"FPGA资源上限: {FPGA_TOTAL_RESOURCE}\n")
            f.write(f"GPU资源上限: {GPU_TOTAL_RESOURCE}\n")
            f.write(f"数据传输速率: {DATA_TRANSFER_RATE}\n\n")
            
            # 写入调度结果
            f.write("================ 调度结果 ================\n")
            f.write(f"最终makespan: {best_makespan:.2f}\n\n")
            
            # 写入任务分配
            f.write("任务分配方案:\n")
            for i, proc in enumerate(best_solution.processor_allocation):
                f.write(f"任务 {i+1} -> {proc}\n")
                
            # 写入执行顺序
            f.write("\n任务执行顺序:\n")
            for i, task_id in enumerate(best_solution.task_sequence):
                f.write(f"{i+1}. 任务 {task_id+1}\n")
        
        print(f"\n结果已保存到文件: {output_file}")
        print(f"最终makespan: {best_makespan:.2f}")
            
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()