import networkx as nx
import random
import numpy as np
import copy

# 定义常量
FPGA_TOTAL_RESOURCE = 10
GPU_TOTAL_RESOURCE = 6
DATA_TRANSFER_RATE = 2

# 遗传算法参数
POPULATION_SIZE = 50
MAX_GENERATIONS = 100
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8

class Individual:
    def __init__(self, task_count):
        self.processor_allocation = []  # 处理器分配
        self.task_sequence = []        # 任务执行顺序
        self.fitness = float('inf')    # 适应度(makespan)
        self.level_matrix = []         # 任务层级矩阵 - GAA-A特有

    def initialize(self, dag, task_count, task_attributes):
        """初始化个体
        Args:
            dag: DAG图
            task_count: 任务数量
            task_attributes: 任务属性字典
        """
        # 1. 计算任务层级
        self.level_matrix = calculate_task_levels(dag, task_count)
        
        # 2. 基于层级的处理器分配
        self.processor_allocation = self.allocate_processors_by_level(task_count)
        
        # 3. 基于拓扑排序和层级生成任务序列
        topo_order = list(nx.topological_sort(dag))
        self.task_sequence = []
        
        # 确保每一层的任务按照拓扑序列的顺序添加
        for level in self.level_matrix:
            level_tasks = [(task, topo_order.index(task+1)) for task in level]
            level_tasks.sort(key=lambda x: x[1])  # 按拓扑顺序排序
            self.task_sequence.extend(task for task, _ in level_tasks)

    def allocate_processors_by_level(self, task_count):
        """基于层级的处理器分配策略"""
        allocation = []
        
        # 遍历每层任务
        for level in self.level_matrix:
            level_size = len(level)
            
            # 根据层内任务数量分配处理器
            if level_size <= 2:
                # 小层优先分配给CPU
                allocation.extend(["CPU"] * level_size)
            else:
                # 大层混合分配
                cpu_count = min(2, level_size // 3)  # 保证CPU负载
                fpga_count = (level_size - cpu_count) // 2
                gpu_count = level_size - cpu_count - fpga_count
                
                allocation.extend(["CPU"] * cpu_count)
                allocation.extend(["FPGA"] * fpga_count)
                allocation.extend(["GPU"] * gpu_count)
        
        # 确保分配结果长度正确
        while len(allocation) < task_count:
            allocation.append(random.choice(["CPU", "FPGA", "GPU"]))
        
        return allocation

    def generate_sequence_by_level(self, dag, task_count, task_attributes):
        """基于层级和执行时间的任务序列生成"""
        sequence = []
        
        # 对每层内的任务按执行时间排序
        for level in self.level_matrix:
            # 计算任务的执行时间
            level_tasks = [(task_id, min(
                task_attributes[task_id]['software_execution_time'],
                task_attributes[task_id]['fpga_execution_time'],
                task_attributes[task_id]['gpu_execution_time']
            )) for task_id in level]
            
            # 按执行时间升序排序
            level_tasks.sort(key=lambda x: x[1])
            sequence.extend(task_id for task_id, _ in level_tasks)
        
        return sequence

def calculate_task_levels(dag, task_count):
    """计算任务的层级矩阵"""
    levels = []
    current_level = set()
    visited = set()
    
    # 找到所有入度为0的任务作为第一层
    for task_id in range(1, task_count + 1):  # 修改这里，使用1-based索引
        if not list(dag.predecessors(task_id)):
            current_level.add(task_id - 1)  # 转换为0-based索引存储
            visited.add(task_id - 1)
    
    if not current_level:  # 如果没有找到入度为0的任务，返回单层
        return [[i for i in range(task_count)]]
    
    while current_level:
        levels.append(list(current_level))
        next_level = set()
        
        for task_id in current_level:
            real_id = task_id + 1  # 转换回1-based索引用于图操作
            for succ in dag.successors(real_id):
                succ_id = succ - 1  # 转换为0-based索引
                if succ_id not in visited:
                    # 检查所有前驱是否都已访问
                    predecessors = set(pred-1 for pred in dag.predecessors(succ))
                    if predecessors.issubset(visited):
                        next_level.add(succ_id)
                        visited.add(succ_id)
        
        current_level = next_level
    
    # 确保所有任务都被分配到层
    unassigned = set(range(task_count)) - visited
    if unassigned:
        levels.append(list(unassigned))
    
    return levels

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
                # 从Task行提取任务ID并添加节点
                task_id = int(line.split()[1].strip(':'))
                dag.add_node(task_id)  # 确保添加节点
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
                        # 确保源节点和目标节点都存在
                        if not dag.has_node(source):
                            dag.add_node(source)
                        if not dag.has_node(target):
                            dag.add_node(target)
                        dag.add_edge(source, target, data_size=data_size)
                        
    print(f"读取到的任务数量: {len(task_attributes)}")
    print(f"读取到的边数量: {len(dag.edges())}")
    print(f"图中的节点: {list(dag.nodes())}")  # 添加调试信息
    
    return dag, task_attributes

def evaluate_makespan(individual, dag, task_attributes):
    """计算调度方案的makespan"""
    num_tasks = len(task_attributes)
    finish_times = [0] * num_tasks
    cpu_busy = [0] * 2  # 2个CPU核心
    fpga_resources = [0] * FPGA_TOTAL_RESOURCE
    gpu_resources = [0] * GPU_TOTAL_RESOURCE
    
    for task_id in individual.task_sequence:
        # 计算最早开始时间(考虑前驱任务)
        est = 0
        for pred in dag.predecessors(task_id + 1):
            pred_id = pred - 1
            comm_time = dag[pred][task_id + 1]['data_size'] / DATA_TRANSFER_RATE
            est = max(est, finish_times[pred_id] + comm_time)
            
        # 根据分配的处理器计算执行时间
        processor = individual.processor_allocation[task_id]
        if processor == "CPU":
            # 找到最早可用的CPU核心
            cpu_id = cpu_busy.index(min(cpu_busy))
            start_time = max(est, cpu_busy[cpu_id])
            exec_time = task_attributes[task_id]['software_execution_time']
            finish_times[task_id] = start_time + exec_time
            cpu_busy[cpu_id] = finish_times[task_id]
            
        elif processor == "FPGA":
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

def evaluate_fitness(individual, dag, task_attributes):
    """GAA-A特有的适应度评估，考虑makespan、负载均衡和层级感知"""
    # 1. 计算基本makespan
    makespan = evaluate_makespan(individual, dag, task_attributes)
    
    # 2. 计算处理器负载均衡度
    proc_loads = {"CPU": 0, "FPGA": 0, "GPU": 0}
    for proc in individual.processor_allocation:
        proc_loads[proc] += 1
    
    # 标准差表示负载不均衡程度
    avg_load = len(individual.processor_allocation) / 3
    load_variance = sum((load - avg_load) ** 2 for load in proc_loads.values()) / 3
    
    # 3. 计算层级分配合理性
    level_score = 0
    for level in individual.level_matrix:
        level_procs = [individual.processor_allocation[task_id] for task_id in level]
        # 同层任务尽量分配到不同处理器
        unique_procs = len(set(level_procs))
        level_score += unique_procs / len(level_procs)
    level_score /= len(individual.level_matrix)
    
    # 4. 计算通信开销
    comm_cost = 0
    for task_id in range(len(individual.processor_allocation)):
        for succ in dag.successors(task_id + 1):
            succ_id = succ - 1
            # 不同处理器间的通信开销
            if individual.processor_allocation[task_id] != individual.processor_allocation[succ_id]:
                comm_cost += dag[task_id + 1][succ]['data_size'] / DATA_TRANSFER_RATE
    
    # 综合评分 (权重可调)
    w1, w2, w3, w4 = 0.4, 0.2, 0.2, 0.2
    fitness = w1 * makespan + \
             w2 * load_variance + \
             w3 * (1 - level_score) * makespan + \
             w4 * comm_cost
             
    return fitness

def crossover(parent1, parent2):
    """层级感知的交叉操作"""
    if random.random() > CROSSOVER_RATE:
        return copy.deepcopy(parent1)
        
    child = copy.deepcopy(parent1)
    
    # 随机选择一个层级进行交叉
    level_idx = random.randint(0, len(parent1.level_matrix) - 1)
    level_tasks = parent1.level_matrix[level_idx]
    
    # 获取该层的任务在序列中的位置
    task_positions = {task: i for i, task in enumerate(parent1.task_sequence)}
    
    # 对选中层的任务应用交叉
    for task_id in level_tasks:
        if random.random() < 0.5:
            # 交换处理器分配
            child.processor_allocation[task_id] = parent2.processor_allocation[task_id]
            
            # 调整任务序列
            p1_pos = task_positions[task_id]
            p2_pos = parent2.task_sequence.index(task_id)
            
            # 在不违反层级约束的情况下调整位置
            level_positions = [i for i, t in enumerate(child.task_sequence) if t in level_tasks]
            if len(level_positions) > 1:
                min_pos = min(level_positions)
                max_pos = max(level_positions)
                
                # 计算新位置，保持在层内范围
                if min_pos < max_pos:
                    new_pos = min_pos + abs(p2_pos - p1_pos) % (max_pos - min_pos + 1)
                    # 移动任务到新位置
                    task = child.task_sequence.pop(p1_pos)
                    child.task_sequence.insert(new_pos, task)
    
    return child

def mutate(individual):
    """层级感知的变异操作"""
    if random.random() > MUTATION_RATE:
        return
        
    # 选择一个随机层级
    level_idx = random.randint(0, len(individual.level_matrix) - 1)
    level = individual.level_matrix[level_idx]
    
    if len(level) > 1:
        # 在层内随机选择两个任务
        task1, task2 = random.sample(level, 2)
        
        # 1. 处理器分配变异
        # 确保不会过度分配给某种处理器
        current_procs = individual.processor_allocation.count("CPU")
        if current_procs < 2:
            procs = ["CPU", "FPGA", "GPU"]
        else:
            procs = ["FPGA", "GPU"]
            
        individual.processor_allocation[task1] = random.choice(procs)
        individual.processor_allocation[task2] = random.choice(procs)
        
        # 2. 任务序列变异
        # 在层内交换任务位置
        pos1 = individual.task_sequence.index(task1)
        pos2 = individual.task_sequence.index(task2)
        individual.task_sequence[pos1], individual.task_sequence[pos2] = \
            individual.task_sequence[pos2], individual.task_sequence[pos1]

def tournament_select(population, tournament_size=3):
    """锦标赛选择"""
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: x.fitness)

def gaa_scheduling(dag, task_attributes):
    """修改后的GAA-A主算法"""
    num_tasks = len(task_attributes)
    population = []
    
    # 初始化种群
    for _ in range(POPULATION_SIZE):
        individual = Individual(num_tasks)
        individual.initialize(dag, num_tasks, task_attributes)  # 修改这里，添加task_attributes参数
        individual.fitness = evaluate_fitness(individual, dag, task_attributes)
        population.append(individual)
    
    best_solution = None
    best_makespan = float('inf')
    
    # 主循环
    for generation in range(MAX_GENERATIONS):
        # 精英保留
        population.sort(key=lambda x: x.fitness)
        new_population = population[:2]
        
        # 适应度缩放
        max_fitness = max(ind.fitness for ind in population)
        min_fitness = min(ind.fitness for ind in population)
        if max_fitness > min_fitness:
            for ind in population:
                ind.scaled_fitness = (max_fitness - ind.fitness) / (max_fitness - min_fitness)
        else:
            for ind in population:
                ind.scaled_fitness = 1.0
        
        # 基于缩放后的适应度选择
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population)
            parent2 = tournament_select(population)
            
            child = crossover(parent1, parent2)
            mutate(child)
            
            child.fitness = evaluate_fitness(child, dag, task_attributes)
            new_population.append(child)
            
        population = new_population
        
        # 更新最优解
        current_best = min(population, key=lambda x: x.fitness)
        current_makespan = evaluate_makespan(current_best, dag, task_attributes)
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_solution = copy.deepcopy(current_best)
            
        if generation % 10 == 0:
            print(f"Generation {generation}:")
            print(f"  Best makespan: {best_makespan:.2f}")
            print(f"  Best fitness: {current_best.fitness:.2f}")
            
    return best_solution, best_makespan

def calculate_mrr(schedule_makespan, dag, task_attributes):
    """计算MRR值"""
    # 使用拓扑排序确保依赖关系
    topo_order = list(nx.topological_sort(dag))
    current_time = 0
    task_finish_times = {}
    
    for task_id in topo_order:
        task_idx = task_id - 1
        # 考虑释放时间
        start_time = max(current_time, task_attributes[task_idx]['release_time'])
        
        # 考虑前驱任务完成时间
        for pred in dag.predecessors(task_id):
            pred_idx = pred - 1
            if pred_idx in task_finish_times:
                # 考虑通信时间
                comm_time = dag[pred][task_id]['data_size'] / DATA_TRANSFER_RATE
                start_time = max(start_time, task_finish_times[pred_idx] + comm_time)
        
        # 使用软件执行时间
        exec_time = task_attributes[task_idx]['software_execution_time']
        finish_time = start_time + exec_time
        
        task_finish_times[task_idx] = finish_time
        current_time = finish_time
    
    Tmax = max(task_finish_times.values())
    return schedule_makespan / Tmax

if __name__ == "__main__":
    try:
        # 1. 读取数据集
        file_path = "/root/123/FFT/dataset/ccr=1.0/dag_fft_5tasks.txt"  # 修改为正确的路径
        print(f"正在读取数据集：{file_path}")
        dag, task_attributes = read_dataset(file_path)
        
        if not task_attributes:
            raise ValueError("未读取到任何任务信息")
            
        print(f"成功读取 {len(task_attributes)} 个任务")
        
        # 2. 执行GAA-A调度
        print("\n开始执行GAA-A调度...")
        best_solution, makespan = gaa_scheduling(dag, task_attributes)
        
        if not best_solution:
            raise ValueError("调度失败")
            
        # 3. 计算MRR
        mrr = calculate_mrr(makespan, dag, task_attributes)
        
        # 4. 输出结果到文件
        output_file = "/root/123/FFT/result/ccr=1.0/GAA-A/dag_fft_5tasks.txt"  # 修改为正确的路径
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入基本信息
            f.write("================ 调度环境信息 ================\n")
            f.write(f"总任务数: {len(task_attributes)}\n")
            f.write(f"FPGA资源上限: {FPGA_TOTAL_RESOURCE}\n")
            f.write(f"GPU资源上限: {GPU_TOTAL_RESOURCE}\n")
            f.write(f"数据传输速率: {DATA_TRANSFER_RATE}\n\n")
            
            # 写入调度结果
            f.write("================ 调度结果 ================\n")
            for i, task_id in enumerate(best_solution.task_sequence):
                f.write(f"任务 {task_id + 1}:\n")
                f.write(f"  处理器: {best_solution.processor_allocation[task_id]}\n")
                
            # 写入性能统计
            f.write("\n================ 性能统计 ================\n")
            f.write(f"调度总时间(makespan): {makespan:.2f}\n")
            f.write(f"平均任务执行时间: {makespan/len(task_attributes):.2f}\n")
            
            # 写入MRR分析
            f.write("\n================ MRR分析 ================\n")
            f.write(f"当前调度makespan (T): {makespan:.2f}\n")
            f.write(f"考虑依赖的串行执行时间 (Tmax): {makespan/mrr:.2f}\n")
            f.write(f"MRR (T/Tmax): {mrr:.4f}\n")
            f.write(f"加速比 (Tmax/T): {1/mrr:.4f}\n")
            
        print(f"\n结果已保存到文件: {output_file}")
        print(f"\nMRR分析:")
        print(f"当前调度makespan (T): {makespan:.2f}")
        print(f"MRR: {mrr:.4f}")
        print(f"加速比: {1/mrr:.4f}")
            
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()