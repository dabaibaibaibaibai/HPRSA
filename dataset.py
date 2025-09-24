import numpy as np
import networkx as nx
import random
import argparse  # 添加这行导入

def generate_task_attributes(num_tasks, dag, ccr):
    """生成任务属性，基于论文中的参数设置"""
    task_attributes = []
    
    # 计算层级信息
    layers = {1: 0}  # 根节点在第0层
    for node in nx.topological_sort(dag):
        if node not in layers:
            pred_max_layer = max([layers[pred] for pred in dag.predecessors(node)] or [0])
            layers[node] = pred_max_layer + 1
            
    max_layer = max(layers.values())
    
    # 生成基础计算时间
    for task_id in range(1, num_tasks + 1):
        # 基础执行时间 - 论文中建议使用范围
        base_time = random.randint(10, 35)
        
        # 异构处理器的执行时间
        software_execution_time = base_time
        fpga_execution_time = int(base_time * random.uniform(0.5, 0.8))  # FPGA快于CPU
        gpu_execution_time = int(base_time * random.uniform(0.6, 0.9))   # GPU快于CPU
        
        # 资源需求
        fpga_resource_requirement = random.randint(1, 3)
        gpu_resource_requirement = random.randint(1, 3)
        
        # 计算释放时间和截止时间
        layer = layers[task_id]
        release_time = layer * 5  # 每层5个时间单位
        
        # 截止时间考虑关键路径
        path_length = (max_layer - layer + 1) * base_time
        deadline = release_time + path_length + int(path_length * 0.5)  # 增加50%的松弛时间
        
        task_attributes.append({
            'release_time': release_time,
            'deadline': deadline,
            'software_execution_time': software_execution_time,
            'fpga_execution_time': fpga_execution_time,
            'gpu_execution_time': gpu_execution_time,
            'fpga_resource_requirement': fpga_resource_requirement,
            'gpu_resource_requirement': gpu_resource_requirement
        })
    
    # 调整通信量以满足CCR要求
    total_comp = sum(min(attr['software_execution_time'], 
                        attr['fpga_execution_time'],
                        attr['gpu_execution_time']) 
                    for attr in task_attributes)
    avg_comp = total_comp / num_tasks
    target_comm = avg_comp * ccr
    
    # 调整边的权重
    edges = list(dag.edges(data=True))
    if edges:
        current_total_comm = sum(data['data_size'] for _, _, data in edges)
        if current_total_comm > 0:
            scale_factor = (target_comm * len(edges)) / current_total_comm
            for _, _, data in edges:
                data['data_size'] = max(1, int(data['data_size'] * scale_factor))
    
    return task_attributes


def generate_tree_dag(num_tasks):
    """生成树形DAG"""
    G = nx.DiGraph()
    
    # 添加所有节点
    for i in range(1, num_tasks + 1):
        G.add_node(i)
    
    # 限制每个节点的最大子节点数为3
    current_level = [1]  # 从根节点开始
    next_node = 2
    
    while next_node <= num_tasks and current_level:
        next_level = []
        for parent in current_level:
            # 为每个父节点生成1-3个子节点
            num_children = min(random.randint(1, 3), num_tasks - next_node + 1)
            for _ in range(num_children):
                if next_node <= num_tasks:
                    G.add_edge(parent, next_node, data_size=random.randint(2, 8))
                    next_level.append(next_node)
                    next_node += 1
        current_level = next_level
    
    return G

def generate_fork_dag(num_tasks):
    """生成fork型DAG"""
    G = nx.DiGraph()
    
    # 添加所有节点
    for i in range(1, num_tasks + 1):
        G.add_node(i)
    
    # 计算中间并行层的节点数
    parallel_nodes = num_tasks - 2  # 减去入口和出口节点
    middle_node = num_tasks // 2
    
    # 添加fork边（从入口节点到中间节点）
    for i in range(2, middle_node + 1):
        G.add_edge(1, i, data_size=random.randint(2, 8))
    
    # 添加join边（从中间节点到出口节点）
    for i in range(middle_node + 1, num_tasks):
        parent = random.randint(2, middle_node)
        G.add_edge(parent, i, data_size=random.randint(2, 8))
    
    # 添加到最后一个节点的边
    for i in range(middle_node + 1, num_tasks):
        G.add_edge(i, num_tasks, data_size=random.randint(2, 8))
    
    return G

def generate_random_dag(num_tasks):
    """生成随机DAG"""
    G = nx.DiGraph()
    
    # 添加所有节点
    for i in range(1, num_tasks + 1):
        G.add_node(i)
    
    # 确保每个节点（除了入口节点）至少有一个入边
    for i in range(2, num_tasks + 1):
        # 选择1-3个前驱节点
        num_predecessors = random.randint(1, min(3, i-1))
        predecessors = random.sample(range(1, i), num_predecessors)
        for pred in predecessors:
            G.add_edge(pred, i, data_size=random.randint(2, 8))
    
    # 添加一些额外的随机边
    max_extra_edges = num_tasks // 2
    for _ in range(max_extra_edges):
        source = random.randint(1, num_tasks - 1)
        target = random.randint(source + 1, num_tasks)
        if not G.has_edge(source, target):
            G.add_edge(source, target, data_size=random.randint(2, 8))
            # 如果产生环，则移除该边
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(source, target)
    
    return G

def verify_dag(G, num_tasks):
    """验证DAG的有效性"""
    if not nx.is_directed_acyclic_graph(G):
        return False
    if len(G.nodes()) != num_tasks:
        return False
    if not nx.is_weakly_connected(G):
        return False
    return True

def parse_arguments():
    parser = argparse.ArgumentParser(description='生成具有指定CCR的DAG')
    parser.add_argument('--ccr', type=float, default=0.1)
    parser.add_argument('--num_tasks', type=int, default=30)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    num_tasks = args.num_tasks
    ccr = args.ccr
    max_attempts = 5
    
    print(f"开始生成DAG: 任务数={num_tasks}, CCR={ccr}")
    
    for attempt in range(max_attempts):
        try:
            dag_types = ['tree', 'fork', 'random']
            selected_dag_type = random.choice(dag_types)
            
            if selected_dag_type == 'tree':
                dag = generate_tree_dag(num_tasks)
            elif selected_dag_type == 'fork':
                dag = generate_fork_dag(num_tasks)
            else:
                dag = generate_random_dag(num_tasks)
                
            if verify_dag(dag, num_tasks):
                tasks = generate_task_attributes(num_tasks, dag, ccr)

                filename = f'/root/123/Energy consumption/ccr=0.1/dag_{selected_dag_type}_{num_tasks}_ccr_{ccr:.1f}.txt'
                with open(filename, 'w') as file:
                    file.write(f"DAG Type: {selected_dag_type}\n")
                    file.write(f"Number of Tasks: {num_tasks}\n")
                    file.write(f"CCR Value: {ccr}\n\n")
                    
                    for i, task in enumerate(tasks):
                        file.write(f"Task {i + 1} Attributes:\n")
                        for attr, value in task.items():
                            file.write(f"  {attr}: {value}\n")
                        file.write("\n")
                    
                    file.write("DAG Edges:\n")
                    total_comm = 0
                    for u, v, data in dag.edges(data=True):
                        file.write(f"{u} -> {v} (Data Size: {data['data_size']})\n")
                        total_comm += data['data_size']
                    
                    file.write(f"\nTotal Communication Volume: {total_comm}\n")
                
                print(f"成功生成 {selected_dag_type} 型 DAG")
                print(f"任务数: {num_tasks}")
                print(f"CCR值: {ccr}")
                print(f"总通信量: {total_comm}")
                print(f"结果已保存到 {filename}")
                break
                
        except Exception as e:
            print(f"尝试 {attempt + 1} 失败: {str(e)}")
            if attempt == max_attempts - 1:
                print("达到最大尝试次数,生成失败")