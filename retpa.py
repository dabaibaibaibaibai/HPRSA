import pulp
import networkx as nx
import numpy as np

FPGA_TOTAL_RESOURCE = 10
GPU_TOTAL_RESOURCE = 6
DATA_TRANSFER_RATE = 2
task_processor_map = {}  # 添加全局变量
# ================== 数据集读取部分 ==================
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
                          'fpga_execution_time',  
                          'fpga_resource_requirement']:
                    task_attributes[task_index][attr] = int(value)
            elif line.startswith("DAG Edges:"):
                for edge_line in lines[lines.index(line) + 1:]:
                    if " -> " in edge_line:
                        parts = edge_line.strip().split(" (Data Size: ")
                        source, target = map(int, parts[0].split(" -> "))
                        data_size = int(parts[1].rstrip(")"))
                        dag.add_edge(source, target, data_size=data_size)
                        
    # 添加调试信息
    print(f"读取到的任务数量: {len(task_attributes)}")
    print(f"读取到的边数量: {len(dag.edges())}")
    for i, attr in enumerate(task_attributes):
        print(f"任务 {i+1} 属性: {attr}")
        
    return dag, task_attributes

# ================== ILP任务分配部分 ==================
def ilp_task_allocation(dag, task_attributes, fpga_total_resource, data_transfer_rate):
    """使用ILP进行任务分配，只考虑CPU和FPGA"""
    num_tasks = len(task_attributes)
    model = pulp.LpProblem("Task_Allocation_Problem", pulp.LpMinimize)
    
    # 定义任务分配变量 - 移除GPU变量
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # CPU
    y = {i: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # FPGA
    
    # 为通信引入辅助变量 - 移除GPU相关
    comm_xy = {}
    comm_yx = {}
    
    for u, v, _ in dag.edges(data=True):
        u, v = u-1, v-1
        comm_xy[(u,v)] = pulp.LpVariable(f"comm_xy_{u}_{v}", cat=pulp.LpBinary)
        comm_yx[(u,v)] = pulp.LpVariable(f"comm_yx_{u}_{v}", cat=pulp.LpBinary)
        
        # 添加通信变量约束
        model += comm_xy[(u,v)] <= x[u]
        model += comm_xy[(u,v)] <= y[v]
        model += comm_xy[(u,v)] >= x[u] + y[v] - 1
        
        model += comm_yx[(u,v)] <= y[u]
        model += comm_yx[(u,v)] <= x[v]
        model += comm_yx[(u,v)] >= y[u] + x[v] - 1
    
    # 构建目标函数 - 移除GPU相关
    execution_time = pulp.lpSum(
        x[i] * task_attributes[i]['software_execution_time'] + 
        y[i] * task_attributes[i]['fpga_execution_time']
        for i in range(num_tasks)
    )
    
    # 通信时间部分
    communication_time = pulp.lpSum(
        (data['data_size'] / data_transfer_rate) * (
            comm_xy[(u-1,v-1)] + comm_yx[(u-1,v-1)]
        )
        for u, v, data in dag.edges(data=True)
    )
    
    model += execution_time + communication_time, "Total_Time_Sum"
    
    # 约束条件
    for i in range(num_tasks):
        model += x[i] + y[i] == 1  # 每个任务只能分配到CPU或FPGA
    
    # FPGA资源约束
    model += pulp.lpSum(y[i] * task_attributes[i]['fpga_resource_requirement'] 
                       for i in range(num_tasks)) <= fpga_total_resource
    
    status = model.solve()
    
    # 处理求解结果
    allocation_result = []
    if pulp.LpStatus[model.status] == "Optimal":
        for i in range(num_tasks):
            if x[i].value() > 0.5:
                allocation_result.append(("CPU", i))
            elif y[i].value() > 0.5:
                allocation_result.append(("FPGA", i))
    else:
        print("ILP未找到最优解")
        
    return allocation_result  # 添加这行

# 计算任务权重
def calculate_task_weight(dag, task_attributes, est, lft):
    """计算任务权重
    Args:
        dag: DAG图
        task_attributes: 任务属性
        est: 最早开始时间
        lft: 最晚完成时间
    """
    def _recursive_weight(task_id):
        successors = list(dag.successors(task_id + 1))
        if not successors:  # 叶子节点
            # 使用处理器上最小执行时间
            min_exec_time = min(
                task_attributes[task_id]['software_execution_time'],
                task_attributes[task_id]['fpga_execution_time']
            )
            window = lft[task_id] - est[task_id]
            return min_exec_time / window if window > 0 else float('inf')
            
        # 考虑后继任务权重和通信时间
        weight_sum = 0
        for succ in successors:
            succ_id = succ - 1
            comm_time = dag[task_id + 1][succ]['data_size']
            weight_sum += (_recursive_weight(succ_id) + comm_time)
            
        # 使用处理器上最小执行时间    
        min_exec_time = min(
            task_attributes[task_id]['software_execution_time'],
            task_attributes[task_id]['fpga_execution_time']
        )
        window = lft[task_id] - est[task_id]
        return weight_sum + (min_exec_time / window if window > 0 else float('inf'))

    num_tasks = len(task_attributes)
    task_weights = []
    for i in range(num_tasks):
        try:
            weight = _recursive_weight(i)
            task_weights.append(weight)
        except Exception as e:
            print(f"计算任务 {i+1} 权重时出错: {e}")
            task_weights.append(float('inf'))
    return task_weights

# 计算任务的综合优先级
def calculate_comprehensive_priority(task_id, ct, task_attributes, est, lft, task_weights, dag, 
                                  fpga_total_resource, cpu_processors, 
                                  fpga_processor, task_processor_map):
    """计算任务的综合优先级，仅考虑CPU和FPGA"""
    try:
        # 1. 时间紧迫度
        aed = lft[task_id] - max(est[task_id], ct)
        time_urgency = 1 / aed if aed > 0 else float('inf')
        
        # 2. 关键路径优先级
        successors = list(dag.successors(task_id + 1))
        path_priority = len(successors) / len(task_attributes) if task_attributes else 0
        
        # 3. FPGA资源利用率
        if task_processor_map[task_id] == "FPGA":
            resource_usage = task_attributes[task_id]['fpga_resource_requirement'] / fpga_total_resource
        else:
            resource_usage = 0
        
        # 4. 通信开销
        comm_edges = list(dag.edges(data=True))
        if comm_edges:
            max_comm = max(edge[2]['data_size'] for edge in comm_edges)
            comm_cost = sum(dag[task_id + 1][succ]['data_size'] for succ in successors) if successors else 0
            comm_factor = 1 - (comm_cost / max_comm) if max_comm > 0 else 1
        else:
            comm_factor = 1
        
        # 5. 处理器负载均衡
        cpu_load = sum(len(cpu.current_tasks) for cpu in cpu_processors) / len(cpu_processors)
        fpga_load = len(fpga_processor.current_tasks) / fpga_processor.resource_limit if fpga_processor.resource_limit else 0

        processor_loads = {
            "CPU": cpu_load,
            "FPGA": fpga_load
        }
        
        # 获取任务分配的处理器类型
        processor_type = task_processor_map.get(task_id)
        if not processor_type:
            print(f"警告: 任务 {task_id} 没有处理器分配")
            return float('-inf')
            
        # 获取对应处理器的负载
        load = processor_loads.get(processor_type, 0)
        
        # 6. 计算优先级得分
        # 权重配置
        w1, w2, w3, w4, w5 = 0.3, 0.2, 0.2, 0.15, 0.15
        
        priority_score = (
            w1 * time_urgency +
            w2 * path_priority +
            w3 * (1 - resource_usage) +
            w4 * comm_factor +
            w5 * (1 - load)
        )
        
        # 添加调试信息
        print(f"任务 {task_id} 优先级计算详情:")
        print(f"  时间紧迫度: {time_urgency:.2f}")
        print(f"  路径优先级: {path_priority:.2f}")
        print(f"  资源利用率: {resource_usage:.2f}")
        print(f"  通信因子: {comm_factor:.2f}")
        print(f"  处理器负载: {load:.2f}")
        print(f"  最终得分: {priority_score:.2f}")
        
        return priority_score
        
    except Exception as e:
        print(f"计算任务 {task_id} 优先级时发生错误: {str(e)}")
        return float('-inf')

#任务批处理和分组优化模块
def optimize_task_groups(dag, task_attributes):
    """优化任务分组
    将通信密集的任务分配到同一处理器，减少通信开销
    """
    def calculate_communication_density(task_id):
        """计算任务的通信密度"""
        try:
            # 注意：task_id 已经是从0开始的索引
            node_id = task_id + 1  # 转换为图中的节点ID(从1开始)
            if not dag.has_node(node_id):
                print(f"警告: 节点 {node_id} 不在图中")
                return 0
                
            # 获取前驱和后继
            successors = list(dag.successors(node_id))
            predecessors = list(dag.predecessors(node_id))
            
            # 计算总通信量
            comm_volume = sum(dag[node_id][succ]['data_size'] for succ in successors)
            comm_volume += sum(dag[pred][node_id]['data_size'] for pred in predecessors)
            
            return comm_volume
        except Exception as e:
            print(f"计算任务 {task_id} 通信密度时出错: {str(e)}")
            return 0

    task_groups = []
    visited = set()
    num_tasks = len(task_attributes)
    
    # 遍历所有任务
    for task_id in range(num_tasks):
        if task_id in visited:
            continue
            
        # 创建新组
        group = {task_id}
        visited.add(task_id)
        
        # 获取当前任务的通信密度
        density = calculate_communication_density(task_id)
        
        # 获取相邻任务(注意节点ID转换)
        node_id = task_id + 1
        if dag.has_node(node_id):
            # 收集所有相邻任务(转换回0基索引)
            neighbors = set(pred-1 for pred in dag.predecessors(node_id)) | \
                       set(succ-1 for succ in dag.successors(node_id))
                       
            # 添加通信密集的相邻任务
            for neighbor in neighbors:
                if neighbor not in visited and \
                   0 <= neighbor < num_tasks and \
                   calculate_communication_density(neighbor) > density * 0.7:
                    group.add(neighbor)
                    visited.add(neighbor)
                
        task_groups.append(group)
        
    return task_groups

# 改进ReTPA调度
def improved_retpa_scheduling(dag, task_attributes, allocation_result, data_transfer_rate):
    """改进的ReTPA调度算法，支持FPGA和GPU并行执行"""
    global task_processor_map  # 声明使用全局变量
    task_processor_map.clear()  # 清空映射
    group_processor_map = {}  # 添加分组处理器映射字典

    print("初始化调度参数...")
    num_tasks = len(task_attributes)
    
    # 添加错误检查
    if not allocation_result:
        raise ValueError("任务分配结果为空")
    
    # 获取任务分组
    task_groups = optimize_task_groups(dag, task_attributes)
    print(f"任务分组结果: {task_groups}")
    
    # 计算组间通信开销
    group_comm_cost = {}
    for i, group1 in enumerate(task_groups):
        for j, group2 in enumerate(task_groups):
            if i != j:
                cost = sum(
                    dag[u+1][v+1]['data_size'] 
                    for u in group1 
                    for v in group2 
                    if dag.has_edge(u+1, v+1)
                )
                if cost > 0:
                    group_comm_cost[(i,j)] = cost

 
    
    # 为每个组分配主要处理器
    for group_id, group in enumerate(task_groups):
        processor_loads = {
            "CPU": 0,
            "FPGA": 0
        }
        
        # 计算每个处理器的组内任务数
        for task_id in group:
            for proc, t_id in allocation_result:
                if t_id == task_id:
                    processor_loads[proc] += 1
                    break
        
        # 选择负载最重的处理器作为组的主要处理器
        group_processor_map[group_id] = max(processor_loads.items(), key=lambda x: x[1])[0]
    
    # 更新任务处理器映射
    for processor, task_id in allocation_result:
        if task_id >= len(task_attributes):
            raise ValueError(f"无效的任务ID: {task_id}")
        task_processor_map[task_id] = processor

    try:
        est, lft = calculate_est_lft(dag, task_attributes)
        task_weights = calculate_task_weight(dag, task_attributes, est, lft)
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        raise

    print("开始任务调度循环...")
    
    # 处理器状态管理类
    class ProcessorStatus:
        def __init__(self, resource_limit=None, processor_type=None):
            self.busy_until = {}
            self.current_tasks = set()
            self.current_groups = set()
            self.resource_limit = resource_limit
            self.current_resource = 0
            self.processor_type = processor_type  # 添加处理器类型标识

            # FPGA特有的资源相关属性
            if processor_type == "FPGA":
                self.resource_limit = FPGA_TOTAL_RESOURCE
                self.current_resource = 0


        def can_accept_task(self, task_resource):
            """根据处理器类型检查资源"""
            if self.processor_type == "CPU":
                # 修改：CPU按照任务数量控制，每个CPU核心同时只能执行一个任务
                return len(self.current_tasks) == 0
            elif self.processor_type == "FPGA":
                # FPGA按照资源总量控制
                return self.current_resource + (task_resource or 0) <= self.resource_limit
            return False

        def add_task(self, task_id, task_resource, group_id, finish_time):
            """添加新任务"""
            self.current_tasks.add(task_id)
            self.current_groups.add(group_id)
            self.busy_until[task_id] = finish_time
            if self.resource_limit is not None:
                self.current_resource += task_resource

        def remove_task(self, task_id, task_resource):
            """移除完成的任务"""
            if task_id in self.current_tasks:
                self.current_tasks.remove(task_id)
                del self.busy_until[task_id]
                if self.resource_limit is not None:
                    self.current_resource -= task_resource

        def update_status(self, current_time):
            """更新处理器状态"""
            completed_tasks = {
                task_id for task_id, finish_time in self.busy_until.items()
                if finish_time <= current_time
            }
            for task_id in completed_tasks:
                self.remove_task(task_id, 
                    task_attributes[task_id]['fpga_resource_requirement'] 
                    if self == fpga_processor else 0
                )
    
    # 初始化处理器
    cpu_processors = [ProcessorStatus(processor_type="CPU") for _ in range(2)]
    fpga_processor = ProcessorStatus(
        resource_limit=FPGA_TOTAL_RESOURCE, 
        processor_type="FPGA"
    )


    schedule = []
    current_time = 0
    ready_tasks = set()
    communication_finish_times = {}
    scheduled_tasks = set()

    while len(schedule) < num_tasks:
        # 更新处理器状态
        for cpu in cpu_processors:
            cpu.update_status(current_time)
        fpga_processor.update_status(current_time)


        # 更新就绪任务集合
        ready_tasks.clear()
        active_groups = set()
        
        # 收集活跃组
        for cpu in cpu_processors:
            active_groups.update(cpu.current_groups)
        active_groups.update(fpga_processor.current_groups)


        # 优先检查活跃组中的任务
        for group_id in active_groups:
            for task_id in task_groups[group_id]:
                if task_id not in scheduled_tasks:
                    if is_task_ready(task_id, schedule, communication_finish_times, 
                                   current_time, task_attributes):
                        ready_tasks.add(task_id)

        # 如果没有活跃组的就绪任务，检查所有任务
        if not ready_tasks:
            for task_id in range(num_tasks):
                if task_id not in scheduled_tasks:
                    if is_task_ready(task_id, schedule, communication_finish_times, 
                                   current_time, task_attributes):
                        ready_tasks.add(task_id)

        if not ready_tasks:
            # 时间推进
            next_times = []
            for cpu in cpu_processors:
                if cpu.busy_until:
                    next_times.extend(cpu.busy_until.values())
            if fpga_processor.busy_until:
                next_times.extend(fpga_processor.busy_until.values())
            next_times.extend(communication_finish_times.values())
            next_times.extend(
                task_attributes[t]['release_time'] 
                for t in range(num_tasks) 
                if t not in scheduled_tasks
            )
            next_times = [t for t in next_times if t > current_time]
            
            if next_times:
                current_time = min(next_times)
                continue
            else:
                break

        # 选择最优任务
        best_task = None
        best_score = float('-inf')
        
        for task_id in ready_tasks:
            priority = calculate_comprehensive_priority(
                task_id, current_time, task_attributes, est, lft,
                task_weights, dag, FPGA_TOTAL_RESOURCE, 
                cpu_processors, fpga_processor, task_processor_map
            )
            
            task_group_id = next(
                (i for i, group in enumerate(task_groups) if task_id in group),
                None
            )
            
            # 组优先级调整
            if task_group_id is not None:
                if task_processor_map[task_id] == group_processor_map[task_group_id]:
                    priority *= 1.2
                if task_group_id in active_groups:
                    priority *= 1.5

            if priority > best_score:
                best_score = priority
                best_task = task_id

        # 调度选中的任务
        processor = task_processor_map[best_task]
        task_group_id = next(i for i, group in enumerate(task_groups) if best_task in group)
        
        if processor == "CPU":
                # 检查是否有空闲的CPU处理器
                available_cpus = [cpu for cpu in cpu_processors if cpu.can_accept_task(None)]
                if not available_cpus:
                    # 如果没有空闲CPU，将任务重新加入就绪队列
                    ready_tasks.add(best_task)
                    # 找到最早空闲的时间点
                    next_time = min(
                        time for cpu in cpu_processors 
                        for time in cpu.busy_until.values()
                        if time > current_time
                    )
                    current_time = next_time
                    continue

                # 选择空闲的CPU处理器
                available_cpu = available_cpus[0]
                exec_time = task_attributes[best_task]['software_execution_time']
                start_time = current_time
                finish_time = start_time + exec_time
                available_cpu.add_task(best_task, None, task_group_id, finish_time)
            
        elif processor == "FPGA":
            resource_req = task_attributes[best_task]['fpga_resource_requirement']
            if fpga_processor.can_accept_task(resource_req):
                start_time = current_time
                exec_time = task_attributes[best_task]['fpga_execution_time']
                finish_time = start_time + exec_time
                fpga_processor.add_task(best_task, resource_req, task_group_id, finish_time)
            else:
                continue
            


        schedule.append((best_task, processor, start_time, finish_time))
        scheduled_tasks.add(best_task)
        ready_tasks.remove(best_task)

        # 更新通信时间
        for succ in dag.successors(best_task + 1):
            succ_id = succ - 1
            if succ_id not in scheduled_tasks:
                data_size = dag[best_task + 1][succ]['data_size']
                # 同组任务通信时间减半
                if any(best_task in group and succ_id in group for group in task_groups):
                    comm_time = data_size / (data_transfer_rate * 2)
                else:
                    comm_time = data_size / data_transfer_rate
                communication_finish_times[succ_id] = finish_time + comm_time

    return sorted(schedule, key=lambda x: x[2]), task_groups
    
# 计算EST和LFT
def calculate_est_lft(dag, task_attributes):
    """计算任务最早开始时间(EST)和最晚完成时间(LFT)"""
    num_tasks = len(task_attributes)
    est = [0] * num_tasks
    lft = [float('inf')] * num_tasks
    
    try:
        # 计算EST（拓扑排序正序）
        for task_id in nx.topological_sort(dag):
            task_idx = task_id - 1  # 转换为0基索引
            predecessors = list(dag.predecessors(task_id))
            if predecessors:
                max_est = max(est[pred - 1] + task_attributes[pred - 1]['software_execution_time'] 
                            for pred in predecessors)
                est[task_idx] = max(max_est, task_attributes[task_idx]['release_time'])
            else:
                est[task_idx] = task_attributes[task_idx]['release_time']
        
        # 计算LFT（拓扑排序逆序）
        reversed_topo = list(reversed(list(nx.topological_sort(dag))))
        for task_id in reversed_topo:
            task_idx = task_id - 1
            successors = list(dag.successors(task_id))
            if successors:
                min_lft = min(lft[succ - 1] for succ in successors)
                lft[task_idx] = min(min_lft, task_attributes[task_idx]['deadline'])
            else:
                lft[task_idx] = task_attributes[task_idx]['deadline']
                
        print(f"EST计算结果: {est}")
        print(f"LFT计算结果: {lft}")
        
    except Exception as e:
        print(f"计算EST/LFT时出错: {str(e)}")
        raise
        
    return est, lft

def is_task_ready(task_id, schedule, communication_finish_times, current_time, task_attributes):
    """判断任务是否就绪"""
    # 检查是否已调度
    if task_id in [s[0] for s in schedule]:
        return False
        
    # 检查释放时间
    if current_time < task_attributes[task_id]['release_time']:
        return False
        
    # 检查节点是否存在于图中
    node_id = task_id + 1
    if not dag.has_node(node_id):
        print(f"警告: 节点 {node_id} 不在图中")
        return False
        
    # 检查前驱任务完成情况
    predecessors = list(dag.predecessors(node_id))
    if not predecessors:  # 没有前驱的任务
        return True
        
    # 检查所有前驱任务是否完成且通信已完成
    for pred in predecessors:
        pred_id = pred - 1
        if pred_id not in [s[0] for s in schedule]:
            return False
        if pred_id in communication_finish_times and communication_finish_times[pred_id] > current_time:
            return False
            
    return True

# ================== 主函数部分 ==================
if __name__ == "__main__":
    try:
        # 1. 读取数据集
        file_path = "/root/123/FFT/dataset/ccr=1.0/dag_fft_5tasks.txt"
        print(f"正在读取数据集：{file_path}")
        dag, task_attributes = read_dataset(file_path)
        
        if not task_attributes:
            raise ValueError("未读取到任何任务信息")
            
        print(f"成功读取 {len(task_attributes)} 个任务")
        
        # 2. 设置资源参数
        fpga_total_resource = 10
        data_transfer_rate = 2
        
        # 3. 执行ILP任务分配
        print("\n开始执行ILP任务分配...")
        allocation_result = ilp_task_allocation(dag, task_attributes, 
                                             fpga_total_resource, 
                                             data_transfer_rate)
        
        if not allocation_result:
            raise ValueError("ILP任务分配失败")
            
        print(f"任务分配完成，共分配 {len(allocation_result)} 个任务")
        
        
        # 4. 执行改进ReTPA调度
        print("\n开始执行改进ReTPA调度...")
        schedule_result, task_groups = improved_retpa_scheduling(
            dag, 
            task_attributes, 
            allocation_result, 
            data_transfer_rate
        )
        
        if not schedule_result:
            raise ValueError("任务调度失败")
            
        print(f"调度完成，共调度 {len(schedule_result)} 个任务")
        print(f"任务分组数量: {len(task_groups)}")
        
        # 5. 输出结果到文件
        output_file = "/root/123/FFT/result/ccr=1.0/RETPA/dag_fft_5tasks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 5.1 写入基本信息
            f.write("================ 调度环境信息 ================\n")
            f.write(f"总任务数: {len(task_attributes)}\n")
            f.write(f"FPGA资源上限: {fpga_total_resource}\n")
            f.write(f"数据传输速率: {data_transfer_rate}\n\n")
            
            # 5.2 写入任务分配结果
            f.write("================ 任务分配结果 ================\n")
            for processor, task_id in sorted(allocation_result, key=lambda x: x[1]):
                f.write(f"任务 {task_id + 1} 分配到 {processor}\n")
            
            # 5.3 写入调度结果
            f.write("\n================ 调度结果 ================\n")
            for task_id, processor, start_time, finish_time in sorted(schedule_result, key=lambda x: x[0]):
                f.write(f"任务 {task_id + 1}:\n")
                f.write(f"  处理器: {processor}\n")
                f.write(f"  开始时间: {start_time:.2f}\n")
                f.write(f"  结束时间: {finish_time:.2f}\n")
                f.write(f"  执行时间: {finish_time - start_time:.2f}\n\n")
            
            # 5.4 写入性能统计信息
            f.write("================ 性能统计 ================\n")
            makespan = max(finish_time for _, _, _, finish_time in schedule_result)
            total_tasks = len(schedule_result)
            cpu_tasks = sum(1 for _, proc, _, _ in schedule_result if proc == "CPU")
            fpga_tasks = sum(1 for _, proc, _, _ in schedule_result if proc == "FPGA")
            gpu_tasks = sum(1 for _, proc, _, _ in schedule_result if proc == "GPU")
            
            f.write(f"调度总时间(makespan): {makespan:.2f}\n")
            f.write(f"平均任务执行时间: {makespan/total_tasks:.2f}\n")
            f.write("\n资源利用率:\n")
            f.write(f"CPU: {cpu_tasks/total_tasks*100:.2f}%\n")
            f.write(f"FPGA: {fpga_tasks/total_tasks*100:.2f}%\n")

            # 5.4 写入分组优化分析
            f.write("\n================ 分组优化分析 ================\n")
            
            # 计算组内通信成本
            intra_group_comm = 0
            inter_group_comm = 0
            for task_id, _, start_time, finish_time in schedule_result:
                successors = list(dag.successors(task_id + 1))
                for succ in successors:
                    succ_id = succ - 1
                    data_size = dag[task_id + 1][succ]['data_size']
                    
                    # 判断是否为组内通信
                    is_intra_group = False
                    for group in task_groups:
                        if task_id in group and succ_id in group:
                            is_intra_group = True
                            break
                    
                    if is_intra_group:
                        intra_group_comm += data_size
                    else:
                        inter_group_comm += data_size
            
            total_comm = intra_group_comm + inter_group_comm
            f.write(f"组内通信总量: {intra_group_comm}\n")
            f.write(f"组间通信总量: {inter_group_comm}\n")
            f.write(f"通信优化比例: {(intra_group_comm/total_comm*100):.2f}%\n\n")
            
            # 分析组调度效果
            group_execution_times = {}
            for group_id, group in enumerate(task_groups):
                group_times = []
                for task_id in group:
                    task_schedule = next((s for s in schedule_result if s[0] == task_id), None)
                    if task_schedule:
                        group_times.append((task_schedule[2], task_schedule[3]))  # start_time, finish_time
                
                if group_times:
                    group_start = min(t[0] for t in group_times)
                    group_finish = max(t[1] for t in group_times)
                    group_execution_times[group_id] = (group_start, group_finish)
                    
                    f.write(f"组 {group_id} 执行情况:\n")
                    f.write(f"  任务数量: {len(group)}\n")
                    f.write(f"  开始时间: {group_start:.2f}\n")
                    f.write(f"  结束时间: {group_finish:.2f}\n")
                    f.write(f"  执行时间: {group_finish - group_start:.2f}\n")
                    f.write(f"  处理器分布: {[task_processor_map[task_id] for task_id in group]}\n\n")
            
            # 5.5 写入优化效果分析
            f.write("\n================ 优化效果分析 ================\n")
            
            # 计算负载均衡度
            processor_times = {
                "CPU": [],
                "FPGA": []
            }
            
            for task_id, processor, start_time, finish_time in schedule_result:
                processor_times[processor].append(finish_time - start_time)
            
            for proc, times in processor_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    max_time = max(times)
                    min_time = min(times)
                    f.write(f"\n{proc} 处理器分析:\n")
                    f.write(f"  平均执行时间: {avg_time:.2f}\n")
                    f.write(f"  最长执行时间: {max_time:.2f}\n")
                    f.write(f"  最短执行时间: {min_time:.2f}\n")
                    f.write(f"  任务数量: {len(times)}\n")
                    f.write(f"  利用率: {sum(times)/makespan*100:.2f}%\n")
            
            # 计算关键路径上的任务
            critical_path = []
            current_task = max(schedule_result, key=lambda x: x[3])[0]  # 结束时间最晚的任务
            while True:
                critical_path.append(current_task)
                predecessors = list(dag.predecessors(current_task + 1))
                if not predecessors:
                    break
                current_task = max(
                    (p-1 for p in predecessors),
                    key=lambda x: next(s[3] for s in schedule_result if s[0] == x)
                )
            
            critical_path.reverse()
            f.write("\n关键路径分析:\n")
            f.write(f"关键路径任务序列: {[t+1 for t in critical_path]}\n")
            f.write(f"关键路径长度: {len(critical_path)}\n")
            
            # 计算并行度
            timeline = []
            for task_id, processor, start_time, finish_time in schedule_result:
                timeline.extend([(start_time, 1), (finish_time, -1)])
            
            timeline.sort()
            current_parallel = 0
            max_parallel = 0
            total_parallel = 0
            last_time = timeline[0][0]
            
            for time, change in timeline:
                if time != last_time:
                    total_parallel += current_parallel * (time - last_time)
                    last_time = time
                current_parallel += change
                max_parallel = max(max_parallel, current_parallel)
            
            avg_parallel = total_parallel / makespan
            f.write(f"\n并行度分析:\n")
            f.write(f"最大并行度: {max_parallel}\n")
            f.write(f"平均并行度: {avg_parallel:.2f}\n")

# 在输出部分添加以下代码

            # 计算MRR
            # 1. 获取当前算法的makespan (T)
            T = max(finish_time for _, _, _, finish_time in schedule_result)
            
            # 2. 计算单处理器串行执行时间 (Tmax)
            def calculate_single_processor_time():
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
                            comm_time = dag[pred][task_id]['data_size'] / data_transfer_rate
                            start_time = max(start_time, task_finish_times[pred_idx] + comm_time)
                    
                    # 使用软件执行时间
                    exec_time = task_attributes[task_idx]['software_execution_time']
                    finish_time = start_time + exec_time
                    
                    task_finish_times[task_idx] = finish_time
                    current_time = finish_time
                
                return max(task_finish_times.values())
            
            Tmax = calculate_single_processor_time()
            
            # 3. 计算MRR
            MRR = T / Tmax
            
            # 在性能统计部分添加MRR输出
            f.write("\n================ MRR分析 ================\n")
            f.write(f"当前调度makespan (T): {T:.2f}\n")
            f.write(f"考虑依赖的串行执行时间 (Tmax): {Tmax:.2f}\n")
            f.write(f"MRR (T/Tmax): {MRR:.4f}\n")
            f.write(f"加速比 (Tmax/T): {1/MRR:.4f}\n")
            
            # 在控制台也输出MRR信息
            print(f"\nMRR分析:")
            print(f"当前调度makespan (T): {T:.2f}")
            print(f"考虑依赖的串行执行时间 (Tmax): {Tmax:.2f}")
            print(f"MRR (T/Tmax): {MRR:.4f}")
            print(f"加速比 (Tmax/T): {1/MRR:.4f}")
        
        # 6. 在控制台显示结果
        print(f"\n结果已保存到文件: {output_file}")
        print("\n文件内容:")
        with open(output_file, 'r', encoding='utf-8') as f:
            print(f.read())
            
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()