import pulp
import networkx as nx
import numpy as np

FPGA_TOTAL_RESOURCE = 10
GPU_TOTAL_RESOURCE = 6
DATA_TRANSFER_RATE = 2
# 新增：CPU处理器数量与总设备单元数（2个CPU并行单元 + 1 FPGA + 1 GPU）
CPU_PROCESSOR_COUNT = 2
TOTAL_PROCESSOR_UNITS = CPU_PROCESSOR_COUNT + 2

# 新增：自校准权重的辅助函数
# 说明：在 DAG.graph 中缓存统计量，避免重复计算；compute_dynamic_weights 使用 softmax(β·ρ)
# ρ 向量由五个“场景压力”组成：截止期紧迫、路径深度压力、资源稀缺、通信压力、运行时负载。
def _init_dynamic_stats_if_needed(dag: nx.DiGraph, task_attributes):
    try:
        # 最大出度
        if 'max_outdeg' not in dag.graph:
            if dag.number_of_nodes() > 0:
                dag.graph['max_outdeg'] = max((dag.out_degree(n) for n in dag.nodes()), default=1) or 1
            else:
                dag.graph['max_outdeg'] = 1
        # 最大“单节点外发通信总量”（用于归一化通信因子）
        if 'max_comm' not in dag.graph:
            max_comm = 1
            for n in dag.nodes():
                out_sum = 0
                for _, v, data in dag.out_edges(n, data=True):
                    out_sum += data.get('data_size', 0)
                if out_sum > max_comm:
                    max_comm = out_sum
            dag.graph['max_comm'] = max_comm
        # CCR（平均通信量/平均计算量）
        if 'ccr' not in dag.graph:
            avg_comm = 0.0
            m = dag.number_of_edges()
            if m > 0:
                avg_comm = sum((data.get('data_size', 0) for _, _, data in dag.edges(data=True)), 0.0) / m
            n = len(task_attributes) or 1
            avg_comp = sum((t.get('software_execution_time', 1) for t in task_attributes), 0.0) / max(n, 1)
            dag.graph['ccr'] = (avg_comm / avg_comp) if avg_comp > 0 else 0.0
        # 近似图深度
        if 'depth' not in dag.graph:
            try:
                dag.graph['depth'] = nx.dag_longest_path_length(dag) if dag.number_of_nodes() > 0 else 1
            except Exception:
                dag.graph['depth'] = 1
    except Exception:
        # 任何异常下，给出保守默认
        dag.graph.setdefault('max_outdeg', 1)
        dag.graph.setdefault('max_comm', 1)
        dag.graph.setdefault('ccr', 0.0)
        dag.graph.setdefault('depth', 1)


def compute_dynamic_weights(ct, dag: nx.DiGraph, task_attributes, est, lft,
                             cpu_processors, fpga_processor, gpu_processor, beta: float = 1.0):
    """依据场景压力自校准权重 w = softmax(β·ρ)。
    ρ = [ρ_deadline, ρ_path, ρ_res, ρ_comm, ρ_load] 各分量∈[0,1]，越大表示该因素更“吃紧”。
    返回 (w1..w5)。
    """
    _init_dynamic_stats_if_needed(dag, task_attributes)

    # ρ1: 截止期紧迫（以全局平均归一化松弛度的反值表示）
    try:
        n = len(task_attributes) or 1
        slacks = []
        for i in range(n):
            window = max(lft[i] - est[i], 1)
            aed = max(lft[i] - max(est[i], ct), 0)
            slacks.append(aed / window)
        rho_deadline = 1.0 - float(np.clip(np.mean(slacks), 0.0, 1.0))
    except Exception:
        rho_deadline = 0.5

    # ρ2: 路径深度压力（图深度/任务数，越深越“串行化”）
    try:
        depth = max(dag.graph.get('depth', 1), 1)
        n_tasks = max(dag.number_of_nodes(), 1)
        rho_path = float(np.clip(depth / n_tasks, 0.0, 1.0))
    except Exception:
        rho_path = 0.3

    # ρ3: 资源稀缺（平均需求/总资源），FPGA 与 GPU 各占一半
    try:
        n = len(task_attributes) or 1
        avg_fpga_need = sum((t.get('fpga_resource_requirement', 0) for t in task_attributes), 0.0) / n
        avg_gpu_need = sum((t.get('gpu_resource_requirement', 0) for t in task_attributes), 0.0) / n
        r_fpga = avg_fpga_need / max(FPGA_TOTAL_RESOURCE, 1)
        r_gpu  = avg_gpu_need / max(GPU_TOTAL_RESOURCE, 1)
        rho_res = float(np.clip(0.5 * (r_fpga + r_gpu), 0.0, 1.0))
    except Exception:
        rho_res = 0.4

    # ρ4: 通信压力（CCR 归一化）
    try:
        ccr = dag.graph.get('ccr', 0.0)
        rho_comm = float(np.clip(ccr / (1.0 + ccr), 0.0, 1.0))
    except Exception:
        rho_comm = 0.5

    # ρ5: 运行时负载（按当前各设备利用率加权）
    try:
        cpu_load = 0.0
        if cpu_processors:
            cpu_load = sum(len(getattr(cpu, 'current_tasks', [])) for cpu in cpu_processors) / max(len(cpu_processors), 1)
        fpga_cap = max(getattr(fpga_processor, 'resource_limit', 1), 1)
        gpu_cap  = max(getattr(gpu_processor,  'resource_limit', 1), 1)
        fpga_load = len(getattr(fpga_processor, 'current_tasks', [])) / fpga_cap
        gpu_load  = len(getattr(gpu_processor,  'current_tasks', [])) / gpu_cap
        rho_load = float(np.clip(0.34 * cpu_load + 0.33 * fpga_load + 0.33 * gpu_load, 0.0, 1.0))
    except Exception:
        rho_load = 0.3

    rhos = np.array([rho_deadline, rho_path, rho_res, rho_comm, rho_load], dtype=float)
    # softmax(β·ρ)
    exps = np.exp(np.clip(beta, 0.0, 5.0) * rhos)
    weights = exps / np.sum(exps) if np.isfinite(exps).all() and np.sum(exps) > 0 else np.ones_like(exps) / len(exps)
    return float(weights[0]), float(weights[1]), float(weights[2]), float(weights[3]), float(weights[4])

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
                        
    # 添加调试信息
    print(f"读取到的任务数量: {len(task_attributes)}")
    print(f"读取到的边数量: {len(dag.edges())}")
    for i, attr in enumerate(task_attributes):
        print(f"任务 {i+1} 属性: {attr}")
        
    return dag, task_attributes

# ================== ILP任务分配部分 ==================
def ilp_task_allocation(dag, task_attributes, fpga_total_resource, gpu_total_resource, data_transfer_rate):
    """使用ILP进行任务分配，平衡执行时间、通信时间、处理器负载和异构特性
    优化点：
    - 通信变量压缩（每边6个->3个same变量）
    - 不可能分配预固定（资源/时间窗剪枝）
    - 启发式初始解（MIP Start）
    - CBC求解器时间限制与gap，支持大规模实例
    - 非最优也导出可行解，必要时回退启发式
    """
    num_tasks = len(task_attributes)
    model = pulp.LpProblem("Task_Allocation_Problem", pulp.LpMinimize)

    # 定义任务分配变量（one-hot）
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # CPU分配
    y = {i: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # FPGA分配
    z = {i: pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # GPU分配

    # 负载均衡辅助变量
    cpu_dev_plus = pulp.LpVariable("cpu_dev_plus", lowBound=0)
    cpu_dev_minus = pulp.LpVariable("cpu_dev_minus", lowBound=0)
    fpga_dev_plus = pulp.LpVariable("fpga_dev_plus", lowBound=0)
    fpga_dev_minus = pulp.LpVariable("fpga_dev_minus", lowBound=0)
    gpu_dev_plus = pulp.LpVariable("gpu_dev_plus", lowBound=0)
    gpu_dev_minus = pulp.LpVariable("gpu_dev_minus", lowBound=0)

    # 通信变量压缩：same_C/F/G 表示边(u,v)两端是否在同一处理器
    same_C, same_F, same_G = {}, {}, {}
    for u, v, _ in dag.edges(data=True):
        u0, v0 = u - 1, v - 1  # 0-based
        same_C[(u0, v0)] = pulp.LpVariable(f"sameC_{u0}_{v0}", cat=pulp.LpBinary)
        same_F[(u0, v0)] = pulp.LpVariable(f"sameF_{u0}_{v0}", cat=pulp.LpBinary)
        same_G[(u0, v0)] = pulp.LpVariable(f"sameG_{u0}_{v0}", cat=pulp.LpBinary)
        # 线性化 same = and(assign_u_p, assign_v_p)
        model += same_C[(u0, v0)] <= x[u0]
        model += same_C[(u0, v0)] <= x[v0]
        model += same_C[(u0, v0)] >= x[u0] + x[v0] - 1

        model += same_F[(u0, v0)] <= y[u0]
        model += same_F[(u0, v0)] <= y[v0]
        model += same_F[(u0, v0)] >= y[u0] + y[v0] - 1

        model += same_G[(u0, v0)] <= z[u0]
        model += same_G[(u0, v0)] <= z[v0]
        model += same_G[(u0, v0)] >= z[u0] + z[v0] - 1

    # 构建目标函数的各个部分
    # 1. 执行时间部分
    execution_time = pulp.lpSum(
        x[i] * task_attributes[i]['software_execution_time'] +
        y[i] * task_attributes[i]['fpga_execution_time'] +
        z[i] * task_attributes[i]['gpu_execution_time']
        for i in range(num_tasks)
    )

    # 2. 通信时间部分（跨处理器通信才计入成本）
    communication_time = pulp.lpSum(
        (data['data_size'] / data_transfer_rate) * (
            1 - (same_C[(u-1, v-1)] + same_F[(u-1, v-1)] + same_G[(u-1, v-1)])
        )
        for u, v, data in dag.edges(data=True)
    )

    # 3. 负载均衡部分
    cpu_load = pulp.lpSum(x[i] for i in range(num_tasks))
    fpga_load = pulp.lpSum(y[i] for i in range(num_tasks))
    gpu_load = pulp.lpSum(z[i] for i in range(num_tasks))
    ideal_load = num_tasks / 3
    model += cpu_load - ideal_load == cpu_dev_plus - cpu_dev_minus
    model += fpga_load - ideal_load == fpga_dev_plus - fpga_dev_minus
    model += gpu_load - ideal_load == gpu_dev_plus - gpu_dev_minus
    load_balance_penalty = (
        cpu_dev_plus + cpu_dev_minus +
        fpga_dev_plus + fpga_dev_minus +
        gpu_dev_plus + gpu_dev_minus
    )

    # 4. 异构性优化部分
    heterogeneity_reward = pulp.lpSum(
        y[i] * (1 - task_attributes[i]['fpga_execution_time'] / task_attributes[i]['software_execution_time']) +
        z[i] * (1 - task_attributes[i]['gpu_execution_time'] / task_attributes[i]['software_execution_time'])
        for i in range(num_tasks)
    )

    # 根据CCR动态调整权重
    avg_comm_cost = sum(data['data_size'] for _, _, data in dag.edges(data=True)) / len(dag.edges()) if dag.edges() else 0
    avg_comp_cost = sum(task_attributes[i]['software_execution_time'] for i in range(num_tasks)) / num_tasks if num_tasks else 1
    ccr = avg_comm_cost / avg_comp_cost if avg_comp_cost > 0 else 0
    w1 = 1.0
    w2 = 1.0 / (1 + ccr)
    w3 = 0.5 * (1 + 1 / (1 + ccr))
    w4 = 0.3 * (1 + 1 / (1 + ccr))

    # 组合目标函数
    model += (
        w1 * execution_time +
        w2 * communication_time +
        w3 * load_balance_penalty -
        w4 * heterogeneity_reward
    ), "Total_Cost"

    # 约束条件
    # 1) 每个任务只能分配到一个处理器（one-hot）
    for i in range(num_tasks):
        model += x[i] + y[i] + z[i] == 1

    # 2) 资源总量约束
    model += pulp.lpSum(y[i] * task_attributes[i]['fpga_resource_requirement'] for i in range(num_tasks)) <= fpga_total_resource
    model += pulp.lpSum(z[i] * task_attributes[i]['gpu_resource_requirement'] for i in range(num_tasks)) <= gpu_total_resource

    # 3) 最小分配约束（适度放宽，避免大实例不可行）
    min_tasks_per_processor = 0 if num_tasks >= 80 else max(1, int(num_tasks * 0.05))  # 大实例放宽为0
    model += cpu_load >= min_tasks_per_processor
    model += fpga_load >= min_tasks_per_processor
    model += gpu_load >= min_tasks_per_processor

    # 4) 不可能分配的预固定（剪枝）：资源超限或时间窗不满足时禁用对应设备
    for i in range(num_tasks):
        # 资源剪枝
        if task_attributes[i]['fpga_resource_requirement'] > fpga_total_resource:
            model += y[i] == 0
        if task_attributes[i]['gpu_resource_requirement'] > gpu_total_resource:
            model += z[i] == 0
        # 时间窗剪枝（保守：仅禁用FPGA/GPU，CPU保留以确保可行解）
        window = max(0, task_attributes[i]['deadline'] - task_attributes[i]['release_time'])
        if task_attributes[i]['fpga_execution_time'] > window:
            model += y[i] == 0
        if task_attributes[i]['gpu_execution_time'] > window:
            model += z[i] == 0

    # 启发式初始解（MIP Start）：最快且可行的设备优先，受资源上限约束
    def build_heuristic_initial():
        hx = [0] * num_tasks
        hy = [0] * num_tasks
        hz = [0] * num_tasks
        fpga_left = fpga_total_resource
        gpu_left = gpu_total_resource
        # 简单策略：按(最短执行时间)排序尝试分配
        order = sorted(range(num_tasks), key=lambda i: min(
            task_attributes[i]['software_execution_time'],
            task_attributes[i]['fpga_execution_time'],
            task_attributes[i]['gpu_execution_time']
        ))
        for i in order:
            # 候选列表 (device, exec_time)
            candidates = [
                ("CPU", task_attributes[i]['software_execution_time']),
                ("FPGA", task_attributes[i]['fpga_execution_time']),
                ("GPU", task_attributes[i]['gpu_execution_time'])
            ]
            candidates.sort(key=lambda t: t[1])
            placed = False
            for dev, _ in candidates:
                if dev == "FPGA":
                    need = task_attributes[i]['fpga_resource_requirement']
                    if need <= fpga_left and task_attributes[i]['fpga_execution_time'] <= max(0, task_attributes[i]['deadline'] - task_attributes[i]['release_time']):
                        hy[i] = 1; fpga_left -= need; placed = True; break
                elif dev == "GPU":
                    need = task_attributes[i]['gpu_resource_requirement']
                    if need <= gpu_left and task_attributes[i]['gpu_execution_time'] <= max(0, task_attributes[i]['deadline'] - task_attributes[i]['release_time']):
                        hz[i] = 1; gpu_left -= need; placed = True; break
                else:
                    # CPU 无资源上限，时间窗不剪枝
                    hx[i] = 1; placed = True; break
            if not placed:
                # 最后兜底到CPU
                hx[i] = 1
        return hx, hy, hz

    hx, hy, hz = build_heuristic_initial()
    for i in range(num_tasks):
        try:
            x[i].setInitialValue(hx[i])
            y[i].setInitialValue(hy[i])
            z[i].setInitialValue(hz[i])
        except Exception:
            pass

    # 求解（设置时间/Gap限制，便于大规模实例）
    try:
        solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=180, gapRel=0.02)
    except TypeError:
        try:
            solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=180)
        except TypeError:
            solver = pulp.PULP_CBC_CMD(msg=1)
    status = model.solve(solver)

    # 提取结果（允许非最优但可行解）；若无解则退化到启发式
    allocation_result = []
    def extract_from_vars():
        res = []
        for i in range(num_tasks):
            xv = x[i].value(); yv = y[i].value(); zv = z[i].value()
            if xv is None and yv is None and zv is None:
                # 无值，使用启发式
                if hx[i] == 1:
                    res.append(("CPU", i))
                elif hy[i] == 1:
                    res.append(("FPGA", i))
                else:
                    res.append(("GPU", i))
                continue
            # 选取最大值对应的设备（容错处理浮点）
            vals = [("CPU", xv or 0), ("FPGA", yv or 0), ("GPU", zv or 0)]
            dev = max(vals, key=lambda t: t[1])[0]
            res.append((dev, i))
        return res

    try:
        # 优先使用求解器返回
        allocation_result = extract_from_vars()
        if pulp.LpStatus[model.status] not in ("Optimal", "Not Solved", "Infeasible", "Undefined", "Unbounded"):
            print(f"ILP状态: {pulp.LpStatus[model.status]}")
    except Exception:
        # 回退到启发式
        allocation_result = [("CPU", i) if hx[i] else ("FPGA", i) if hy[i] else ("GPU", i) for i in range(num_tasks)]

    # 打印分配结果统计
    cpu_count = sum(1 for proc, _ in allocation_result if proc == "CPU")
    fpga_count = sum(1 for proc, _ in allocation_result if proc == "FPGA")
    gpu_count = sum(1 for proc, _ in allocation_result if proc == "GPU")
    print(f"\n任务分配统计:")
    print(f"CPU任务数: {cpu_count} ({cpu_count/num_tasks*100:.1f}%)")
    print(f"FPGA任务数: {fpga_count} ({fpga_count/num_tasks*100:.1f}%)")
    print(f"GPU任务数: {gpu_count} ({gpu_count/num_tasks*100:.1f}%)")

    return allocation_result

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
                task_attributes[task_id]['fpga_execution_time'],
                task_attributes[task_id]['gpu_execution_time']
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
            task_attributes[task_id]['fpga_execution_time'],
            task_attributes[task_id]['gpu_execution_time']
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
                                  fpga_total_resource, gpu_total_resource, cpu_processors, 
                                  fpga_processor, gpu_processor, task_processor_map):  # 添加参数
    """计算任务的综合优先级（自校准权重版）"""
    try:
        # 1. 时间紧迫度
        aed = lft[task_id] - max(est[task_id], ct)
        time_urgency = 1 / aed if aed > 0 else float('inf')
        
        # 2. 关键路径优先级（本任务出度/图最大出度）
        successors = list(dag.successors(task_id + 1))
        _init_dynamic_stats_if_needed(dag, task_attributes)
        max_out = max(dag.graph.get('max_outdeg', 1), 1)
        path_priority = (len(successors) / max_out) if max_out > 0 else 0.0
        
        # 3. 资源利用率（小为佳，故聚合时使用 1 - resource_usage）
        resource_usage = min(
            task_attributes[task_id]['fpga_resource_requirement'] / max(fpga_total_resource, 1),
            task_attributes[task_id]['gpu_resource_requirement'] / max(gpu_total_resource, 1)
        )
        
        # 4. 通信因子（越大越好）：1 - (本任务出边总数据量/max_comm)
        max_comm = dag.graph.get('max_comm', 1)
        comm_cost = sum(dag[task_id + 1][succ]['data_size'] for succ in successors) if successors else 0
        comm_factor = 1 - (comm_cost / max_comm) if max_comm > 0 else 1.0
        
        # 5. 处理器负载（按任务已分配处理器类型，越低越好）
        cpu_load = sum(len(cpu.current_tasks) for cpu in cpu_processors) / max(len(cpu_processors), 1)
        fpga_load = (len(fpga_processor.current_tasks) / max(getattr(fpga_processor, 'resource_limit', 1), 1)) if getattr(fpga_processor, 'resource_limit', None) else 0.0
        gpu_load  = (len(gpu_processor.current_tasks)  / max(getattr(gpu_processor, 'resource_limit', 1), 1)) if getattr(gpu_processor, 'resource_limit', None) else 0.0
        processor_loads = {"CPU": cpu_load, "FPGA": fpga_load, "GPU": gpu_load}
        processor_type = task_processor_map.get(task_id)
        if not processor_type:
            return float('-inf')
        load = processor_loads.get(processor_type, 0.0)
        
        # 自校准权重（softmax(β·ρ)）
        w1, w2, w3, w4, w5 = compute_dynamic_weights(ct, dag, task_attributes, est, lft,
                                                      cpu_processors, fpga_processor, gpu_processor, beta=1.0)
        
        # 聚合优先级
        priority_score = (
            w1 * time_urgency +
            w2 * path_priority +
            w3 * (1 - resource_usage) +
            w4 * comm_factor +
            w5 * (1 - load)
        )
        
        # 调试打印可保留/可关闭
        print(f"任务 {task_id} 优先级计算(自校准): SU={time_urgency:.3f}, C={path_priority:.3f}, 1-R={1-resource_usage:.3f}, 1-M={comm_factor:.3f}, 1-L={1-load:.3f}; w={w1:.2f},{w2:.2f},{w3:.2f},{w4:.2f},{w5:.2f}; score={priority_score:.3f}")
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
        # 确保任务ID在图中存在
        node_id = task_id + 1  # 转换为1-based索引
        if not dag.has_node(node_id):
            return 0
            
        successors = list(dag.successors(node_id))
        predecessors = list(dag.predecessors(node_id))
        
        # 计算通信量
        comm_volume = 0
        for succ in successors:
            comm_volume += dag[node_id][succ]['data_size']
        for pred in predecessors:
            comm_volume += dag[pred][node_id]['data_size']
        return comm_volume

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
        
        # 查找相关任务
        density = calculate_communication_density(task_id)
        
        # 获取相邻任务（注意索引转换）
        node_id = task_id + 1
        if dag.has_node(node_id):
            neighbors = set()
            for pred in dag.predecessors(node_id):
                neighbors.add(pred - 1)  # 转换回0-based索引
            for succ in dag.successors(node_id):
                neighbors.add(succ - 1)  # 转换回0-based索引
            
            # 添加通信密集的相邻任务
            for neighbor in neighbors:
                if neighbor not in visited and neighbor < num_tasks:
                    neighbor_density = calculate_communication_density(neighbor)
                    if neighbor_density > density * 0.7:
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
            "FPGA": 0,
            "GPU": 0
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

        def can_accept_task(self, task_resource):
            """根据处理器类型检查资源"""
            if self.processor_type == "CPU":
                # 修改：检查CPU是否空闲
                return len(self.current_tasks) == 0
            elif self.processor_type in ("FPGA", "GPU"):
                # 检查是否有足够资源
                return self.current_resource + task_resource <= self.resource_limit
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
                    if self == fpga_processor else
                    task_attributes[task_id]['gpu_resource_requirement']
                    if self == gpu_processor else 0
                )
    
    # 初始化处理器
    cpu_processors = [ProcessorStatus(processor_type="CPU") for _ in range(CPU_PROCESSOR_COUNT)]
    fpga_processor = ProcessorStatus(
        resource_limit=FPGA_TOTAL_RESOURCE,
        processor_type="FPGA"
    )
    gpu_processor = ProcessorStatus(
        resource_limit=GPU_TOTAL_RESOURCE,
        processor_type="GPU"
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
        gpu_processor.update_status(current_time)

        # 更新就绪任务集合
        ready_tasks.clear()
        active_groups = set()
        
        # 收集活跃组
        for cpu in cpu_processors:
            active_groups.update(cpu.current_groups)
        active_groups.update(fpga_processor.current_groups)
        active_groups.update(gpu_processor.current_groups)

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

        # 修改时间推进部分
        if not ready_tasks:
            # 收集所有未来的时间点
            next_times = []
            for cpu in cpu_processors:
                if cpu.busy_until:
                    next_times.extend(cpu.busy_until.values())
            if fpga_processor.busy_until:
                next_times.extend(fpga_processor.busy_until.values())
            if gpu_processor.busy_until:
                next_times.extend(gpu_processor.busy_until.values())
            if communication_finish_times:
                next_times.extend(communication_finish_times.values())
    
            # 添加未调度任务的释放时间
            next_times.extend(
                task_attributes[t]['release_time'] 
                for t in range(num_tasks) 
                if t not in scheduled_tasks and 
                task_attributes[t]['release_time'] > current_time
            )
    
            # 过滤掉当前时间及之前的时间点
            next_times = [t for t in next_times if t > current_time]
    
            if next_times:
                current_time = min(next_times)
                continue
            else:
                # 如果没有未来时间点且任务未完成，说明存在死锁
                print("警告：调度过程可能存在死锁")
                break

        # 选择最优任务
        best_task = None
        best_score = float('-inf')
        best_tuple = (float('inf'), float('-inf'))  # (EFT, priority)
        
        for task_id in ready_tasks:
            priority = calculate_comprehensive_priority(
                task_id, current_time, task_attributes, est, lft,
                task_weights, dag, FPGA_TOTAL_RESOURCE, GPU_TOTAL_RESOURCE,
                cpu_processors, fpga_processor, gpu_processor, task_processor_map
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

            # 计算若“现在开始”的最早完成时间（EFT），仅考虑可立即开工的任务
            proc = task_processor_map[task_id]
            feasible_now = False
            exec_time = None
            if proc == "CPU":
                feasible_now = any(cpu.can_accept_task(None) for cpu in cpu_processors)
                if feasible_now:
                    exec_time = task_attributes[task_id]['software_execution_time']
            elif proc == "FPGA":
                req = task_attributes[task_id]['fpga_resource_requirement']
                feasible_now = fpga_processor.can_accept_task(req)
                if feasible_now:
                    exec_time = task_attributes[task_id]['fpga_execution_time']
            else:  # GPU
                req = task_attributes[task_id]['gpu_resource_requirement']
                feasible_now = gpu_processor.can_accept_task(req)
                if feasible_now:
                    exec_time = task_attributes[task_id]['gpu_execution_time']

            if feasible_now and exec_time is not None:
                eft = current_time + exec_time
                cand = (eft, priority)
                if cand < best_tuple:
                    best_tuple = cand
                    best_task = task_id
                    best_score = priority

        # 若没有任务可立即开工，退回到“最高优先级”策略，保持原逻辑
        if best_task is None and ready_tasks:
            for task_id in ready_tasks:
                priority = calculate_comprehensive_priority(
                    task_id, current_time, task_attributes, est, lft,
                    task_weights, dag, FPGA_TOTAL_RESOURCE, GPU_TOTAL_RESOURCE,
                    cpu_processors, fpga_processor, gpu_processor, task_processor_map
                )
                task_group_id = next((i for i, group in enumerate(task_groups) if task_id in group), None)
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
                ready_tasks.add(best_task)
                current_time = min(
                    time for cpu in cpu_processors 
                    for time in cpu.busy_until.values()
                    if time > current_time
                )
                continue
        
            # 选择空闲的CPU处理器
            available_cpu = available_cpus[0]  # 选择第一个空闲的CPU
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
            
        else:  # GPU
            resource_req = task_attributes[best_task]['gpu_resource_requirement']
            if gpu_processor.can_accept_task(resource_req):
                start_time = current_time
                exec_time = task_attributes[best_task]['gpu_execution_time']
                finish_time = start_time + exec_time
                gpu_processor.add_task(best_task, resource_req, task_group_id, finish_time)
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
        
    # 检查前驱任务完成情况
    node_id = task_id + 1  # 转换为1-based索引
    if not dag.has_node(node_id):  # 添加节点检查
        print(f"警告: 节点 {node_id} 不在图中")
        return False
        
    predecessors = list(dag.predecessors(node_id))
    if not predecessors:  # 没有前驱的任务
        return True
        
    # 检查所有前驱任务是否完成且通信已完成
    for pred in predecessors:
        pred_id = pred - 1  # 转换回0-based索引
        if pred_id not in [s[0] for s in schedule]:
            return False
        if pred_id in communication_finish_times and communication_finish_times[pred_id] > current_time:
            return False
            
    return True

# ================== 主函数部分 ==================
if __name__ == "__main__":
    try:
        # 1. 读取数据集
        file_path = "/root/123/FFT/dataset/ccr=1.0/dag_fft_223tasks.txt"
        print(f"正在读取数据集：{file_path}")
        dag, task_attributes = read_dataset(file_path)
        
        if not task_attributes:
            raise ValueError("未读取到任何任务信息")
            
        print(f"成功读取 {len(task_attributes)} 个任务")
        
        # 2. 设置资源参数
        fpga_total_resource = 10
        gpu_total_resource = 6
        data_transfer_rate = 2
        
        # 3. 执行ILP任务分配
        print("\n开始执行ILP任务分配...")
        allocation_result = ilp_task_allocation(dag, task_attributes, 
                                             fpga_total_resource, 
                                             gpu_total_resource, 
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
        output_file = "/root/123/FFT/result/ccr=1.0/HPRSA/dag_fft_223tasks.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            # 5.1 写入基本信息
            f.write("================ 调度环境信息 ================\n")
            f.write(f"总任务数: {len(task_attributes)}\n")
            f.write(f"FPGA资源上限: {fpga_total_resource}\n")
            f.write(f"GPU资源上限: {gpu_total_resource}\n")
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
            f.write(f"GPU: {gpu_tasks/total_tasks*100:.2f}%\n")

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
                "FPGA": [],
                "GPU": []
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
            # 修正效率指标（使用全局设备单元数）
            try:
                f.write(f"效率 (T/(Tmax*处理器数量)): {T/(Tmax*TOTAL_PROCESSOR_UNITS):.4f}\n")
                print(f"效率 (T/(Tmax*处理器数量)): {T/(Tmax*TOTAL_PROCESSOR_UNITS):.4f}")
            except Exception:
                pass
        
        # 6. 在控制台显示结果
        print(f"\n结果已保存到文件: {output_file}")
        print("\n文件内容:")
        with open(output_file, 'r', encoding='utf-8') as f:
            print(f.read())
            
    except Exception as e:
        print(f"\n程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()