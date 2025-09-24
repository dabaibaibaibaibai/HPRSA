import networkx as nx
import pulp

# 定义常量
GPU_TOTAL_RESOURCE = 6
DATA_TRANSFER_RATE = 2

# 数据集读取函数
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
                          'gpu_execution_time', 'gpu_resource_requirement']:
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

# LP任务分配函数
def lp_task_allocation(dag, task_attributes):
    """使用线性规划进行任务分配"""
    num_tasks = len(task_attributes)
    model = pulp.LpProblem("Task_Allocation", pulp.LpMinimize)

    # 定义变量
    x = {i: pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # CPU分配
    z = {i: pulp.LpVariable(f"z_{i}", cat=pulp.LpBinary) for i in range(num_tasks)}  # GPU分配

    # 构建目标函数
    execution_time = pulp.lpSum(
        x[i] * task_attributes[i]['software_execution_time'] + 
        z[i] * task_attributes[i]['gpu_execution_time']
        for i in range(num_tasks)
    )

    # 修复通信时间的计算逻辑
    # 为通信引入辅助变量
    comm_xz = {}
    comm_zx = {}

    for u, v, _ in dag.edges(data=True):
        u, v = u-1, v-1  # 转换为0基索引
        comm_xz[(u, v)] = pulp.LpVariable(f"comm_xz_{u}_{v}", cat=pulp.LpBinary)
        comm_zx[(u, v)] = pulp.LpVariable(f"comm_zx_{u}_{v}", cat=pulp.LpBinary)

        # 添加通信变量约束
        model += comm_xz[(u, v)] <= x[u]
        model += comm_xz[(u, v)] <= z[v]
        model += comm_xz[(u, v)] >= x[u] + z[v] - 1

        model += comm_zx[(u, v)] <= z[u]
        model += comm_zx[(u, v)] <= x[v]
        model += comm_zx[(u, v)] >= z[u] + x[v] - 1

    # 构建通信时间部分
    communication_time = pulp.lpSum(
        (data['data_size'] / DATA_TRANSFER_RATE) * (comm_xz[(u-1, v-1)] + comm_zx[(u-1, v-1)])
        for u, v, data in dag.edges(data=True)
    )

    model += execution_time + communication_time, "Total_Time"

    # 添加约束
    for i in range(num_tasks):
        model += x[i] + z[i] == 1  # 每个任务只能分配到一个处理器

    model += pulp.lpSum(z[i] * task_attributes[i]['gpu_resource_requirement'] for i in range(num_tasks)) <= GPU_TOTAL_RESOURCE

    # 求解
    status = model.solve()

    allocation = []
    if pulp.LpStatus[model.status] == "Optimal":
        for i in range(num_tasks):
            if x[i].value() > 0.5:
                allocation.append(("CPU", i))
            elif z[i].value() > 0.5:
                allocation.append(("GPU", i))
    else:
        print("LP未找到最优解")

    return allocation

# HEFT调度函数
def heft_scheduling(dag, task_attributes, allocation):
    """基于HEFT的任务调度"""
    schedule = []
    processor_busy = {"CPU": 0, "GPU": 0}

    # 按拓扑序遍历任务
    for task_id in nx.topological_sort(dag):
        task_idx = task_id - 1
        processor, _ = next((proc, idx) for proc, idx in allocation if idx == task_idx)

        # 计算EST (最早开始时间)
        # 1. 考虑处理器可用时间
        est = processor_busy[processor]

        # 2. 考虑前驱任务的完成时间和通信延迟
        predecessors = list(dag.predecessors(task_id))
        if predecessors:
            pred_finish_times = []
            for pred in predecessors:
                pred_id = pred - 1
                # 在已调度的任务中找到前驱任务的信息
                pred_schedule = next(s for s in schedule if s[0] == pred_id)
                pred_finish = pred_schedule[3]  # 前驱任务的完成时间
                
                # 如果前驱任务在不同处理器上，添加通信延迟
                if pred_schedule[1] != processor:
                    comm_time = dag[pred][task_id]['data_size'] / DATA_TRANSFER_RATE
                    pred_finish_times.append(pred_finish + comm_time)
                else:
                    pred_finish_times.append(pred_finish)

            if pred_finish_times:  # 确保有前驱任务的完成时间
                est = max(est, max(pred_finish_times))

        # 计算执行时间和完成时间
        if processor == "CPU":
            exec_time = task_attributes[task_idx]['software_execution_time']
        else:  # GPU
            exec_time = task_attributes[task_idx]['gpu_execution_time']
            
        eft = est + exec_time  # 最早完成时间
        
        # 更新处理器状态和调度结果
        processor_busy[processor] = eft
        schedule.append((task_idx, processor, est, eft))

    return schedule

# 主函数
def main():
    file_path = "/root/123/FFT/dataset/ccr=0.1/dag_fft_223tasks.txt"
    print(f"读取数据集：{file_path}")
    dag, task_attributes = read_dataset(file_path)

    print("\n开始LP任务分配...")
    allocation = lp_task_allocation(dag, task_attributes)
    print(f"任务分配结果: {allocation}")

    print("\n开始HEFT调度...")
    schedule = heft_scheduling(dag, task_attributes, allocation)

    makespan = max(eft for _, _, _, eft in schedule)
    print(f"\n调度完成，makespan: {makespan:.2f}")

    output_file = "/root/123/FFT/result/ccr=0.1/HEFT/dag_fft_223tasks.txt"
    with open(output_file, 'w') as f:
        f.write("================ 调度结果 ================\n")
        for task_id, processor, est, eft in schedule:
            f.write(f"任务 {task_id + 1}:\n")
            f.write(f"  处理器: {processor}\n")
            f.write(f"  开始时间: {est:.2f}\n")
            f.write(f"  结束时间: {eft:.2f}\n")
        f.write(f"\n总调度时间 (makespan): {makespan:.2f}\n")

if __name__ == "__main__":
    main()