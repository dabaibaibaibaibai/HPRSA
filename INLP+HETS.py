import numpy as np
import networkx as nx
import pulp
import random

class HETSScheduler:
    def __init__(self, num_processors=3):
        self.num_processors = num_processors  # 总处理器数量
        self.est = {}  # 最早开始时间
        self.eft = {}  # 最早完成时间
        self.processor_available_time = [0] * num_processors  # 处理器可用时间
        self.task_processor_map = {}  # 任务到处理器的映射
        
    def read_dataset(self, file_path):
        """读取数据集文件"""
        task_attributes = []
        dag = nx.DiGraph()
        with open(file_path, 'r') as file:
            lines = file.readlines()
            task_index = -1
            reading_edges = False
            num_tasks = 0
            
            for line in lines:
                if line.startswith("Number of Tasks:"):
                    num_tasks = int(line.strip().split(": ")[1])
                    # 预先添加所有节点
                    dag.add_nodes_from(range(num_tasks))
                elif line.startswith("Task "):
                    task_index += 1
                    task_attributes.append({})
                elif ": " in line and not reading_edges:
                    attr, value = line.strip().split(": ")
                    if attr in ['release_time', 'deadline', 'software_execution_time', 
                              'fpga_execution_time', 'gpu_execution_time', 
                              'fpga_resource_requirement', 'gpu_resource_requirement']:
                        task_attributes[task_index][attr] = int(value)
                elif line.startswith("DAG Edges:"):
                    reading_edges = True
                elif reading_edges and " -> " in line:
                    parts = line.strip().split(" (Data Size: ")
                    source, target = map(int, parts[0].split(" -> "))
                    # 将节点ID调整为从0开始
                    source -= 1
                    target -= 1
                    data_size = int(parts[1].rstrip(")"))
                    dag.add_edge(source, target, data_size=data_size)
        
        return dag, task_attributes

    def calculate_rank(self, task_id, dag, task_attributes):
        """计算任务的rank值(upward rank)"""
        if task_id in self.rank_cache:
            return self.rank_cache[task_id]
            
        # 获取平均执行时间
        exec_times = [
            task_attributes[task_id]['software_execution_time'],
            task_attributes[task_id]['fpga_execution_time'],
            task_attributes[task_id]['gpu_execution_time']
        ]
        avg_exec_time = sum(exec_times) / len(exec_times)
        
        successors = list(dag.successors(task_id))
        if not successors:
            rank = avg_exec_time
        else:
            # 计算后继任务的最大rank值
            max_successor_rank = 0
            for succ in successors:
                # 获取通信时间
                comm_time = dag[task_id][succ]['data_size']
                succ_rank = self.calculate_rank(succ, dag, task_attributes)
                max_successor_rank = max(max_successor_rank, comm_time + succ_rank)
            rank = avg_exec_time + max_successor_rank
            
        self.rank_cache[task_id] = rank
        return rank

    def get_task_execution_time(self, task_id, processor_id, task_attributes):
        """获取任务在指定处理器上的执行时间"""
        task_attr = task_attributes[task_id]  # 直接使用task_id作为索引，因为现在是从0开始的
        if processor_id == 0:  # CPU
            return task_attr['software_execution_time']
        elif processor_id == 1:  # FPGA
            return task_attr['fpga_execution_time']
        else:  # GPU
            return task_attr['gpu_execution_time']

    def calculate_est(self, task_id, processor_id, dag, task_attributes, scheduled_tasks):
        """计算任务在指定处理器上的最早开始时间"""
        predecessors = list(dag.predecessors(task_id))
        if not predecessors:
            return max(task_attributes[task_id]['release_time'], 
                      self.processor_available_time[processor_id])
            
        max_ready_time = task_attributes[task_id]['release_time']
        for pred in predecessors:
            if pred not in scheduled_tasks:
                return float('inf')
                
            pred_finish_time = self.eft[pred]
            comm_time = dag[pred][task_id]['data_size']
            
            # 如果前驱任务在同一处理器上，通信时间为0
            if self.task_processor_map[pred] == processor_id:
                comm_time = 0
                
            ready_time = pred_finish_time + comm_time
            max_ready_time = max(max_ready_time, ready_time)
            
        return max(max_ready_time, self.processor_available_time[processor_id])

    def schedule_tasks(self, dag, task_attributes):
        """使用HETS算法进行任务调度"""
        num_tasks = len(task_attributes)
        self.rank_cache = {}
        scheduled_tasks = set()
        schedule = []
        
        # 1. 计算所有任务的rank值
        task_ranks = []
        for task_id in range(num_tasks):  # 从0开始
            rank = self.calculate_rank(task_id, dag, task_attributes)
            task_ranks.append((task_id, rank))
            
        # 2. 按rank值降序排序任务
        task_ranks.sort(key=lambda x: x[1], reverse=True)
        
        # 3. 调度任务
        for task_id, _ in task_ranks:
            min_finish_time = float('inf')
            best_processor = -1
            best_start_time = -1
            
            # 尝试每个处理器
            for processor_id in range(self.num_processors):
                # 计算EST和EFT
                est = self.calculate_est(task_id, processor_id, dag, task_attributes, scheduled_tasks)
                if est == float('inf'):
                    continue
                    
                exec_time = self.get_task_execution_time(task_id, processor_id, task_attributes)
                eft = est + exec_time
                
                # 更新最佳选择
                if eft < min_finish_time:
                    min_finish_time = eft
                    best_processor = processor_id
                    best_start_time = est
            
            if best_processor == -1:
                raise Exception(f"无法为任务 {task_id} 找到可行的处理器")
                
            # 更新调度信息
            self.est[task_id] = best_start_time
            self.eft[task_id] = min_finish_time
            self.processor_available_time[best_processor] = min_finish_time
            self.task_processor_map[task_id] = best_processor
            scheduled_tasks.add(task_id)
            
            # 记录调度结果（输出时将task_id加1以保持与原始数据集一致）
            processor_type = ["CPU", "FPGA", "GPU"][best_processor]
            schedule.append((task_id + 1, processor_type, best_start_time, min_finish_time))
            
        return schedule

def main():
    # 创建HETS调度器实例
    scheduler = HETSScheduler(num_processors=3)
    
    # 读取数据集
    file_path = "/root/123/FFT/dataset/ccr=1.0/dag_fft_5tasks.txt"
    dag, task_attributes = scheduler.read_dataset(file_path)
    
    # 打印调试信息
    print("图中的节点:", list(dag.nodes()))
    print("图中的边:", list(dag.edges()))
    print("任务属性数量:", len(task_attributes))
    
    try:
        # 执行调度
        schedule = scheduler.schedule_tasks(dag, task_attributes)
        
        # 计算makespan
        makespan = max(finish_time for _, _, _, finish_time in schedule)
        
        # 保存结果
        output_file = "/root/123/FFT/result/ccr=1.0/HETS/dag_fft_5tasks.txt"
        with open(output_file, 'w') as f:
            f.write("============== HETS调度结果 ==============\n")
            f.write(f"总任务数: {len(task_attributes)}\n")
            f.write(f"最终makespan: {makespan}\n\n")
            
            f.write("任务调度详情:\n")
            for task_id, processor, start_time, finish_time in sorted(schedule, key=lambda x: x[0]):
                f.write(f"任务 {task_id}:\n")
                f.write(f"  处理器: {processor}\n")
                f.write(f"  开始时间: {start_time}\n")
                f.write(f"  结束时间: {finish_time}\n")
                f.write(f"  执行时间: {finish_time - start_time}\n\n")
                
            # 计算各处理器利用率
            processor_times = {"CPU": [], "FPGA": [], "GPU": []}
            for _, processor, start_time, finish_time in schedule:
                processor_times[processor].append(finish_time - start_time)
                
            f.write("\n处理器利用率统计:\n")
            for processor, times in processor_times.items():
                if times:
                    total_time = sum(times)
                    utilization = total_time / (makespan * len(times)) * 100
                    f.write(f"{processor}:\n")
                    f.write(f"  任务数: {len(times)}\n")
                    f.write(f"  总执行时间: {total_time}\n")
                    f.write(f"  平均执行时间: {total_time/len(times):.2f}\n")
                    f.write(f"  利用率: {utilization:.2f}%\n\n")
        
        print(f"调度完成！结果已保存至: {output_file}")
        print(f"最终makespan: {makespan}")
        
    except Exception as e:
        print(f"调度过程出错: {str(e)}")

if __name__ == "__main__":
    main()