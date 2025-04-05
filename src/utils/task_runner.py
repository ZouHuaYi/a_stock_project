from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

class TaskRunner:
    """
    一个封装类，用于方便执行多线程或多进程任务
    """
    def __init__(self, use_threads=True, max_workers=4):
        """
        初始化 TaskRunner
        :param use_threads: True 使用线程池，False 使用进程池
        :param max_workers: 最大工作线程/进程数，默认为 4
        """
        self.use_threads = use_threads
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers) if use_threads else ProcessPoolExecutor(max_workers)

    def run_tasks(self, task_func, task_args_list, timeout=None):
        """
        执行一组任务
        :param task_func: 要执行的任务函数
        :param task_args_list: 任务参数列表，每个元素是一个元组，包含 task_func 的参数
        :param timeout: 每个任务的超时时间（秒），可选
        :return: 任务结果列表
        """
        results = []
        with self.executor as executor:
            # 提交所有任务
            futures = [executor.submit(task_func, *args) for args in task_args_list]
            # 收集结果
            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except TimeoutError:
                    results.append(f"Task timed out after {timeout} seconds")
                except Exception as e:
                    results.append(f"Task failed with error: {str(e)}")
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()


# 示例任务函数
def io_task(name):
    """模拟 I/O 密集型任务"""
    print(f"开始执行 I/O 任务: {name}")
    time.sleep(2)  # 模拟 I/O 等待
    return f"{name} 完成"

def cpu_task(n):
    """模拟 CPU 密集型任务"""
    print(f"开始执行 CPU 任务: 计算 {n} 的平方")
    return n * n

# # 使用示例
if __name__ == "__main__":
    # 1. 使用线程池执行 I/O 密集型任务
    print("=== 测试多线程（I/O 密集型任务） ===")
    io_runner = TaskRunner(use_threads=True, max_workers=3)
    io_tasks = [("任务1",), ("任务2",), ("任务3",)]
    io_results = io_runner.run_tasks(io_task, io_tasks, timeout=3)
    for result in io_results:
        print(result)

    # 2. 使用进程池执行 CPU 密集型任务
    print("\n=== 测试多进程（CPU 密集型任务） ===")
    cpu_runner = TaskRunner(use_threads=False, max_workers=3)
    cpu_tasks = [(5,), (10,), (15,)]
    cpu_results = cpu_runner.run_tasks(cpu_task, cpu_tasks)
    for result in cpu_results:
        print(result)
