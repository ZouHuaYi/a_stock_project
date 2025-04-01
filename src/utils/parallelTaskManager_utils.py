import os
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Callable, List, Any, Union, Dict, Optional, Tuple


class ParallelTaskManager:
    """
    一个完美的多线程多进程管理类，支持灵活的并行任务处理
    """
    
    def __init__(self):
        """初始化并行任务管理器"""
        self.cpu_count = multiprocessing.cpu_count()
        self.active_threads = {}
        self.active_processes = {}
        self.thread_results = {}
        self.process_results = {}
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu_count": self.cpu_count,
            "active_threads": len(self.active_threads),
            "active_processes": len(self.active_processes)
        }
    
    def run_in_thread(self, func: Callable, *args, thread_name: Optional[str] = None, **kwargs) -> str:
        """
        在新线程中运行函数
        
        Args:
            func: 要执行的函数
            *args: 传递给函数的位置参数
            thread_name: 线程名称，如果不提供则自动生成
            **kwargs: 传递给函数的关键字参数
            
        Returns:
            thread_id: 线程ID，可用于获取结果
        """
        if thread_name is None:
            thread_name = f"Thread-{len(self.active_threads) + 1}"
            
        thread_id = f"{thread_name}-{time.time()}"
        
        def wrapper():
            try:
                result = func(*args, **kwargs)
                self.thread_results[thread_id] = {"status": "completed", "result": result}
            except Exception as e:
                self.thread_results[thread_id] = {"status": "error", "error": str(e)}
            finally:
                if thread_id in self.active_threads:
                    del self.active_threads[thread_id]
        
        thread = threading.Thread(target=wrapper, name=thread_name)
        self.active_threads[thread_id] = thread
        self.thread_results[thread_id] = {"status": "running"}
        thread.start()
        
        return thread_id
    
    def run_in_process(self, func: Callable, *args, process_name: Optional[str] = None, **kwargs) -> str:
        """
        在新进程中运行函数
        
        Args:
            func: 要执行的函数
            *args: 传递给函数的位置参数
            process_name: 进程名称，如果不提供则自动生成
            **kwargs: 传递给函数的关键字参数
            
        Returns:
            process_id: 进程ID，可用于获取结果
        """
        if process_name is None:
            process_name = f"Process-{len(self.active_processes) + 1}"
            
        process_id = f"{process_name}-{time.time()}"
        
        # 使用队列在进程间通信
        result_queue = multiprocessing.Queue()
        
        def wrapper(queue):
            try:
                result = func(*args, **kwargs)
                queue.put({"status": "completed", "result": result})
            except Exception as e:
                queue.put({"status": "error", "error": str(e)})
        
        process = multiprocessing.Process(target=wrapper, args=(result_queue,), name=process_name)
        self.active_processes[process_id] = {"process": process, "queue": result_queue}
        self.process_results[process_id] = {"status": "running"}
        process.start()
        
        # 启动一个线程来监听结果队列
        def monitor_queue():
            try:
                result = result_queue.get()
                self.process_results[process_id] = result
                if process_id in self.active_processes:
                    del self.active_processes[process_id]
            except Exception as e:
                self.process_results[process_id] = {"status": "error", "error": f"监听队列时出错: {str(e)}"}
        
        threading.Thread(target=monitor_queue).start()
        
        return process_id
    
    def get_thread_result(self, thread_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        获取线程执行结果
        
        Args:
            thread_id: 线程ID
            timeout: 等待超时时间（秒），None表示无限等待
            
        Returns:
            结果字典，包含status和result/error字段
        """
        if thread_id not in self.thread_results:
            return {"status": "not_found"}
        
        if self.thread_results[thread_id]["status"] != "running":
            return self.thread_results[thread_id]
        
        if thread_id in self.active_threads:
            self.active_threads[thread_id].join(timeout)
            
        return self.thread_results[thread_id]
    
    def get_process_result(self, process_id: str, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        获取进程执行结果
        
        Args:
            process_id: 进程ID
            timeout: 等待超时时间（秒），None表示无限等待
            
        Returns:
            结果字典，包含status和result/error字段
        """
        if process_id not in self.process_results:
            return {"status": "not_found"}
        
        if self.process_results[process_id]["status"] != "running":
            return self.process_results[process_id]
        
        if process_id in self.active_processes:
            self.active_processes[process_id]["process"].join(timeout)
            
        return self.process_results[process_id]
    
    def execute_in_thread_pool(self, func: Callable, tasks: List[Tuple], max_workers: Optional[int] = None) -> List[Any]:
        """
        使用线程池执行多个任务
        
        Args:
            func: 要执行的函数
            tasks: 任务列表，每个元素是一个包含参数的元组
            max_workers: 最大线程数，默认为None（由ThreadPoolExecutor决定）
            
        Returns:
            结果列表，与任务列表顺序对应
        """
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(func, *task) for task in tasks]
            
            # 获取结果（按提交顺序）
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
                    
        return results
    
    def execute_in_process_pool(self, func: Callable, tasks: List[Tuple], max_workers: Optional[int] = None) -> List[Any]:
        """
        使用进程池执行多个任务
        
        Args:
            func: 要执行的函数
            tasks: 任务列表，每个元素是一个包含参数的元组
            max_workers: 最大进程数，默认为None（由ProcessPoolExecutor决定）
            
        Returns:
            结果列表，与任务列表顺序对应
        """
        if max_workers is None:
            max_workers = self.cpu_count
            
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            futures = [executor.submit(func, *task) for task in tasks]
            
            # 获取结果（按完成顺序）
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
                    
        return results
    
    def terminate_thread(self, thread_id: str) -> bool:
        """
        注意：Python不支持安全地终止线程
        此方法只是从管理器中移除线程记录，线程实际上会继续运行
        
        Args:
            thread_id: 线程ID
            
        Returns:
            是否成功移除线程记录
        """
        if thread_id in self.active_threads:
            del self.active_threads[thread_id]
            self.thread_results[thread_id] = {"status": "terminated"}
            return True
        return False
    
    def terminate_process(self, process_id: str) -> bool:
        """
        终止进程
        
        Args:
            process_id: 进程ID
            
        Returns:
            是否成功终止
        """
        if process_id in self.active_processes:
            self.active_processes[process_id]["process"].terminate()
            del self.active_processes[process_id]
            self.process_results[process_id] = {"status": "terminated"}
            return True
        return False
    
    def terminate_all_threads(self) -> int:
        """
        移除所有线程记录
        
        Returns:
            移除的线程数量
        """
        count = len(self.active_threads)
        for thread_id in list(self.active_threads.keys()):
            self.terminate_thread(thread_id)
        return count
    
    def terminate_all_processes(self) -> int:
        """
        终止所有进程
        
        Returns:
            终止的进程数量
        """
        count = len(self.active_processes)
        for process_id in list(self.active_processes.keys()):
            self.terminate_process(process_id)
        return count
    
    def cleanup(self):
        """清理所有资源"""
        self.terminate_all_threads()
        self.terminate_all_processes()
        self.thread_results.clear()
        self.process_results.clear()


# 提供一个装饰器方式使用多线程/多进程
def parallel(mode='thread', max_workers=None):
    """
    并行执行装饰器
    
    Args:
        mode: 'thread' 或 'process'
        max_workers: 最大工作者数量
        
    Returns:
        装饰后的函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = ParallelTaskManager()
            
            if mode == 'thread':
                return manager.run_in_thread(func, *args, **kwargs)
            elif mode == 'process':
                return manager.run_in_process(func, *args, **kwargs)
            else:
                raise ValueError(f"不支持的模式: {mode}")
        
        return wrapper
    
    return decorator