# -*- coding: utf-8 -*-
"""任务执行器工具模块，用于多线程或多进程处理任务"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
from utils.logger import get_logger

# 创建日志记录器
logger = get_logger(__name__)

class TaskRunner:
    """
    任务执行器类，用于方便执行多线程或多进程任务
    """
    def __init__(self, use_threads=True, max_workers=4):
        """
        初始化任务执行器
        
        参数:
            use_threads (bool): True使用线程池，False使用进程池
            max_workers (int): 最大工作线程/进程数，默认为4
        """
        self.use_threads = use_threads
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers) if use_threads else ProcessPoolExecutor(max_workers)
        self.futures = []  # 存储所有提交的任务
        
        logger.info(f"初始化{'线程' if use_threads else '进程'}池，最大工作数量: {max_workers}")

    def run_tasks(self, task_func, task_args_list, timeout=None):
        """
        执行一组任务
        
        参数:
            task_func (callable): 要执行的任务函数
            task_args_list (list): 任务参数列表，每个元素是一个元组，包含task_func的参数
            timeout (float, 可选): 每个任务的超时时间（秒）
            
        返回:
            list: 任务结果列表
        """
        results = []
        task_count = len(task_args_list)
        logger.info(f"准备执行{task_count}个任务")
        
        with self.executor as executor:
            # 提交所有任务
            futures = [executor.submit(task_func, *args) for args in task_args_list]
            
            # 收集结果
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                    logger.debug(f"任务[{i+1}/{task_count}]完成")
                except TimeoutError:
                    logger.warning(f"任务[{i+1}/{task_count}]超时（{timeout}秒）")
                    results.append(None)
                except Exception as e:
                    logger.error(f"任务[{i+1}/{task_count}]失败: {str(e)}")
                    results.append(None)
                    
        logger.info(f"所有任务执行完毕，成功: {sum(1 for r in results if r is not None)}/{task_count}")
        return results
    
    def submit_task(self, task_func, *args, **kwargs):
        """
        提交单个任务到执行器
        
        参数:
            task_func (callable): 要执行的任务函数
            *args: 任务函数的位置参数
            **kwargs: 任务函数的关键字参数
            
        返回:
            Future: Future对象
        """
        # 处理kwargs
        if kwargs:
            # 创建一个包装函数，将kwargs传递给原始函数
            def wrapped_func(*args):
                return task_func(*args, **kwargs)
            future = self.executor.submit(wrapped_func, *args)
        else:
            future = self.executor.submit(task_func, *args)
            
        self.futures.append(future)
        logger.debug(f"提交任务: {task_func.__name__}")
        return future
    
    def wait_all_done(self, timeout=None):
        """
        等待所有已提交的任务完成
        
        参数:
            timeout (float, 可选): 每个任务的超时时间（秒）
            
        返回:
            list: 任务结果列表
        """
        results = []
        task_count = len(self.futures)
        
        if task_count == 0:
            logger.warning("没有等待中的任务")
            return []
            
        logger.info(f"等待{task_count}个任务完成")
        
        for i, future in enumerate(self.futures):
            try:
                result = future.result(timeout=timeout)
                results.append(result)
                logger.debug(f"任务[{i+1}/{task_count}]完成")
            except TimeoutError:
                logger.warning(f"任务[{i+1}/{task_count}]超时（{timeout}秒）")
                results.append(None)
            except Exception as e:
                logger.error(f"任务[{i+1}/{task_count}]失败: {str(e)}")
                results.append(None)
        
        # 清空futures列表，以便后续复用
        self.futures = []
        logger.info(f"所有等待的任务执行完毕，成功: {sum(1 for r in results if r is not None)}/{task_count}")
        return results

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()
        logger.info("任务执行器已关闭")


# 示例任务函数
def io_task(name):
    """模拟I/O密集型任务"""
    logger.info(f"开始执行I/O任务: {name}")
    time.sleep(2)  # 模拟I/O等待
    return f"{name}完成"

def cpu_task(n):
    """模拟CPU密集型任务"""
    logger.info(f"开始执行CPU任务: 计算{n}的平方")
    return n * n 