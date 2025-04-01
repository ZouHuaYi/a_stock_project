from utils.parallelTaskManager_utils import ConcurrentWorker

if __name__ == "__main__":
    worker = ConcurrentWorker(use_process=False)
    def test_func(x):
        return x
    results = worker.map(test_func, [1, 2, 3, 4, 5])
    print(results)
    worker.shutdown()
