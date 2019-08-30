# ベンチマーク計測
# ==============================================================================

import time

class Benchmark:

    def __init__(self):
        self._start = time.time()

    def start(self):
        self._start = time.time()

    def stopAndStart(self):
        current = time.time()
        tm = current - self._start
        self._start = current
        return tm

def calc_benchmark(func, **kwargs):
    """
    ベンチマークを計測

    Arguments:
        func: 実行する関数
        **kwargs: 実行する関数の引数
    Return: 
        (funcの実行結果, 計測時間)
    """
    bench = Benchmark()
    result = func(**kwargs)
    tm = bench.stopAndStart()
    return result, tm
