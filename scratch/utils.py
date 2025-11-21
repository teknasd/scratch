import matplotlib.pyplot as plt
import cProfile
import pstats
import io
import tracemalloc
import functools
from loguru import logger
from scratch.env import DEBUG


def plot_history(history):
    """Plot training loss history"""
    epochs = [x[0] for x in history]
    losses = [x[1] for x in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


def profile_cpu(sort_by='cumulative', top_n=20):
    """
    Profile CPU usage with cProfile (only when DEBUG=True)
    
    Usage:
        @profile_cpu()
        def train_step():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not DEBUG:
                return func(*args, **kwargs)
            
            profiler = cProfile.Profile()
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            s = io.StringIO()
            stats = pstats.Stats(profiler, stream=s)
            stats.sort_stats(sort_by)
            stats.print_stats(top_n)
            
            logger.info(f"\n{'='*60}\nCPU PROFILE: {func.__name__}\n{'='*60}\n{s.getvalue()}")
            return result
        return wrapper
    return decorator


def profile_mem(top_n=10):
    """
    Profile memory usage with tracemalloc (only when DEBUG=True)
    
    Args:
        top_n: Number of top memory allocations to display
    
    Usage:
        @profile_mem()
        def create_model():
            pass
            
        @profile_mem(top_n=5)
        def another_function():
            pass
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not DEBUG:
                return func(*args, **kwargs)
            
            tracemalloc.start()
            snapshot_before = tracemalloc.take_snapshot()
            result = func(*args, **kwargs)
            snapshot_after = tracemalloc.take_snapshot()
            
            current, peak = tracemalloc.get_traced_memory()
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            
            logger.info(f"\n{'='*60}\nMEMORY PROFILE: {func.__name__}\n{'='*60}")
            logger.info(f"Current: {current / 1024 / 1024:.2f} MB | Peak: {peak / 1024 / 1024:.2f} MB")
            logger.info(f"\nTop {top_n} allocations:")
            for stat in top_stats[:top_n]:
                logger.info(stat)
            
            tracemalloc.stop()
            return result
        return wrapper
    return decorator
