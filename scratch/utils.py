
import matplotlib.pyplot as plt

def plot_history(history):
    """
    Plot the training loss history
    
    Args:
        history: List of tuples (epoch, loss)
    """
    epochs = [x[0] for x in history]
    losses = [x[1] for x in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

  