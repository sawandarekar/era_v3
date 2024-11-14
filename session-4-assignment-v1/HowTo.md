# MNIST CNN Training with Real-time Visualization

This project implements a 4-layer CNN for MNIST digit classification with real-time training visualization.

## Setup Instructions

1. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Start the Flask server:
   ```bash
   python server.py
   ```

4. In a new terminal, start the training:
   ```bash
   python train.py
   ```

5. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

- `model.py`: Contains the CNN architecture
- `train.py`: Training script with real-time logging
- `server.py`: Flask server for web interface
- `index.html`: Web interface for visualization
- `requirements.txt`: Required Python packages
- `static/`: Directory for storing training statistics and test samples

## Features

- 4-layer CNN architecture with max pooling and dropout
- Real-time training loss and accuracy visualization
- CUDA support for GPU acceleration
- Display of 10 random test samples with predictions after training
- Interactive web interface with Plotly graphs

## Notes

- The training progress is automatically updated every 2 seconds on the web interface
- Training statistics are saved in `static/training_stats.json`
- Test samples are saved in `static/test_samples.json`
- The trained model is saved as `mnist_cnn.pth` 