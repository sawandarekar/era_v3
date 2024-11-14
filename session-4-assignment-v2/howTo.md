# Fashion MNIST Model Comparison Tool

This tool allows you to compare two different CNN architectures trained on the Fashion MNIST dataset.

## Setup Instructions

1. Create a virtual environment: 

bash

```python -m venv venv
```
source venv/bin/activate # On Windows: venv\Scripts\activate

```bash
pip install -r requirements.txt
```

2. Install requirements:

```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```
4. Open your browser and navigate to `http://localhost:5000`

## Using the Tool

1. Configure Model 1 and Model 2 by entering comma-separated filter values (e.g., "16,32,64,128")
2. Select the optimizer (Adam or SGD)
3. Set the batch size and number of epochs
4. Click "Train Models" to start training
5. Monitor the training progress in the training log
6. Compare the results using the loss and accuracy plots

## System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended)
- Modern web browser
- 4GB+ RAM

## Notes

- Training time will vary depending on your hardware
- Using a GPU is recommended for faster training
- The application will automatically use CUDA if available