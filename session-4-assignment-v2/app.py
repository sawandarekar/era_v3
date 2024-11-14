from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from model import train_model, ConvNet, evaluate_random_samples
import torch
import json
import queue
import threading

app = Flask(__name__)
app.app_context()
training_logs = queue.Queue()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream_logs')
def stream_logs():
    def generate():
        while True:
            log = training_logs.get()
            if log == "DONE":
                break
            yield f"data: {json.dumps(log)}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    
    # Extract parameters for both models
    model1_filters = [int(x) for x in data['model1_filters']]
    model2_filters = [int(x) for x in data['model2_filters']]
    model1_kernels = [int(x) for x in data['model1_kernels']]
    model2_kernels = [int(x) for x in data['model2_kernels']]
    optimizer = data['optimizer']
    batch_size = int(data['batch_size'])
    epochs = int(data['epochs'])
    
    def train_models():
        # Train both models
        results1 = train_model(model1_filters, model1_kernels, optimizer, batch_size, epochs, 
                             training_logs, model_name="Model 1")
        results2 = train_model(model2_filters, model2_kernels, optimizer, batch_size, epochs, 
                             training_logs, model_name="Model 2")
        
        # Signal completion
        training_logs.put("DONE")
        
        return jsonify({
            'model1': results1,
            'model2': results2
        })
    
    # Start training in a separate thread
    thread = threading.Thread(target=train_models)
    thread.start()
    
    return jsonify({"status": "Training started"})

if __name__ == '__main__':
    app.run(debug=True) 