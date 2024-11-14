let eventSource = null;
const model1Data = {
    train_losses: [],
    train_accs: [],
    epochs: [],
    currentLoss: null,
    currentAcc: null
};
const model2Data = {
    train_losses: [],
    train_accs: [],
    epochs: [],
    currentLoss: null,
    currentAcc: null
};

function updatePlots() {
    // Training Loss Plot
    const train_loss_traces = [
        {
            x: model1Data.epochs,
            y: model1Data.train_losses,
            name: 'Model 1',
            type: 'scatter',
            line: {color: '#1f77b4'}
        },
        {
            x: model2Data.epochs,
            y: model2Data.train_losses,
            name: 'Model 2',
            type: 'scatter',
            line: {color: '#ff7f0e'}
        }
    ];
    
    const lossLayout = {
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Loss' },
        margin: { t: 30, r: 100 },
        showlegend: true,
        legend: {
            orientation: 'v',
            yanchor: 'top',
            y: 1,
            xanchor: 'right',
            x: 1.15
        },
        annotations: []
    };

    // Add current loss values as annotations
    if (model1Data.currentLoss !== null) {
        lossLayout.annotations.push({
            orientation: 'v',
            x: 0,
            y: 1,
            xref: 'paper',
            yref: 'paper',
            text: `Model-1 Loss: ${model1Data.currentLoss.toFixed(4)}`,
            showarrow: false,
            font: { color: '#1f77b4' }
        });
    }
    if (model2Data.currentLoss !== null) {
        lossLayout.annotations.push({
            orientation: 'v',
            x: 0.5,
            y: 1,
            xref: 'paper',
            yref: 'paper',
            text: `Model-2 Loss: ${model2Data.currentLoss.toFixed(4)}`,
            showarrow: false,
            font: { color: '#ff7f0e' }
        });
    }
    
    Plotly.newPlot('train_loss_plot', train_loss_traces, lossLayout);
    
    // Training Accuracy Plot
    const train_acc_traces = [
        {
            x: model1Data.epochs,
            y: model1Data.train_accs,
            name: 'Model 1',
            type: 'scatter',
            line: {color: '#1f77b4'}
        },
        {
            x: model2Data.epochs,
            y: model2Data.train_accs,
            name: 'Model 2',
            type: 'scatter',
            line: {color: '#ff7f0e'}
        }
    ];
    
    const accLayout = {
        xaxis: { title: 'Epoch' },
        yaxis: { title: 'Accuracy (%)', range: [0, 100] },
        margin: { t: 30, r: 100 },
        showlegend: true,
        legend: {
            orientation: 'v',
            yanchor: 'top',
            y: 1,
            xanchor: 'right',
            x: 1.15
        },
        annotations: []
    };

    // Add current accuracy values as annotations
    if (model1Data.currentAcc !== null) {
        accLayout.annotations.push({
            x: 0,
            y: 1,
            xref: 'paper',
            yref: 'paper',
            text: `Model-1 Acc: ${model1Data.currentAcc.toFixed(2)}%`,
            showarrow: false,
            font: { color: '#1f77b4' }
        });
    }
    if (model2Data.currentAcc !== null) {
        accLayout.annotations.push({
            x: 0.5,
            y: 1,
            xref: 'paper',
            yref: 'paper',
            text: `Model-2 Acc: ${model2Data.currentAcc.toFixed(2)}%`,
            showarrow: false,
            font: { color: '#ff7f0e' }
        });
    }
    
    Plotly.newPlot('train_acc_plot', train_acc_traces, accLayout);
}

document.getElementById('train_button').addEventListener('click', async () => {
    const model1_filters = document.getElementById('model1_filters').value.split(',').map(x => parseInt(x.trim()));
    const model2_filters = document.getElementById('model2_filters').value.split(',').map(x => parseInt(x.trim()));
    const model1_kernels = document.getElementById('model1_kernel').value.split(',').map(x => parseInt(x.trim()));
    const model2_kernels = document.getElementById('model2_kernel').value.split(',').map(x => parseInt(x.trim()));
    const optimizer = document.getElementById('optimizer').value;
    const batch_size = parseInt(document.getElementById('batch_size').value);
    const epochs = parseInt(document.getElementById('epochs').value);
    
    // Reset data
    model1Data.train_losses = [];
    model1Data.train_accs = [];
    model1Data.epochs = [];
    model1Data.currentLoss = null;
    model1Data.currentAcc = null;
    model2Data.train_losses = [];
    model2Data.train_accs = [];
    model2Data.epochs = [];
    model2Data.currentLoss = null;
    model2Data.currentAcc = null;
    
    try {
        // Start training
        await fetch('/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model1_filters,
                model2_filters,
                model1_kernels,
                model2_kernels,
                optimizer,
                batch_size,
                epochs
            }),
        });
        
        // Setup SSE for real-time updates
        if (eventSource) {
            eventSource.close();
        }
        
        eventSource = new EventSource('/stream_logs');
        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data === "DONE") {
                eventSource.close();
                return;
            }
            
            const modelData = data.model === "Model 1" ? model1Data : model2Data;
            
            // Update current metrics
            modelData.currentLoss = data.train_loss;
            modelData.currentAcc = data.train_acc;
            
            if (data.final_epoch) {
                modelData.train_losses.push(data.train_loss);
                modelData.train_accs.push(data.train_acc);
                modelData.epochs.push(data.epoch);
            }
            
            // Update plots every epoch
            updatePlots();
        };
        
    } catch (error) {
        console.error('Error during training:', error);
    }
}); 