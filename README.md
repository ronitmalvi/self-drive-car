# Self-Driving Car Simulation

This project simulates a self-driving car using a deep learning model. The model is trained to predict steering angles based on input images from a car's front camera.

## Project Structure
``


## Requirements

To install the required packages, run:

```sh
pip install -r requirements.txt
```

## Training the Model
To train the model, run the TrainingSimulation.py script:
```
python TrainingSimulation.py
```

This script will:
1.Import and preprocess the training data.
2.Balance the data to ensure even distribution of steering angles.
3.Augment the data with various transformations.
4.Train a convolutional neural network (CNN) to predict steering angles.
5.Save the trained model to model.h5.

## Testing the Model
To test the model, run the TestSimulation.py script:
```
python TestSimulation.py
```

This script will:

1.Load the trained model from model.h5.
2.Start a Flask server with Socket.IO to receive telemetry data.
3.Preprocess incoming images and predict steering angles.
4.Send control commands (steering and throttle) back to the simulator.
