package blueprint

import (
	"math/rand/v2"
	"strconv"
)

// CreateCustomNetworkConfig creates a network configuration with incrementing neuron and layer IDs.
func (bp *Blueprint) CreateCustomNetworkConfig(numInputs, numHiddenNeurons, numOutputs int, outputActivationTypes []string, modelID, projectName string) {
	// Initialize metadata with counts
	bp.Config.Metadata = ModelMetadata{
		ModelID:              modelID,
		ProjectName:          projectName,
		LastTrainingAccuracy: 0.0,
		LastTestAccuracy:     0.0,
	}

	// Counters for neurons and layers using int64
	var neuronCount, layerCount int64

	// Define the input layer
	inputLayer := Layer{
		LayerType: "dense",
		Neurons:   make(map[string]Neuron),
	}
	for i := 0; i < numInputs; i++ {
		neuronID := "neuron" + strconv.FormatInt(neuronCount, 10)
		inputLayer.Neurons[neuronID] = Neuron{}
		neuronCount++
	}
	bp.Config.Layers.Input = inputLayer
	layerCount++

	// Define the first hidden layer with incremented neuron IDs
	hiddenLayer := Layer{
		LayerType: "dense",
		Neurons:   make(map[string]Neuron),
	}
	for i := 0; i < numHiddenNeurons; i++ {
		neuronID := "neuron" + strconv.FormatInt(neuronCount, 10)
		hiddenLayer.Neurons[neuronID] = Neuron{
			ActivationType: "relu",
			Connections: func() map[string]Connection {
				connections := make(map[string]Connection)
				for j := 0; j < numInputs; j++ {
					inputNeuronID := "neuron" + strconv.FormatInt(int64(j), 10)
					connections[inputNeuronID] = Connection{Weight: rand.Float64()}
				}
				return connections
			}(),
			Bias: rand.Float64(),
		}
		neuronCount++
	}
	bp.Config.Layers.Hidden = []Layer{hiddenLayer}
	layerCount++

	// Define the output layer with customized activation types, random weights, and incremented neuron IDs
	outputLayer := Layer{
		LayerType: "dense",
		Neurons:   make(map[string]Neuron),
	}
	for i := 0; i < numOutputs; i++ {
		neuronID := "neuron" + strconv.FormatInt(neuronCount, 10)
		activationType := "sigmoid"
		if i < len(outputActivationTypes) {
			activationType = outputActivationTypes[i]
		}

		connections := make(map[string]Connection)
		for h := 0; h < numHiddenNeurons; h++ {
			hiddenNeuronID := "neuron" + strconv.FormatInt(int64(numInputs+h), 10)
			connections[hiddenNeuronID] = Connection{Weight: rand.Float64()}
		}

		outputLayer.Neurons[neuronID] = Neuron{
			ActivationType: activationType,
			Connections:    connections,
			Bias:           rand.Float64(),
		}
		neuronCount++
	}
	bp.Config.Layers.Output = outputLayer
	layerCount++

	// Save neuron and layer count in metadata
	bp.Config.Metadata.TotalNeurons = neuronCount
	bp.Config.Metadata.TotalLayers = layerCount
}
