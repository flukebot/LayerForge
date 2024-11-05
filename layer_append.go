// blueprint/layer_append.go
package blueprint

import (
	"fmt"
	"math/rand"
	"strconv"
	// Importing UUID package for neuron IDs
)

// AppendNewLayerFullConnections adds a new hidden dense layer with fully connected neurons
func (bp *Blueprint) AppendNewLayerFullConnections(numNewNeurons int) {
	newLayer := Layer{
		Neurons:   make(map[string]Neuron),
		LayerType: "dense",
	}

	// Get the initial highest neuron ID
	highestID := bp.getHighestNeuronID() + 1 // Start from the next available ID

	for i := 0; i < numNewNeurons; i++ {
		neuronID := "neuron" + strconv.FormatInt(highestID, 10)
		newNeuron := Neuron{
			ActivationType: randomActivationType(),
			Connections:    make(map[string]Connection),
			Bias:           rand.NormFloat64(),
		}

		var previousLayerNeurons map[string]Neuron
		if len(bp.Config.Layers.Hidden) == 0 {
			previousLayerNeurons = bp.Config.Layers.Input.Neurons
		} else {
			previousLayerNeurons = bp.Config.Layers.Hidden[len(bp.Config.Layers.Hidden)-1].Neurons
		}
		for prevNeuronID := range previousLayerNeurons {
			newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
		}
		newLayer.Neurons[neuronID] = newNeuron
		highestID++ // Increment for the next neuron
	}

	bp.Config.Layers.Hidden = append(bp.Config.Layers.Hidden, newLayer)
}

// AppendMultipleLayers appends multiple layers with a specified number of neurons to the network
func (bp *Blueprint) AppendMultipleLayers(numNewLayers, numNewNeurons int) {
	// Get the initial highest neuron ID once at the beginning
	highestID := bp.getHighestNeuronID() + 1 // Start from the next available ID

	for i := 0; i < numNewLayers; i++ {
		layer := Layer{
			Neurons:   make(map[string]Neuron),
			LayerType: "dense",
		}
		for j := 0; j < numNewNeurons; j++ {
			// Generate a new neuron ID sequentially
			neuronID := "neuron" + strconv.FormatInt(highestID, 10)
			newNeuron := Neuron{
				ActivationType: randomActivationType(),
				Connections:    make(map[string]Connection),
				Bias:           rand.NormFloat64(),
			}

			// Set up connections from neurons in the previous layer
			var previousLayerNeurons map[string]Neuron
			if len(bp.Config.Layers.Hidden) == 0 {
				previousLayerNeurons = bp.Config.Layers.Input.Neurons
			} else {
				previousLayerNeurons = bp.Config.Layers.Hidden[len(bp.Config.Layers.Hidden)-1].Neurons
			}
			for prevNeuronID := range previousLayerNeurons {
				newNeuron.Connections[prevNeuronID] = Connection{Weight: rand.NormFloat64()}
			}

			// Add the new neuron to the current layer
			layer.Neurons[neuronID] = newNeuron
			highestID++ // Increment the neuron ID for the next neuron
		}
		// Append the newly created layer to the hidden layers
		bp.Config.Layers.Hidden = append(bp.Config.Layers.Hidden, layer)
	}
}

// AppendCNNLayer adds a CNN layer to the network configuration
func (bp *Blueprint) AppendCNNLayer(filterSize, numFilters, stride, padding int) error {
	if filterSize <= 0 || numFilters <= 0 || stride <= 0 || padding < 0 {
		return fmt.Errorf("invalid CNN layer parameters")
	}

	filters := make([]Filter, numFilters)
	for i := 0; i < numFilters; i++ {
		filters[i] = Filter{
			Weights: Random2DSlice(filterSize, filterSize),
			Bias:    rand.Float64(),
		}
	}

	newLayer := Layer{
		LayerType: "conv",
		Filters:   filters,
		Stride:    stride,
		Padding:   padding,
	}
	bp.Config.Layers.Hidden = append(bp.Config.Layers.Hidden, newLayer)
	return nil
}

// AppendLSTMLayer appends an LSTM layer to the network configuration
func (bp *Blueprint) AppendLSTMLayer() {
	lstmLayer := Layer{
		LayerType: "lstm",
		LSTMCells: []LSTMCell{
			{
				InputWeights:  RandomSlice(10),
				ForgetWeights: RandomSlice(10),
				OutputWeights: RandomSlice(10),
				CellWeights:   RandomSlice(10),
				Bias:          rand.Float64(),
			},
		},
	}
	bp.Config.Layers.Hidden = append(bp.Config.Layers.Hidden, lstmLayer)
}
