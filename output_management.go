// blueprint/output_management.go
package blueprint

import (
	"fmt"
	"math/rand"
)

// GetPreviousOutputActivationTypes retrieves activation types for neurons in the output layer
func (bp *Blueprint) GetPreviousOutputActivationTypes() []string {
	var activationTypes []string
	for _, neuron := range bp.Config.Layers.Output.Neurons {
		activationTypes = append(activationTypes, neuron.ActivationType)
	}
	return activationTypes
}

// ReattachOutputLayerZeroBias reattaches the output layer with specified activation types and zero bias
func (bp *Blueprint) ReattachOutputLayerZeroBias(numOutputs int, outputActivationTypes []string) {
	lastHiddenLayer := bp.Config.Layers.Hidden[len(bp.Config.Layers.Hidden)-1]

	bp.Config.Layers.Output = Layer{
		LayerType: "dense",
		Neurons:   make(map[string]Neuron),
	}

	for i := 0; i < numOutputs; i++ {
		neuronID := fmt.Sprintf("output%d", i)
		activationType := "softmax"
		if i < len(outputActivationTypes) {
			activationType = outputActivationTypes[i]
		}

		connections := make(map[string]Connection)
		for hiddenNeuronID := range lastHiddenLayer.Neurons {
			connections[hiddenNeuronID] = Connection{Weight: rand.Float64() - 0.5}
		}

		bp.Config.Layers.Output.Neurons[neuronID] = Neuron{
			ActivationType: activationType,
			Connections:    connections,
			Bias:           0,
		}
	}
}
