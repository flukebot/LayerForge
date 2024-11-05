// blueprint/utils.go
package blueprint

import (
	"math/rand"
	"strconv"
	"strings"
)

// randomActivationType returns a random activation type for a neuron
func randomActivationType() string {
	activationTypes := []string{"relu", "sigmoid", "tanh", "leaky_relu"}
	return activationTypes[rand.Intn(len(activationTypes))]
}

// Random2DSlice generates a 2D slice of random float64 values with the given dimensions
func Random2DSlice(rows, cols int) [][]float64 {
	slice := make([][]float64, rows)
	for i := range slice {
		slice[i] = RandomSlice(cols)
	}
	return slice
}

// RandomSlice generates a 1D slice of random float64 values of a given length
func RandomSlice(length int) []float64 {
	slice := make([]float64, length)
	for i := range slice {
		slice[i] = rand.Float64()
	}
	return slice
}

// getHighestNeuronID finds the highest numbered neuron ID in the existing layers.
func (bp *Blueprint) getHighestNeuronID() int64 {
	var maxID int64 = -1

	// Helper function to parse and update the maxID
	updateMaxID := func(neurons map[string]Neuron) {
		for id := range neurons {
			if strings.HasPrefix(id, "neuron") {
				numStr := strings.TrimPrefix(id, "neuron")
				if num, err := strconv.ParseInt(numStr, 10, 64); err == nil && num > maxID {
					maxID = num
				}
			}
		}
	}

	// Check neurons in input layer
	updateMaxID(bp.Config.Layers.Input.Neurons)
	// Check neurons in all hidden layers
	for _, layer := range bp.Config.Layers.Hidden {
		updateMaxID(layer.Neurons)
	}
	// Check neurons in output layer
	updateMaxID(bp.Config.Layers.Output.Neurons)

	return maxID
}
