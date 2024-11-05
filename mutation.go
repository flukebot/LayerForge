// blueprint/mutation.go
package blueprint

import (
	"fmt"
	"math/rand"
)

// ApplySingleMutation applies a single mutation to the model configuration.
func (bp *Blueprint) ApplySingleMutation(mutationTypes []string, neuronRange [2]int, layerRange [2]int) {
	// Randomly select a mutation type from the list
	mutationType := mutationTypes[rand.Intn(len(mutationTypes))]

	// Apply the mutation based on the selected type
	switch mutationType {
	case "AppendNewLayer":
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		bp.AppendNewLayerFullConnections(numNewNeuronsOrFilters)

	case "AppendMultipleLayers":
		numNewLayers := rand.Intn(layerRange[1]-layerRange[0]+1) + layerRange[0]
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		bp.AppendMultipleLayers(numNewLayers, numNewNeuronsOrFilters)

	case "AppendCNNAndDenseLayer":
		filterSize := rand.Intn(3) + 3
		numFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		stride := 1
		padding := (filterSize - 1) / 2
		if err := bp.AppendCNNLayer(filterSize, numFilters, stride, padding); err != nil {
			fmt.Printf("Failed to append CNN layer: %v\n", err)
		}
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		bp.AppendNewLayerFullConnections(numNewNeuronsOrFilters)

	case "AppendLSTMLayer":
		bp.AppendLSTMLayer()
		numNewNeuronsOrFilters := rand.Intn(neuronRange[1]-neuronRange[0]+1) + neuronRange[0]
		bp.AppendNewLayerFullConnections(numNewNeuronsOrFilters)

	default:
		fmt.Println("Unknown mutation type:", mutationType)
	}

	// Reattach the output layer with previous activation types
	previousOutputActivationTypes := bp.GetPreviousOutputActivationTypes()
	bp.ReattachOutputLayerZeroBias(len(previousOutputActivationTypes), previousOutputActivationTypes)
}
