package blueprint

// Example of processing a dense layer as a method of Blueprint
func (bp *Blueprint) processDenseLayer(layer Layer, inputData interface{}) interface{} {
	inputValues := inputData.(map[string]float64)
	neurons := make(map[string]float64)

	for nodeID, node := range layer.Neurons {
		sum := 0.0
		for inputID, connection := range node.Connections {
			sum += inputValues[inputID] * connection.Weight
		}
		sum += node.Bias
		neurons[nodeID] = bp.Activate(node.ActivationType, sum)
	}
	return neurons
}
