package blueprint

import "strconv"

func (bp *Blueprint) processLSTMLayer(layer Layer, inputData interface{}) interface{} {
	// inputData is expected to be [][]float64 (sequence) or map[string]float64 (single time step)
	var sequence [][]float64

	switch v := inputData.(type) {
	case [][]float64:
		sequence = v
	case map[string]float64:
		// Convert map to []float64
		inputSlice := make([]float64, len(v))
		i := 0
		for _, val := range v {
			inputSlice[i] = val
			i++
		}
		sequence = [][]float64{inputSlice}
	default:
		// Handle error
		return nil
	}

	// Initialize hidden state and cell state
	var hiddenState []float64
	var cellState []float64

	// Assuming all LSTM cells have the same dimensions
	numCells := len(layer.LSTMCells)
	hiddenState = make([]float64, numCells)
	cellState = make([]float64, numCells)

	for _, timeStepInput := range sequence {
		// For each LSTM cell, compute the new hidden state and cell state
		newHiddenState := make([]float64, numCells)
		newCellState := make([]float64, numCells)

		for i, cell := range layer.LSTMCells {
			// Compute input gate, forget gate, output gate, and cell candidate
			// Assuming weights and inputs are compatible
			inputGate := bp.sigmoid(bp.dotProduct(cell.InputWeights, timeStepInput) + cell.Bias)
			forgetGate := bp.sigmoid(bp.dotProduct(cell.ForgetWeights, timeStepInput) + cell.Bias)
			outputGate := bp.sigmoid(bp.dotProduct(cell.OutputWeights, timeStepInput) + cell.Bias)
			cellCandidate := bp.tanh(bp.dotProduct(cell.CellWeights, timeStepInput) + cell.Bias)

			newCellState[i] = forgetGate*cellState[i] + inputGate*cellCandidate
			newHiddenState[i] = outputGate * bp.tanh(newCellState[i])
		}

		hiddenState = newHiddenState
		cellState = newCellState
	}

	// Return the final hidden state as a map[string]float64
	output := make(map[string]float64)
	for i, value := range hiddenState {
		output["lstm"+strconv.Itoa(i)] = value
	}

	return output
}
