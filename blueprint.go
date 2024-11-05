package blueprint

import (
	"math"
)

// Blueprint is the main struct containing the model and related functions.
type Blueprint struct {
	Config *NetworkConfig
}

// NewBlueprint creates a new instance of Blueprint with a given configuration.
func NewBlueprint(config *NetworkConfig) *Blueprint {
	return &Blueprint{
		Config: config,
	}
}

// Connection represents a connection between two neurons with a weight.
type Connection struct {
	Weight float64 `json:"weight"`
}

// Neuron represents a neuron with an activation function, connections, and a bias.
type Neuron struct {
	ActivationType string                `json:"activationType"`
	Connections    map[string]Connection `json:"connections"`
	Bias           float64               `json:"bias"`
}

// Filter represents a convolutional filter (kernel).
type Filter struct {
	Weights [][]float64 `json:"weights"`
	Bias    float64     `json:"bias"`
}

// LSTMCell represents a cell in an LSTM layer.
type LSTMCell struct {
	InputWeights  []float64 `json:"inputWeights"`
	ForgetWeights []float64 `json:"forgetWeights"`
	OutputWeights []float64 `json:"outputWeights"`
	CellWeights   []float64 `json:"cellWeights"`
	Bias          float64   `json:"bias"`
}

// Layer represents a layer in the network.
type Layer struct {
	LayerType string            `json:"layerType"`
	Neurons   map[string]Neuron `json:"neurons,omitempty"` // For dense layers
	Filters   []Filter          `json:"filters,omitempty"` // For convolutional layers
	Stride    int               `json:"stride,omitempty"`
	Padding   int               `json:"padding,omitempty"`
	LSTMCells []LSTMCell        `json:"lstmCells,omitempty"`
}

// ModelMetadata holds metadata for the model.
type ModelMetadata struct {
	ModelID     string `json:"modelID"`
	ProjectName string `json:"projectName"`

	LastTrainingAccuracy        float64 `json:"lastTrainingAccuracy"`
	LastTestAccuracy            float64 `json:"lastTestAccuracy"`
	LastTestAccuracyGenerous    float64 `json:"lastTestAccuracyGenerous"`
	LastTestAccuracyForgiveness float64 `json:"lastTestAccuracyForgiveness"`
	ForgivenessThreshold        float64 `json:"forgivenessThreshold"`

	Path           string   `json:"path"`
	Evaluated      bool     `json:"evaluated"`
	ParentModelIDs []string `json:"parentModelIDs"`
	ChildModelIDs  []string `json:"childModelIDs"`

	// New fields for neuron and layer counts using int64 for large values
	TotalNeurons int64 `json:"totalNeurons"`
	TotalLayers  int64 `json:"totalLayers"`
}

// NetworkConfig represents the structure of the neural network, containing input, hidden, and output layers, and model metadata.
type NetworkConfig struct {
	Metadata ModelMetadata `json:"metadata"`
	Layers   struct {
		Input  Layer   `json:"input"`
		Hidden []Layer `json:"hidden"`
		Output Layer   `json:"output"`
	} `json:"layers"`
}

// Activate calculates the activation value based on the activation type.
func (bp *Blueprint) Activate(activationType string, input float64) float64 {
	switch activationType {
	case "relu":
		return math.Max(0, input)
	case "sigmoid":
		return 1 / (1 + math.Exp(-input))
	case "tanh":
		return math.Tanh(input)
	case "softmax":
		return math.Exp(input) // Should normalize later in the layer processing
	case "leaky_relu":
		if input > 0 {
			return input
		}
		return 0.01 * input
	case "swish":
		return input * (1 / (1 + math.Exp(-input))) // Beta set to 1 for simplicity
	case "elu":
		alpha := 1.0 // Alpha can be adjusted based on specific needs
		if input >= 0 {
			return input
		}
		return alpha * (math.Exp(input) - 1)
	case "selu":
		lambda := 1.0507    // Scale factor
		alphaSELU := 1.6733 // Alpha for SELU
		if input >= 0 {
			return lambda * input
		}
		return lambda * (alphaSELU * (math.Exp(input) - 1))
	case "softplus":
		return math.Log(1 + math.Exp(input))
	default:
		return input // Linear activation (no change)
	}
}

// Feedforward processes the input values through the network and returns the output values.
func (bp *Blueprint) Feedforward(inputValues map[string]interface{}) map[string]float64 {
	// Load input values into the data variable
	var data interface{}

	switch bp.Config.Layers.Input.LayerType {
	case "dense":
		inputData := make(map[string]float64)
		for k, v := range inputValues {
			if val, ok := v.(float64); ok {
				inputData[k] = val
			} else {
				return nil
			}
		}
		data = inputData
	case "conv":
		if imageData, ok := inputValues["image"].([][]float64); ok {
			data = imageData
		} else {
			return nil
		}
	case "lstm":
		if sequenceData, ok := inputValues["sequence"].([][]float64); ok {
			data = sequenceData
		} else {
			return nil
		}
	}

	// Process hidden layers
	for _, layer := range bp.Config.Layers.Hidden {
		data = bp.ProcessLayer(layer, data)
	}

	// Process output layer
	outputLayer := bp.Config.Layers.Output
	data = bp.ProcessLayer(outputLayer, data)

	// Return output values
	if outputData, ok := data.(map[string]float64); ok {
		return outputData
	}
	return nil
}

// ProcessLayer handles processing of each layer type within the network
func (bp *Blueprint) ProcessLayer(layer Layer, inputData interface{}) interface{} {
	switch layer.LayerType {
	case "dense":
		return bp.processDenseLayer(layer, inputData)
	case "conv":
		return bp.processConvLayer(layer, inputData)
	case "lstm":
		return bp.processLSTMLayer(layer, inputData)
	default:
		return nil
	}
}

func (bp *Blueprint) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (bp *Blueprint) tanh(x float64) float64 {
	return math.Tanh(x)
}

func (bp *Blueprint) dotProduct(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		// Handle error
		return 0
	}
	sum := 0.0
	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}
	return sum
}
