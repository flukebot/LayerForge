package blueprint

import "fmt"

func (bp *Blueprint) processConvLayer(layer Layer, inputData interface{}) interface{} {
	// inputData is expected to be [][]float64 (2D image) or [][][]float64 (multiple feature maps)
	inputImages, ok := inputData.([][][]float64)
	if !ok {
		// Try to convert single image to array of images
		if singleImage, ok := inputData.([][]float64); ok {
			inputImages = [][][]float64{singleImage}
		} else {
			// Handle error
			return nil
		}
	}

	outputFeatureMaps := [][][]float64{}

	for _, filter := range layer.Filters {
		featureMapsForFilter := [][][]float64{}
		for _, inputImage := range inputImages {
			featureMap := bp.convolve(inputImage, filter.Weights, layer.Stride, layer.Padding)
			// Apply activation function to each element in featureMap
			for i := range featureMap {
				for j := range featureMap[i] {
					featureMap[i][j] = bp.Activate("relu", featureMap[i][j]+filter.Bias) // Assuming ReLU activation
				}
			}
			featureMapsForFilter = append(featureMapsForFilter, featureMap)
		}
		// For simplicity, just append them
		outputFeatureMaps = append(outputFeatureMaps, featureMapsForFilter...)
	}

	// Flatten outputFeatureMaps into map[string]float64
	flattenedOutput := make(map[string]float64)
	idx := 0
	for _, featureMap := range outputFeatureMaps {
		for i := range featureMap {
			for j := range featureMap[i] {
				key := fmt.Sprintf("conv_output%d", idx)
				flattenedOutput[key] = featureMap[i][j]
				idx++
			}
		}
	}

	return flattenedOutput
}

func (bp *Blueprint) convolve(input [][]float64, kernel [][]float64, stride int, padding int) [][]float64 {
	// Pad the input if padding > 0
	paddedInput := bp.pad2D(input, padding)

	inputHeight := len(paddedInput)
	inputWidth := len(paddedInput[0])

	kernelHeight := len(kernel)
	kernelWidth := len(kernel[0])

	// Calculate output dimensions
	outputHeight := (inputHeight-kernelHeight)/stride + 1
	outputWidth := (inputWidth-kernelWidth)/stride + 1

	output := make([][]float64, outputHeight)
	for i := 0; i < outputHeight; i++ {
		output[i] = make([]float64, outputWidth)
		for j := 0; j < outputWidth; j++ {
			sum := 0.0
			for ki := 0; ki < kernelHeight; ki++ {
				for kj := 0; kj < kernelWidth; kj++ {
					sum += paddedInput[i*stride+ki][j*stride+kj] * kernel[ki][kj]
				}
			}
			output[i][j] = sum
		}
	}

	return output
}

func (bp *Blueprint) pad2D(input [][]float64, padding int) [][]float64 {
	if padding == 0 {
		return input
	}

	inputHeight := len(input)
	inputWidth := len(input[0])

	paddedHeight := inputHeight + 2*padding
	paddedWidth := inputWidth + 2*padding

	paddedInput := make([][]float64, paddedHeight)
	for i := range paddedInput {
		paddedInput[i] = make([]float64, paddedWidth)
	}

	for i := 0; i < inputHeight; i++ {
		for j := 0; j < inputWidth; j++ {
			paddedInput[i+padding][j+padding] = input[i][j]
		}
	}

	return paddedInput
}
