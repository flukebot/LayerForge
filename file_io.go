package blueprint

import (
	"encoding/json"
	"fmt"
	"os"
)

// SaveModel saves the Blueprint's NetworkConfig to a specified file.
func (bp *Blueprint) SaveModel(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create model file: %w", err)
	}
	defer file.Close()

	if err := json.NewEncoder(file).Encode(bp.Config); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	return nil
}

// LoadModel loads a NetworkConfig from a specified file into the Blueprint.
func (bp *Blueprint) LoadModel(filePath string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open model file: %w", err)
	}
	defer file.Close()

	var modelConfig NetworkConfig
	if err := json.NewDecoder(file).Decode(&modelConfig); err != nil {
		return fmt.Errorf("failed to decode model: %w", err)
	}

	bp.Config = &modelConfig
	return nil
}
