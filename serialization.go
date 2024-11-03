package blueprint

import (
	"encoding/json"
	"fmt"
)

// Serialize converts the Blueprint's NetworkConfig to a JSON string.
func (bp *Blueprint) Serialize() string {
	data, err := json.Marshal(bp.Config)
	if err != nil {
		fmt.Printf("Error serializing network configuration: %v\n", err)
		return ""
	}
	return string(data)
}

// Deserialize loads a JSON string into the Blueprint's NetworkConfig.
func (bp *Blueprint) Deserialize(data string) {
	var config NetworkConfig
	err := json.Unmarshal([]byte(data), &config)
	if err != nil {
		fmt.Printf("Error deserializing network configuration: %v\n", err)
		return
	}
	bp.Config = &config
}
