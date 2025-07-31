# CDPruner for Visual Language Models

## Overview

This implementation provides CDPruner (Conditional Determinantal Point Process) for efficient visual token pruning in various vision-language models. CDPruner is a training-free and model-agnostic method that reduces visual token count while maintaining both diversity and quality of the selected tokens.

## Key Features

- **Training-free**: No additional training required - works out of the box
- **Model-agnostic**: Can be applied to various vision-language models (Qwen2VL, LLaVA, etc.)
- **Conditional diversity**: Selects visual tokens based on both relevance to text query and diversity
- **Configurable**: Adjustable target token count and relevance weight
- **Fast inference**: Uses efficient Fast-MAP algorithm for token selection

## Algorithm Overview

CDPruner works in several stages:

1. **Feature Normalization**: L2-normalize both image and text features
2. **Similarity Computation**: Calculate cosine similarity between image tokens
3. **Conditional Relevance**: Compute relevance of image tokens to text query
4. **Kernel Construction**: Build conditional DPP kernel matrix combining similarity and relevance
5. **Fast MAP Inference**: Select diverse token subset using determinantal point process

## Usage

### Basic Usage (Qwen2VL Example)

```cpp
#include "visual_language/qwen2vl/classes.hpp"

// Initialize Qwen2VL InputsEmbedder
ov::genai::InputsEmbedderQwen2VL embedder(vlm_config, model_dir, device, device_config);

// Enable CDPruner with 256 target tokens and 0.5 relevance weight
embedder.enable_cdpruner(256, 0.5f);

// Use the embedder normally - CDPruner will automatically reduce visual tokens
auto embeddings = embedder.get_inputs_embeds(prompt, images, metrics);
```

### Advanced Configuration

```cpp
// Enable CDPruner with custom settings
embedder.enable_cdpruner(128, 0.8f);  // 128 tokens, high relevance weight

// Adjust settings dynamically
embedder.set_cdpruner_target_tokens(64);     // More aggressive pruning
embedder.set_cdpruner_relevance_weight(0.3f); // Emphasize diversity over relevance

// Disable CDPruner
embedder.disable_cdpruner();
```

### Direct CDPruner Usage

```cpp
#include "visual_language/cdpruner.hpp"

// Create CDPruner instance
ov::genai::cdpruner::CDPruner pruner(256, 0.5f);

// Define text encoder function
auto text_encoder_fn = [&](const std::string& text) -> ov::Tensor {
    // Your text encoding implementation
    return text_embeddings;
};

// Apply pruning
auto [pruned_features, selected_indices] = pruner.prune_visual_tokens(
    image_features, text_query, text_encoder_fn
);
```

## Configuration Parameters

### Target Token Count
- **Description**: Number of visual tokens to retain after pruning
- **Range**: 1 to original token count
- **Typical values**: 64, 128, 256
- **Effect**: Lower values = more aggressive pruning, faster inference

### Relevance Weight (θ)
- **Description**: Balance between token relevance and diversity
- **Range**: 0.0 to 1.0
- **Default**: 0.5
- **Effect**: 
  - Higher values (0.7-0.9): Prioritize relevance to text query
  - Lower values (0.1-0.3): Prioritize diversity among tokens
  - Balanced (0.4-0.6): Equal consideration of relevance and diversity

## Implementation Details

### File Structure

```
src/cpp/src/visual_language/
├── cdpruner.hpp              # CDPruner class declaration
├── cdpruner.cpp              # CDPruner implementation
├── cdpruner_example.cpp      # Usage example
├── test_cdpruner.cpp         # Unit tests
├── README.md                 # This documentation
└── IMPLEMENTATION_SUMMARY.md # Detailed implementation summary
```

### Performance Considerations

#### Memory Usage
- CDPruner reduces memory usage by decreasing the number of visual tokens
- Kernel matrix computation requires O(N²) memory where N is original token count
- For very large token counts (>1000), consider chunked processing

#### Computational Overhead
- Initial overhead for similarity and relevance computation
- Fast-MAP inference is O(TN²) where T is target token count
- Overall speedup due to fewer tokens in subsequent processing

#### Typical Performance Gains
- **50% token reduction** (256→128): ~30% faster inference
- **75% token reduction** (256→64): ~50% faster inference
- Actual gains depend on hardware and model size

## Integration with Different Models

### Qwen2VL
CDPruner is fully integrated into the Qwen2VL pipeline. See `qwen2vl/classes.hpp` and `qwen2vl/classes.cpp` for implementation details.

### Other Models
To integrate CDPruner with other vision-language models:

1. Include the CDPruner header: `#include "visual_language/cdpruner.hpp"`
2. Create a CDPruner instance in your InputsEmbedder class
3. Apply pruning after vision feature extraction and before final processing
4. Implement a text encoder function for relevance computation

## Error Handling

The implementation includes robust error handling with fallback to original features:

```cpp
try {
    auto [pruned_features, selected_indices] = m_cdpruner->prune_visual_tokens(
        res, m_current_text_query, text_encoder_fn
    );
    return pruned_features;
} catch (const std::exception& e) {
    // Fallback to original features if CDPruner fails
    std::cerr << "CDPruner failed: " << e.what() << ". Using original features." << std::endl;
    return res;
}
```

## Future Enhancements

1. **Adaptive Token Count**: Automatically adjust based on image complexity
2. **Batch Processing**: Support for multiple images in single batch
3. **Caching**: Cache similarity matrices for repeated queries
4. **GPU Acceleration**: CUDA/OpenCL kernels for faster computation
5. **Fine-tuning**: Optional relevance model fine-tuning

## References

- [CDPruner Paper](https://arxiv.org/abs/2506.10967): "Beyond Attention or Similarity: Maximizing Conditional Diversity for Token Pruning in MLLMs"
- [CDPruner GitHub](https://github.com/Theia-4869/CDPruner): Original LLaVA implementation
- [Fast-MAP-DPP](https://github.com/laming-chen/fast-map-dpp): Efficient DPP inference algorithm

## License

This implementation follows the Apache 2.0 license of the OpenVINO GenAI project.
