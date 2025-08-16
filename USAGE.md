# Unified Agent Detection System

## Overview
This system provides unified, context-aware agent detection for categorizing user queries into appropriate agent types.

## Agent Types
- **Emotion Analysis**: Emotional support and analysis
- **Coding**: Programming and development tasks
- **Physics/Chemistry**: Science questions and formulas
- **Math**: Mathematical problems and calculations
- **General Conversation**: General queries and conversation

## Usage

### Basic Usage
```python
from unified_agent_detector import detect_agent

query = "Calculate the pH of a 0.1M HCl solution"
agent = detect_agent(query)
print(f"Agent: {agent}")  # Output: Physics/Chemistry
```

### Integration with Streamlit
Replace the existing `detect_agent` function in `app.py` with the unified system:

```python
from unified_agent_detector import detect_agent

# Use directly in your Streamlit app
agent_choice = detect_agent(combined_text)
```

## Testing
Run the verification test:
```bash
python verify_system.py
```

## Key Features
- **Context-aware detection** - Considers surrounding context
- **False positive filtering** - Reduces misclassification
- **Comprehensive patterns** - Covers edge cases
- **Easy to extend** - Simple pattern addition
- **Consistent results** - Single source of truth

## Adding New Patterns
Edit `unified_agent_detector.py` and add patterns to the appropriate category in the `context_patterns` dictionary.
