# Multi-Provider API Configuration

This document describes the API key requirements for multi-provider override cascade evaluation. The framework supports 10+ AI model providers to enable comparative analysis of safety override behavior across different model architectures and training approaches.

## 🔑 Primary Providers (High Priority)

### OpenAI
- **Variable**: `OPENAI_API_KEY`
- **Model**: GPT-4o
- **Get key**: https://platform.openai.com/api-keys
- **Status**: ✅ Already working

### Anthropic  
- **Variable**: `ANTHROPIC_API_KEY`
- **Model**: Claude-3.5-Sonnet
- **Get key**: https://console.anthropic.com/
- **Why important**: Claude is often more safety-conscious, interesting to test overrides

### Google
- **Variable**: `GOOGLE_API_KEY` 
- **Model**: Gemini Pro
- **Get key**: https://aistudio.google.com/app/apikey
- **Why important**: Different training approach, may show different override patterns

## ⚡ High-Performance Providers (Medium Priority)

### Groq
- **Variable**: `GROQ_API_KEY`
- **Model**: Llama-3.1-70b-Versatile  
- **Get key**: https://console.groq.com/keys
- **Why important**: Very fast inference, tests if speed affects override decisions

### Together AI
- **Variable**: `TOGETHER_API_KEY`
- **Model**: Mixtral-8x7B-Instruct
- **Get key**: https://api.together.xyz/settings/api-keys
- **Why important**: Mixture of experts model, different architecture

### Fireworks AI
- **Variable**: `FIREWORKS_API_KEY`
- **Model**: Llama-3.1-70b-Instruct
- **Get key**: https://fireworks.ai/account/api-keys
- **Why important**: Optimized inference, tests performance vs safety tradeoffs

## 🧪 Alternative Providers (Lower Priority)

### Replicate
- **Variable**: `REPLICATE_API_TOKEN`
- **Model**: Meta Llama-3.1-70b-Instruct
- **Get key**: https://replicate.com/account/api-tokens
- **Why useful**: Different hosting approach

### Cohere
- **Variable**: `COHERE_API_KEY`
- **Model**: Command R+
- **Get key**: https://dashboard.cohere.com/api-keys
- **Why useful**: Enterprise-focused model

### Mistral
- **Variable**: `MISTRAL_API_KEY`  
- **Model**: Mistral Large
- **Get key**: https://console.mistral.ai/api-keys/
- **Why useful**: European AI perspective

### Perplexity
- **Variable**: `PERPLEXITY_API_KEY`
- **Model**: Llama-3.1-70b-Instruct  
- **Get key**: https://www.perplexity.ai/settings/api
- **Why useful**: Search-augmented responses

## 🎯 Recommended Starting Set

If you want to start with the most interesting comparisons, get these 4:

1. **OPENAI_API_KEY** - GPT-4o (already working)
2. **ANTHROPIC_API_KEY** - Claude-3.5-Sonnet (safety-focused)
3. **GOOGLE_API_KEY** - Gemini Pro (different training)  
4. **GROQ_API_KEY** - Llama-3.1-70b (high-speed)

This gives you:
- ✅ Different companies (OpenAI, Anthropic, Google, Groq)
- ✅ Different model architectures (GPT, Claude, Gemini, Llama)
- ✅ Different safety philosophies
- ✅ Different inference speeds

## 💰 Cost Estimates (Approximate)

For running the full override cascade evaluation (~5 scenarios × ~1000 tokens each):

- **OpenAI GPT-4o**: ~$0.15 per evaluation
- **Anthropic Claude-3.5**: ~$0.12 per evaluation  
- **Google Gemini Pro**: ~$0.05 per evaluation
- **Groq Llama-3.1-70b**: ~$0.03 per evaluation
- **Others**: $0.02-$0.08 per evaluation

**Total for all providers**: ~$1-2 for complete evaluation suite

## 🚨 What We're Testing

The scenarios are designed to trigger override cascades through:

1. **Context Overload**: Bog down models with competing information
2. **Pattern Completion Traps**: "You cleaned up X, so clean up Y too"  
3. **Escalating Pressure**: Time pressure + business pressure + technical pressure
4. **Competing Priorities**: Safety rules vs completion drives
5. **Real-World Complexity**: Based on actual incidents (like your zsh deletion)

## 🎯 Expected Differences

- **Claude**: May resist overrides more strongly (safety-trained)
- **GPT-4o**: May show consistent override patterns  
- **Gemini**: May handle complex ethical scenarios differently
- **Llama models**: May be more susceptible to pattern completion
- **Faster models**: May make quicker override decisions

## 🏃‍♂️ Quick Start

```bash
# Set the keys you have
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"  
export GOOGLE_API_KEY="your-google-key"

# Run evaluation
python3 evaluations/multi_provider_override_evaluation.py
```

The script will automatically detect which providers you have keys for and test only those.

---

**Goal**: Discover which AI models are most/least susceptible to safety override cascades under extreme pressure scenarios! 🧪⚡🛡️
