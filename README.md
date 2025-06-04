# Griptape-token-counter
This is a Live API Token Counter node for ComfyUI that provides real-time token usage monitoring by making actual API calls to OpenAI's services. Here's what it does:
Core Function:

Makes live API requests to count tokens and calculate costs in real-time
Integrates with both OpenAI API directly and Griptape framework (if available)

Key Features:

Real-time token counting - Gets actual usage data from API responses
Live cost calculation - Estimates costs based on current pricing for different models
Model support - Works with GPT-4o, GPT-4, GPT-3.5-turbo, O1 models, etc.
Comprehensive analysis - Provides detailed breakdown of input/output tokens, costs, and efficiency metrics
Live model data - Can fetch current available models from OpenAI API

Inputs:

Text to analyze
API key
Model selection
System message options
Whether to fetch live model data

Outputs:

Token counts (input, output, total)
Estimated costs
Detailed analysis report
API success status
Raw response data

The node is designed for users who need accurate, up-to-date token usage information rather than estimates, making it useful for cost monitoring and API usage optimization in AI workflows.
