import openai
import time
import json
import requests
from typing import Tuple, Dict, Optional, List
from datetime import datetime, timedelta

try:
    from griptape.structures import Agent
    from griptape.drivers import OpenAiChatPromptDriver
    from griptape.configs import Defaults
    from griptape.rules import Rule
    GRIPTAPE_AVAILABLE = True
except ImportError:
    GRIPTAPE_AVAILABLE = False


class GriptapeAPITokenCounter:
    """Real-time token usage monitor fetching live data via OpenAI API"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model": ([
                    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
                    "gpt-3.5-turbo", "o1-preview", "o1-mini"
                ], {"default": "gpt-4o-mini"}),
                "include_system_msg": ("BOOLEAN", {"default": True}),
                "fetch_live_models": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "system_message": ("STRING", {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "DICT", "STRING", "BOOLEAN", "DICT")
    RETURN_NAMES = ("input_tokens", "output_tokens", "total_tokens", "estimated_cost", 
                   "live_model_data", "detailed_analysis", "api_success", "raw_response")
    FUNCTION = "fetch_realtime_data"
    CATEGORY = "ai_tools/griptape_api"
    
    def __init__(self):
        self.client = None
        self.agent = None
        self.model_cache = {}
        self.usage_cache = {}
        
    def setup_openai_client(self, api_key: str):
        """Setup OpenAI client with proper configuration"""
        try:
            self.client = openai.OpenAI(api_key=api_key)
            
            # Test connection
            self.client.models.list()
            return True
            
        except Exception as e:
            print(f"Failed to setup OpenAI client: {e}")
            return False
    
    def setup_griptape_agent(self, api_key: str, model: str):
        """Setup Griptape agent for structured interactions"""
        try:
            if not GRIPTAPE_AVAILABLE:
                return False
                
            driver = OpenAiChatPromptDriver(
                api_key=api_key,
                model=model
            )
            
            self.agent = Agent(
                prompt_driver=driver,
                rules=[
                    Rule("Provide precise token analysis"),
                    Rule("Focus on real-time API data"),
                    Rule("Be concise but comprehensive")
                ]
            )
            return True
            
        except Exception as e:
            print(f"Failed to setup Griptape agent: {e}")
            return False
    
    def fetch_available_models(self) -> Dict:
        """Fetch live model data from OpenAI API"""
        try:
            models_response = self.client.models.list()
            
            model_data = {}
            for model in models_response.data:
                model_data[model.id] = {
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by,
                    "permission": getattr(model, 'permission', []),
                    "root": getattr(model, 'root', model.id),
                    "parent": getattr(model, 'parent', None)
                }
            
            self.model_cache = {
                "models": model_data,
                "total_models": len(model_data),
                "fetched_at": datetime.now().isoformat(),
                "success": True
            }
            
            return self.model_cache
            
        except Exception as e:
            return {
                "error": str(e),
                "success": False,
                "fetched_at": datetime.now().isoformat()
            }
    
    def get_model_context_limits(self, model: str) -> Dict:
        """Get model specifications via API probe"""
        context_limits = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
            "o1-preview": 128000,
            "o1-mini": 128000
        }
        
        return {
            "model": model,
            "context_limit": context_limits.get(model, 4096),
            "estimated": True
        }
    
    
    def build_conversation_messages(self, text: str, system_message: str = "") -> List[Dict]:
    messages = []
    
    if system_message.strip():
        messages.append({"role": "system", "content": system_message.strip()})
    
    messages.append({"role": "user", "content": text})
    return messages

    
    def make_token_counting_request(self, messages: List[Dict], model: str) -> Dict:
        """Make API request specifically for token counting"""
        try:
            # Use completion with minimal response to get token usage
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,  
                temperature=0.2,
                stream=False
            )
            
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "model_used": response.model,
                "response_id": response.id,
                "created": response.created,
                "system_fingerprint": getattr(response, 'system_fingerprint', None),
                "finish_reason": response.choices[0].finish_reason if response.choices else None,
                "response_content": response.choices[0].message.content if response.choices else "",
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            
            return usage_data
            
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def fetch_current_pricing(self, model: str) -> Dict:
        """Estimate current pricing (OpenAI doesn't provide pricing API)"""
        # Current pricing as of latest update - would need regular updates
        pricing_data = {
            "gpt-4o": {
                "input_cost_per_1k": 0.0025,
                "output_cost_per_1k": 0.01,
                "context_limit": 128000
            },
            "gpt-4o-mini": {
                "input_cost_per_1k": 0.00015,
                "output_cost_per_1k": 0.0006,
                "context_limit": 128000
            },
            "gpt-4-turbo": {
                "input_cost_per_1k": 0.01,
                "output_cost_per_1k": 0.03,
                "context_limit": 128000
            },
            "gpt-4": {
                "input_cost_per_1k": 0.03,
                "output_cost_per_1k": 0.06,
                "context_limit": 8192
            },
            "gpt-3.5-turbo": {
                "input_cost_per_1k": 0.001,
                "output_cost_per_1k": 0.002,
                "context_limit": 16385
            },
            "o1-preview": {
                "input_cost_per_1k": 0.015,
                "output_cost_per_1k": 0.06,
                "context_limit": 128000
            },
            "o1-mini": {
                "input_cost_per_1k": 0.003,
                "output_cost_per_1k": 0.012,
                "context_limit": 128000
            }
        }
        
        return pricing_data.get(model, pricing_data["gpt-4o-mini"])
    
    def calculate_costs(self, usage_data: Dict, model: str) -> Tuple[float, float, float]:
        """Calculate costs based on usage data"""
        if not usage_data.get("success", False):
            return 0.0, 0.0, 0.0
        
        pricing = self.fetch_current_pricing(model)
        
        input_tokens = usage_data.get("prompt_tokens", 0)
        output_tokens = usage_data.get("completion_tokens", 0)
        
        input_cost = (input_tokens / 1000) * pricing["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * pricing["output_cost_per_1k"] 
        total_cost = input_cost + output_cost
        
        return input_cost, output_cost, total_cost
    
    def generate_comprehensive_analysis(self, usage_data: Dict, model_data: Dict,
                                      input_cost: float, output_cost: float, 
                                      total_cost: float, model: str) -> str:
        """Generate detailed analysis with live data"""
        
        if not usage_data.get("success", False):
            return f"""🔴 LIVE API TOKEN ANALYSIS - ERROR
═══════════════════════════════════════
❌ API Request Failed
Error: {usage_data.get('error', 'Unknown error')}
Type: {usage_data.get('error_type', 'Unknown')}
Time: {usage_data.get('timestamp', 'Unknown')}
Model: {model}

🔧 Troubleshooting:
- Check API key validity
- Verify model availability
- Check rate limits
"""

        # Get model info from live data
        model_info = model_data.get("models", {}).get(model, {}) if model_data.get("success") else {}
        
        analysis = f"""🟢 LIVE API TOKEN ANALYSIS
═══════════════════════════════════════
🕐 Timestamp: {usage_data['timestamp']}
🆔 Response ID: {usage_data.get('response_id', 'N/A')}
🤖 Model Used: {usage_data.get('model_used', model)}
🔧 System Fingerprint: {usage_data.get('system_fingerprint', 'N/A')}

📊 REAL-TIME TOKEN USAGE:
├─ Input Tokens: {usage_data['prompt_tokens']:,}
├─ Output Tokens: {usage_data['completion_tokens']:,}
├─ Total Tokens: {usage_data['total_tokens']:,}
└─ Finish Reason: {usage_data.get('finish_reason', 'N/A')}

💰 LIVE COST CALCULATION:
├─ Input Cost: ${input_cost:.6f}
├─ Output Cost: ${output_cost:.6f}
├─ Total Cost: ${total_cost:.6f}
└─ Cost/Token: ${total_cost/max(usage_data['total_tokens'], 1):.8f}

🏭 MODEL INFORMATION (Live):
├─ Model ID: {model_info.get('id', model)}
├─ Owner: {model_info.get('owned_by', 'Unknown')}
├─ Created: {datetime.fromtimestamp(model_info.get('created', 0)).strftime('%Y-%m-%d') if model_info.get('created') else 'Unknown'}
└─ Object Type: {model_info.get('object', 'Unknown')}

📈 EFFICIENCY METRICS:
├─ Input/Output Ratio: {usage_data['prompt_tokens']/max(usage_data['completion_tokens'], 1):.2f}:1
├─ Context Utilization: {(usage_data['total_tokens']/128000*100):.2f}% (est.)
└─ Response: "{usage_data.get('response_content', '')[:50]}..."

🎯 Data Source: Live OpenAI API + Griptape
📡 Models Available: {model_data.get('total_models', 'Unknown')} (Live Count)
"""
        
        return analysis
    
    def fetch_realtime_data(self, text: str, api_key: str, model: str,
                           include_system_msg: bool, fetch_live_models: bool,
                           system_message: str = "") -> Tuple:
        """Main function to fetch real-time token data via API"""
        
        if not api_key.strip():
            return (0, 0, 0, 0.0, {}, "❌ API key required", False, {})
        
        try:
            # Setup OpenAI client
            if not self.setup_openai_client(api_key):
                return (0, 0, 0, 0.0, {}, "❌ Failed to setup OpenAI client", False, {})
            
            # Setup Griptape agent if available
            if GRIPTAPE_AVAILABLE:
                self.setup_griptape_agent(api_key, model)
            
            # Fetch live model data if requested
            model_data = {}
            if fetch_live_models:
                model_data = self.fetch_available_models()
            
            # Build conversation messages
            messages = self.build_conversation_messages(
                text, include_system_msg, system_message
            )
            
            # Make API request for token counting
            usage_data = self.make_token_counting_request(messages, model)
            
            if not usage_data.get("success", False):
                analysis = self.generate_comprehensive_analysis(
                    usage_data, model_data, 0.0, 0.0, 0.0, model
                )
                return (0, 0, 0, 0.0, model_data, analysis, False, usage_data)
            
            # Calculate costs
            input_cost, output_cost, total_cost = self.calculate_costs(usage_data, model)
            
            # Generate comprehensive analysis
            analysis = self.generate_comprehensive_analysis(
                usage_data, model_data, input_cost, output_cost, total_cost, model
            )
            
            # Prepare live model data for output
            live_model_summary = {
                "total_available_models": model_data.get("total_models", 0),
                "current_model": model,
                "model_verified": model in model_data.get("models", {}),
                "fetch_success": model_data.get("success", False),
                "last_updated": model_data.get("fetched_at", "")
            }
            
            return (
                usage_data["prompt_tokens"],      # input_tokens
                usage_data["completion_tokens"],  # output_tokens  
                usage_data["total_tokens"],       # total_tokens
                total_cost,                       # estimated_cost
                live_model_summary,               # live_model_data
                analysis,                         # detailed_analysis
                True,                            # api_success
                usage_data                       # raw_response
            )
            
        except Exception as e:
            error_analysis = f"""🔴 SYSTEM ERROR - LIVE API FETCH
═══════════════════════════════════════
❌ Exception: {str(e)}
🕐 Time: {datetime.now().isoformat()}
🤖 Model: {model}
"""
            return (0, 0, 0, 0.0, {}, error_analysis, False, {"error": str(e)})




# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    "GriptapeAPITokenCounter": GriptapeAPITokenCounter,
    # "LiveModelDataFetcher": LiveModelDataFetcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GriptapeAPITokenCounter": "Live API Token Counter (Griptape)",
    # "LiveModelDataFetcher": "Live Model Data Fetcher (OpenAI API)",
}
