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



# Enhanced API Data Fetcher
class LiveModelDataFetcher:
    """Fetch comprehensive model data from OpenAI API"""
    
    @classmethod  
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": ""}),
                "include_pricing_estimates": ("BOOLEAN", {"default": True})
            }
        }
    
    RETURN_TYPES = ("DICT", "STRING", "INT")
    RETURN_NAMES = ("all_models_data", "models_summary", "total_models")
    FUNCTION = "fetch_live_models"
    CATEGORY = "ai_tools/griptape_api"
    
    def fetch_live_models(self, api_key: str, include_pricing_estimates: bool) -> Tuple:
        """Fetch comprehensive live model data"""
        
        try:
            client = openai.OpenAI(api_key=api_key)
            
            # Fetch all models
            models_response = client.models.list()
            
            models_data = {}
            chat_models = []
            
            for model in models_response.data:
                model_info = {
                    "id": model.id,
                    "object": model.object,
                    "created": model.created,
                    "owned_by": model.owned_by,
                    "created_date": datetime.fromtimestamp(model.created).strftime('%Y-%m-%d %H:%M:%S'),
                    "is_chat_model": "gpt" in model.id.lower() or "o1" in model.id.lower()
                }
                
                if model_info["is_chat_model"]:
                    chat_models.append(model.id)
                
                models_data[model.id] = model_info
            
            # Generate summary
            summary = f"""📡 LIVE MODEL DATA SUMMARY
═══════════════════════════════════════
🕐 Fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 Total Models: {len(models_data)}
💬 Chat Models: {len(chat_models)}

🤖 AVAILABLE CHAT MODELS:
{chr(10).join(f"├─ {model}" for model in sorted(chat_models)[:10])}
{"└─ ... and more" if len(chat_models) > 10 else ""}

🏭 MODEL OWNERS:
{chr(10).join(f"├─ {owner}: {len([m for m in models_data.values() if m['owned_by'] == owner])}" 
              for owner in sorted(set(m['owned_by'] for m in models_data.values())))}

✅ Data fetched successfully from OpenAI API
"""
            
            return (models_data, summary, len(models_data))
            
        except Exception as e:
            error_summary = f"""❌ LIVE MODEL FETCH ERROR
═══════════════════════════════════════
Error: {str(e)}
Time: {datetime.now().isoformat()}
"""
            return ({}, error_summary, 0)


# ComfyUI Node Registration
NODE_CLASS_MAPPINGS = {
    
    "LiveModelDataFetcher": LiveModelDataFetcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {

    "LiveModelDataFetcher": "Live Model Data Fetcher (OpenAI API)",
}
