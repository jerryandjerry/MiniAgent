from openai import OpenAI
import os
from typing import List, Dict, Any, Optional

class LLMManager:
    """Manage the configured OpenAI-compatible model endpoints."""

    # Provider-specific credential environment variables.
    _KEY_ENV = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
        "qwen": "DASHSCOPE_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "glm": "ZHIPU_API_KEY",
    }

    
    def __init__(self):
        self.models = {
            "gpt-5": {
                "api_key": "",
                "base_url": "https://api.openai.com/v1",
                "model_name": "gpt-5-2025-08-07",
                "provider": "openai"
            },
            "claude-opus": {
                "api_key": "",
                "base_url": "https://api.anthropic.com/v1/",
                "model_name": "claude-opus-4-1-20250805",
                "provider": "anthropic"
            },
            "gemini-pro": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model_name": "gemini-2.5-pro",
                "provider": "google",
                "extra_params": {"reasoning_effort": "low"}
            },
            "gemini-flash": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model_name": "gemini-2.5-flash",
                "provider": "google",
                "extra_params": {"reasoning_effort": "low"}
            },
            "qwen-plus": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen-plus",
                "provider": "qwen"
            },
            "qwen-32b": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen3-32b",
                "provider": "qwen"
            },
            "qwen-14b": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "qwen3-14b",
                "provider": "qwen"
            },
            "deepseek-v3": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "deepseek-v3.1",
                "provider": "deepseek"
            },
            "deepseek-r1": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "deepseek-r1",
                "provider": "deepseek",
                "enable_thinking": True
            },
            "glm-4.5": {
                "api_key": "",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model_name": "glm-4.5",
                "provider": "glm",
                "enable_thinking": True
            }
        }
        self._load_keys_from_env()

    def _load_keys_from_env(self):
        """Load each provider credential from the application environment."""

        from dotenv import load_dotenv

        from data.paths import PATHS

        load_dotenv(PATHS.env_file, override=False)

        for cfg in self.models.values():

            if not cfg.get("api_key"):

                cfg["api_key"] = os.getenv(self._KEY_ENV.get(cfg.get("provider"), ""), "")


    def get_available_models(self) -> List[str]:
        """Return the configured model aliases."""
        return list(self.models.keys())

    def call_model(self, model_name: str, prompt: str = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, tools: Optional[List[Dict]] = None, tool_call_handler = None, stream: bool = False, show_thinking: bool = True) -> str:
        """Invoke a configured model with either messages or a text prompt."""
        if model_name not in self.models:
            raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {self.get_available_models()}")
        
        model_config = self.models[model_name]
        
        if messages is not None:
            final_messages = messages
        else:
            final_messages = []
            if system_prompt:
                final_messages.append({"role": "system", "content": system_prompt})
            if prompt:
                final_messages.append({"role": "user", "content": prompt})
        
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        if model_config["provider"] in ["openai", "anthropic", "google"]:
            return self._call_standard_model(client, model_config, final_messages, tools, tool_call_handler, stream)
        elif model_config["provider"] in ["qwen"]:
            return self._call_qwen_model(client, model_config, final_messages, tools, tool_call_handler, stream)
        elif model_config["provider"] in ["deepseek", "glm"] and model_config.get("enable_thinking", False):
            return self._call_thinking_model(client, model_config, final_messages, stream, show_thinking)
        else:
            return self._call_standard_model(client, model_config, final_messages, tools, tool_call_handler, stream)

    def _call_standard_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], tools: Optional[List[Dict]], tool_call_handler, stream: bool) -> str:
        """Invoke a standard chat-completions model."""
        try:
            extra_params = model_config.get("extra_params", {})
            
            if stream:
                completion = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    stream=True,
                    **extra_params
                )
                
                content_parts = []
                for chunk in completion:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        print(content, end="", flush=True)
                        content_parts.append(content)
                
                print()
                return "".join(content_parts)
            else:
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    **extra_params
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"调用模型时出错: {e}"

    def _call_qwen_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], tools: Optional[List[Dict]], tool_call_handler, stream: bool) -> str:
        """Invoke a Qwen chat-completions model."""
        try:
            if stream:
                completion = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages,
                    stream=True
                )
                
                content_parts = []
                print("AI: ", end="", flush=True)
                
                for chunk in completion:
                    if chunk.choices:
                        content = chunk.choices[0].delta.content or ""
                        print(content, end="", flush=True)
                        content_parts.append(content)
                
                print()
                return "".join(content_parts)
            else:
                response = client.chat.completions.create(
                    model=model_config["model_name"],
                    messages=messages
                )
                return response.choices[0].message.content
                
        except Exception as e:
            return f"调用Qwen模型时出错: {e}"

    def _call_thinking_model(self, client: OpenAI, model_config: Dict, messages: List[Dict], stream: bool, show_thinking: bool) -> str:
        """Invoke a model that returns separate reasoning and answer streams."""
        try:
            completion = client.chat.completions.create(
                model=model_config["model_name"],
                messages=messages,
                extra_body={"enable_thinking": True},
                stream=True,
                stream_options={"include_usage": True}
            )

            reasoning_content = ""
            answer_content = ""
            is_answering = False
            
            if show_thinking:
                print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

            for chunk in completion:
                if not chunk.choices:
                    if show_thinking:
                        print("\n" + "="*20+"Usage"+"="*20)
                        print(chunk.usage)
                    continue
                    
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if show_thinking and not is_answering:
                        print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    if show_thinking and not is_answering:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    if show_thinking:
                        print(delta.content, end="", flush=True)
                    answer_content += delta.content

            if show_thinking:
                print()
            
            return f"[思考]\n{reasoning_content}\n\n[结果]\n{answer_content}" if reasoning_content else answer_content
            
        except Exception as e:
            return f"调用思考模型时出错: {e}"


llm_manager = LLMManager()

def call_llm(model_name: str, prompt: str = None, system_prompt: Optional[str] = None, messages: Optional[List[Dict]] = None, stream: bool = False, show_thinking: bool = True) -> str:
    """Invoke a configured model through the shared manager."""
    # Preserve streaming and thinking controls through explicit keyword arguments.
    return llm_manager.call_model(model_name, prompt, system_prompt, messages,
                                  stream=stream, show_thinking=show_thinking)

def get_model_list() -> List[str]:
    """Return the configured model aliases."""
    return llm_manager.get_available_models()

if __name__ == "__main__":
    print("可用模型:", get_model_list())
    response = call_llm("gpt-5", "你好，请介绍一下自己")
    print(response)
    
    
    
