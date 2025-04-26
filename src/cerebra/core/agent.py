import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from textwrap import dedent

from cerebra.utils.chat_history import ChatHistory
from cerebra.utils.parsing import extract_tags
from cerebra.prompts.prompts import SystemPrompts
from cerebra.api_client import APIClient


class Agent:
    def __init__(
        self,
        name,
        role: str,
        description: str,
        instructions,
        output_format: str = "",
        tools: List[Any] = None,
        model: str = "llama-3.3-70b-versatile",
        verbose: bool = False,
    ):
        if not name:
            raise ValueError("Agent name cannot be empty.")

        self.name = name
        self.role: str = role
        self.description: str = description
        self.instructions = instructions
        self.output_format: str = output_format
        self.tools = tools or []
        self.model = model
        self.verbose: bool = verbose

        self.chat_history: Optional[ChatHistory] = None

        try:
            self.client = APIClient.get_client()
        except ValueError as e:
            raise ValueError(f"API client setup failed for agent '{self.name}': {e}") from e
        except Exception as e:
            raise Exception(f"Unexpected error initializing API client for agent '{self.name}': {e}") from e

    def _build_system_prompt(self, inputs: Optional[Dict] = None):
        inputs = inputs if inputs else {}

        try:
            formats = {
                "role": self.role.format(**inputs) if self.role else "",
                "description": self.description.format(**inputs) if self.description else "",
                "instructions": self.instructions.format(**inputs) if self.instructions else "",
            }
        except KeyError as e:
            raise ValueError(f"Input dictionary missing required key for agent '{self.name}' prompt formatting: {e}") from e

        system_prompt = ""

        if self.role:
            system_prompt += SystemPrompts.role.format(**formats)

        if self.description:
            system_prompt += SystemPrompts.description.format(**formats)

        if self.instructions:
            system_prompt += SystemPrompts.instructions.format(**formats)

        if self.tools:
            try:
                tool_signatures = self.get_tool_signatures()
                tools_format = {"tools": tool_signatures}
                system_prompt += SystemPrompts.tools.format(**tools_format)
                system_prompt += SystemPrompts.tool_example
            except Exception as e:
                raise Exception(f"Could not generate tool signatures for agent '{self.name}': {e}") from e

        if self.output_format:
            out_format = {"output_format": self.output_format}
            system_prompt += SystemPrompts.output_format.format(**out_format)

        return system_prompt

    def get_tool_signatures(self) -> str:
        tool_signatures = [{"function": tool.tool.signature()} for tool in self.tools]
        return json.dumps(tool_signatures, indent=2)

    def invoke(self, messages: list) -> str:
        try:
            response = self.client.chat.completions.create(messages=messages, model=self.model)

            if not response.choices or not response.choices[0].message or response.choices[0].message.content is None:
                raise Exception(f"API response for agent '{self.name}' missing expected content.")

            content = str(response.choices[0].message.content)
            return content
        except Exception as e:
            raise Exception(f"Agent '{self.name}' execution stopped due to API error: {e}") from e

    def process_tool_calls(self, tool_calls: List[str]) -> Dict:
        results: Dict[Any, Any] = {}
        for call_str in tool_calls:
            try:
                call = json.loads(call_str)
                name = call.get("name")
                arguments = call.get("arguments", {})
                call_id = call.get("id")
                matching_tool = next((t for t in self.tools if t.tool.name == name), None)
                if matching_tool is None:
                    results[call_id] = f"Tool {name} not found."
                else:
                    result = matching_tool(**arguments)
                    results[call_id] = result
            except json.JSONDecodeError as e:
                results[None] = f"Error decoding tool call JSON: {str(e)}"
            except Exception as e:
                call_id = call.get("id") if isinstance(call, dict) else None
                results[call_id] = f"Error processing tool call: {str(e)}"
        return results

    def _build_prompt(self, context: Any = None):
        from textwrap import dedent

        context_str = ""
        if context is None:
            context_str = "No context provided."
        elif isinstance(context, dict):
            context_str = "Received context from multiple sources:\n"
            try:
                context_str += json.dumps(context, indent=2)
            except TypeError:
                context_str += str(context)
        elif isinstance(context, str):
            context_str = context
        else:
            context_str = str(context)

        if not context_str:
            context_str = "No context provided"  # Ensure context_str is never empty

        formats = {"context": context_str}

        try:
            prompt = dedent(SystemPrompts.group.format(**formats)).strip()
            return prompt
        except KeyError as e:
            return f"Error building prompt: Missing key {e} in SystemPrompts.group. Provided context was: {context_str}"
        except Exception as e:
            return f"Error building prompt: {e}. Context was: {context_str}"

    def run(self, inputs: dict = None, context: Any = None, max_iterations: int = 10) -> str:
        system_prompt = self._build_system_prompt(inputs)
        self.chat_history = ChatHistory([ChatHistory.format_message(system_prompt, role="system")])

        prompt_text = self._build_prompt(context)
        self.chat_history.add_message(prompt_text, role="user")

        if self.tools:
            for iteration in range(1, max_iterations + 1):
                completion = self.invoke(self.chat_history.get_messages())
                self.chat_history.add_message(completion, "assistant")

                response = extract_tags(str(completion), "response")

                if response.found:
                    final_ans = response.content[0]
                    return final_ans

                tool_calls = extract_tags(str(completion), "tool_call")

                for tool_call in tool_calls.content:
                    print(f"- Tool Calls: {tool_call}")

                if tool_calls.found:
                    observations = self.process_tool_calls(tool_calls.content)
                    print("- Observations:")
                    for call_id, obs in observations.items():
                        print(f"  [{call_id}]: {obs}")
                    self.chat_history.add_message(json.dumps(observations, indent=2), "user")

        final_ans = self.invoke(self.chat_history.get_messages())
        return final_ans
