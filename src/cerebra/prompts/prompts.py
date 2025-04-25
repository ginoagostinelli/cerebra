from dataclasses import dataclass


@dataclass
class SystemPrompts:

    role = "You are {role}."
    description = "{description}\n"
    instructions = "Your task is to follow the next instructions: {instructions}"
    output_format = """\n\nYou must follow the output format provided:
<output_format>
{output_format}
</output_format>
    """
    group = """You are part of a team of agents working together to complete a task.
Follow your instructions carefully. Below is the available context.

<context>
{context}
</context>"""
    tools = """\nYou operate in a continuous loop following these steps:
- <thought>here you explain your reasoning towards solving the task</thought>
- <tool_call>here you can call the tool that you want to use</tool_call>
If you decide you call a tool you will receive the following after the tool is executed:
- <observation>the result of the action</observation>."
Finally:
- When you have a final answer, provide it in <response></response> tags.

For each function call, output a JSON object enclosed in <tool_call></tool_call> tags. The JSON object must include:
   - "name": the exact function name
   - "arguments": an object with key-value pairs for parameters
   - "id": monotonically increasing integer ID starting from 0

The available functions are provided between the following tags:
<tools>
{tools}
</tools>

Important:
You should never make up tools that are not in the list.
If the user's request does not relate to any available tool, provide a free-form response wrapped in <response></response> tags.
If you call a tool, you can't give the final response and vice versa."""

    tool_example = """\n<example>
First call:
<user>What will the weather be like in New York and Philadelphia tomorrow? I need to know if I should pack an umbrella.</user>
<thought>I need to check the weather forecast for both New York and Philadelphia.</thought>
<tool_call>{"name": "get_weather_forecast", "arguments": {"location": "NY", "unit": "celsius", "days": 1}, "id": 0}</tool_call>
<tool_call>{"name": "get_weather_forecast", "arguments": {"location": "PHL", "unit": "celsius", "days": 1}, "id": 1}</tool_call>

You will be called again with this:
<observation>{0: {"temperature": 22, "conditions": "partly cloudy", "precipitation_chance": 10}}</observation>
<observation>{1: {"temperature": 24, "conditions": "scattered showers", "precipitation_chance": 60}}</observation>

Second call:
<thought>I have both forecasts now and can provide a complete answer about umbrella needs</thought>
<response>Tomorrow in New York will be partly cloudy with a low 10 percent chance of rain (22°C), while Philadelphia will have scattered showers with a 60 percent chance of rain (24°C).
You should definitely pack an umbrella if you're going to Philadelphia, but you probably won't need it in New York.</response>
</example>"""
