import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import graphviz
import concurrent.futures

from typing import List, Any
from cerebra.core import Agent


class Workflow(ABC):
    @abstractmethod
    def run(self, agents: List["Agent"], inputs: Any, context: str = None, max_iterations: int = 10) -> Any:
        pass

    def plot(self, agents: List["Agent"]) -> graphviz.Digraph:
        """
        Generates a visual representation of the agent connections for this workflow.
        Should be overridden by subclasses to show specific structure.

        Args:
            agents: List of agents involved in this workflow segment.

        Returns:
            A graphviz.Digraph object representing the internal workflow view.
        """
        dot = graphviz.Digraph(comment=f"{self.__class__.__name__} - Default View")
        dot.attr(label=f"{self.__class__.__name__}", rankdir="LR")

        if not agents:
            dot.node("empty", "No agents in this group")
            return dot

        # Default plot just shows agents without connections
        for agent in agents:
            agent_label = getattr(agent, "name", str(id(agent)))
            # Use a unique identifier for the node within this specific plot
            node_id = f"agent_{getattr(agent, 'name', str(id(agent)))}"
            dot.node(node_id, agent_label)

        if self.__class__ == Workflow:
            dot.node("info", "Default plot: Override plot() in specific Workflow", shape="plaintext")

        return dot


class SequentialWorkflow(Workflow):
    """Executes agents in a predefined linear sequence."""

    def run(self, agents: List["Agent"], inputs: Any, context: str = None, max_iterations: int = 10) -> Any:
        """
        Runs agents one after another, passing the output of one as the input to the next.
        """
        last_output = context
        agent_outputs = {}

        for agent in agents:
            print(f"\n# ---- {agent.name} ---- #")

            output_content = agent.run(inputs=inputs, context=last_output, max_iterations=max_iterations)

            last_output = output_content
            agent_outputs[agent.name] = output_content
            print(f" - Output:\n{json.dumps(output_content, indent=2)}")

        return last_output if last_output else None

    def plot(self, agents: List["Agent"]) -> graphviz.Digraph:
        """Returns a Digraph object for sequential flow."""
        dot = graphviz.Digraph(comment="Sequential Agent Workflow")
        dot.attr(label="Sequential Workflow", rankdir="LR")

        if not agents:
            dot.node("empty", "No agents")
            return dot

        for i, agent in enumerate(agents):
            agent_label = getattr(agent, "name", f"Agent_{i}")
            node_id = f"agent_{agent_label}"  # Use agent name for node ID within workflow plot
            dot.node(node_id, agent_label)
            if i > 0:
                prev_agent_label = getattr(agents[i - 1], "name", f"Agent_{i-1}")
                prev_node_id = f"agent_{prev_agent_label}"
                dot.edge(prev_node_id, node_id)
        return dot


class BroadcastWorkflow(Workflow):
    """Broadcasts inputs to all agents in parallel and collects all outputs."""

    def __init__(self, parallel: bool = True):
        """
        Initialize the broadcast workflow.

        Args:
            parallel: Whether to run agents in parallel using ThreadPoolExecutor
        """
        self.parallel = parallel

    def _run_agent(self, agent, inputs, context, max_iterations):
        """Helper to run a single agent in the parallel executor."""
        print(f"\n# ---- {agent.name} ---- #")
        output = agent.run(inputs=inputs, context=context, max_iterations=max_iterations)
        print(f" - Output:\n{json.dumps(output, indent=2)}")
        return agent.name, output

    def run(self, agents: List["Agent"], inputs: Any, context: str = None, max_iterations: int = 10) -> Dict[str, Any]:
        """
        Runs all agents in parallel with the same input.

        Returns a dictionary mapping agent names to their outputs.
        """
        results = {}

        if self.parallel and len(agents) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(agents), 10)) as executor:
                futures = {executor.submit(self._run_agent, agent, inputs, context, max_iterations): agent for agent in agents}
                for future in concurrent.futures.as_completed(futures):
                    agent_name, output = future.result()
                    results[agent_name] = output

        else:
            for agent in agents:
                agent_name, output = self._run_agent(agent, inputs, max_iterations)
                results[agent_name] = output

        return results

    def plot(self, agents: List["Agent"]) -> graphviz.Digraph:
        """Returns a Digraph object for broadcast flow."""
        dot = graphviz.Digraph(comment="Broadcast Agent Workflow")
        dot.attr(label="Broadcast Workflow", rankdir="TB")  # Top to bottom layout

        if not agents:
            dot.node("empty", "No agents")
            return dot

        # Create a central "broadcast" node *specific to this workflow plot*
        broadcast_node_id = "broadcast_source"
        dot.node(broadcast_node_id, "Broadcast\nInput", shape="box", style="filled", fillcolor="lightblue")

        # Connect the broadcast node to all agents
        for agent in agents:
            agent_label = getattr(agent, "name", str(id(agent)))
            node_id = f"agent_{agent_label}"  # Use agent name for node ID
            dot.node(node_id, agent_label)
            dot.edge(broadcast_node_id, node_id)

        return dot
