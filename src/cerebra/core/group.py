from typing import List, Optional, Any, Dict, Union
from cerebra.core import Agent
from cerebra.core.workflow import Workflow, SequentialWorkflow


class Group:
    def __init__(self, name: str, agents: List[Agent] = [], workflow: Optional[Workflow] = None) -> None:
        """
        Initializes a group of agents with a specific workflow.

        Args:
            agents: A list of Agent instances.
            workflow: An instance of a Workflow subclass. Defaults to SequentialWorkflow if not provided.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("Group name must be a non-empty string.")
        self.name = name
        self.agents: List[Agent] = []
        self.agent_map: Dict[str, Agent] = {}
        self.add_agents(agents)

        if workflow is None:
            self.workflow: Workflow = SequentialWorkflow()
        elif isinstance(workflow, Workflow):
            self.workflow: Workflow = workflow
        else:
            raise TypeError("workflow must be an instance of a Workflow subclass")

    def add_agents(self, agents_to_add: Union[Agent, List[Agent]]):
        """Adds one or more agents to the group and updates the agent_map."""
        if not isinstance(agents_to_add, list):
            agents_to_add = [agents_to_add]

        for agent in agents_to_add:
            if isinstance(agent, Agent):
                if not hasattr(self, "agent_map"):
                    self.agent_map = {}  # Initialize if somehow missing

                if agent.name in self.agent_map:
                    print(
                        f"Warning: Agent with name '{agent.name}' already exists. Replacing the entry in the map, but the list might contain duplicates if added multiple times."
                    )
                    # Find and remove existing agent from list
                    self.agents = [a for a in self.agents if a.name != agent.name]

                self.agents.append(agent)
                self.agent_map[agent.name] = agent
            else:
                raise TypeError(f"Object {agent} is not an instance of Agent")

    def get_agent(self, name: str) -> Optional[Agent]:
        """Retrieves an agent by name from the map."""
        return self.agent_map.get(name)

    def plot(self):
        """Generates and returns a graphviz object representing the workflow."""
        return self.workflow.plot(self.agents)

    def view(self):
        pass

    def run(self, inputs: Any, context: str = None, max_iterations: int = 10) -> Any:
        """
        Runs the group using the configured workflow.

        Args:
            inputs: The starting data or task description for the workflow.
            max_iterations: Max iterations for workflows that loop.

        Returns:
            The final result produced by the workflow.
        """
        if not self.agents:
            print("Warning: Running group with no agents.")
            return None

        try:
            output = self.workflow.run(
                agents=self.agents,
                inputs=inputs,
                context=context,
                max_iterations=max_iterations,
            )
            return output
        except Exception as e:
            print(f"ERROR during execution of group '{self.name}': {e}")
            raise RuntimeError(f"Group '{self.name}' failed during execution.") from e
