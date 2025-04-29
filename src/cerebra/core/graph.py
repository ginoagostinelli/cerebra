import graphviz
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict
from graphlib import TopologicalSorter
from cerebra.core.group import Group
from cerebra.core.workflow import SequentialWorkflow, BroadcastWorkflow

START_NODE_ID = "__START__"


class Node:
    """Represents a node in the execution graph, typically containing a Group."""

    def __init__(self, id: str, content: Group):
        """
        Initializes a Node.

        Args:
            id: A unique identifier for this node within the graph.
            content: The functional unit this node represents (e.g., a Group instance).
                     Currently designed for Group, but could be extended.
        """
        if not isinstance(id, str) or not id:
            raise ValueError("Node ID must be a non-empty string.")

        if not isinstance(content, Group):
            raise TypeError("Content must be a Group. (Only Group support for now)")

        # Basic check, could add more specific type checks later
        # if not hasattr(content, "run") or not callable(content.run):
        #     raise TypeError(f"Node content {content} must have a callable 'run' method.")
        # if not hasattr(content, "name"):
        #     print(f"Warning: Node content {content} lacks a 'name' attribute for clearer representation.")

        self.id: str = id
        self.content: Group = content

    def run(self, inputs: Any = None, context: Any = None, max_iterations: int = 10) -> Any:
        """
        Executes the content of the node.

        Args:
            inputs: The data passed to this node for execution.
            max_iterations: Max iterations limit for the content's run method.

        Returns:
            The result from executing the node's content.
        """
        return self.content.run(inputs=inputs, context=context, max_iterations=max_iterations)

    def __repr__(self) -> str:
        content_name = getattr(self.content, "name", str(self.content))
        return f"Node(id='{self.id}', content='{content_name}')"


class Graph:
    """Manages and executes a graph of connected Nodes."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)
        self._has_start_node_edges = False  # Track if the user explicitly added edges from START_NODE_ID

    def add_node(self, node: Node):
        """Adds a Node to the graph."""
        if not isinstance(node, Node):
            raise TypeError("Can only add objects of type Node to the graph.")
        if node.id == START_NODE_ID:
            raise ValueError(f"Node ID cannot be the reserved ID '{START_NODE_ID}'.")
        if node.id in self.nodes:
            raise ValueError(f"Node with ID '{node.id}' already exists in the graph.")
        self.nodes[node.id] = node
        self.edges[node.id] = self.edges.get(node.id, set())
        self._predecessors[node.id] = self._predecessors.get(node.id, set())

    def add_edge(self, from_node_id: str, to_node_id: str):
        """
        Adds a directed edge connecting two nodes or from the START node.

        Args:
            from_node_id: The ID of the node where the edge originates, or START_NODE_ID.
            to_node_id: The ID of the node where the edge terminates.
        """
        if to_node_id == START_NODE_ID:
            raise ValueError(f"Cannot add edge pointing *to* the reserved start node '{START_NODE_ID}'.")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node '{to_node_id}' not found in graph.")

        if from_node_id == START_NODE_ID:
            self._has_start_node_edges = True

            # TODO: Maybe store these special edges separately for clarity
            self.edges[from_node_id].add(to_node_id)
            self._predecessors[to_node_id].add(from_node_id)

        elif from_node_id not in self.nodes:
            raise ValueError(f"Source node '{from_node_id}' not found in graph.")
        elif from_node_id == to_node_id:
            raise ValueError("Cannot add self-loop edge.")

        self.edges[from_node_id].add(to_node_id)
        self._predecessors[to_node_id].add(from_node_id)

    def _get_graph_definition(self) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
        """
        Determines the actual graph structure including the START node (explicit or implicit).

        Returns:
            A tuple containing:
            - edges: The complete edge map (including START node connections).
            - predecessors: The complete predecessor map.
        """
        edges = defaultdict(set)
        predecessors = defaultdict(set)

        for u, targets in self.edges.items():
            edges[u].update(targets)
        for v, sources in self._predecessors.items():
            predecessors[v].update(sources)

        # Ensure all nodes exist as keys even if they have no outgoing edges/predecessors
        for node_id in self.nodes:
            edges.setdefault(node_id, set())
            predecessors.setdefault(node_id, set())

        # Determine starting nodes
        start_node_targets = set()
        if self._has_start_node_edges:
            start_node_targets = edges.get(START_NODE_ID, set())
            if not start_node_targets:
                print(f"Warning: Start node was marked explicit, but no edges from {START_NODE_ID} found.")
        else:
            for node_id in self.nodes:
                preds = predecessors.get(node_id, set())
                if not preds or preds == {START_NODE_ID}:  # Handles nodes only connected from START explicitly
                    start_node_targets.add(node_id)

            if not start_node_targets and self.nodes:
                print("Warning: Could not automatically detect any starting nodes (no nodes without predecessors). Graph might be cyclic or empty.")
            elif start_node_targets:
                # Add implicit edges from START_NODE_ID
                edges[START_NODE_ID].update(start_node_targets)
                for target_node in start_node_targets:
                    predecessors[target_node].add(START_NODE_ID)

        # Ensure START_NODE_ID exists in the maps if it has connections
        if start_node_targets:
            edges.setdefault(START_NODE_ID, set()).update(start_node_targets)
            predecessors.setdefault(START_NODE_ID, set())

        return dict(edges), dict(predecessors)

    def get_execution_order(self) -> List[str]:
        """Determines the execution order of nodes using topological sort."""
        edges, predecessors = self._get_graph_definition()
        try:
            ts = TopologicalSorter(predecessors)
            return list(ts.static_order())
        except Exception as e:
            raise ValueError(f"Could not determine execution order. Graph might contain cycles or be invalid: {e}") from e

    def run(self, inputs: Any = None, max_iterations_per_node: int = 10) -> Dict[str, Any]:
        """
        Executes the graph workflow.

        Args:
            inputs: The single input data provided to the start of the graph.
            max_iterations_per_node: Max iteration limit passed to each node's run method.

        Returns:
            A dictionary mapping actual node IDs (excluding START_NODE_ID) to their outputs.
        """
        try:
            execution_order = self.get_execution_order()
        except ValueError as e:
            print(f"Halting execution due to error determining order: {e}")
            return {}

        _, node_predecessors = self._get_graph_definition()
        node_outputs: Dict[str, Any] = {}

        if not self.nodes:
            print("Graph has no nodes to execute.")
            return {}
        if START_NODE_ID not in execution_order and any(START_NODE_ID in preds for preds in node_predecessors.values()):
            print(
                f"Warning: Graph seems to have starting nodes connected from {START_NODE_ID}, but it wasn't included in execution order. Check for cycles."
            )

        for node_id in execution_order:
            if node_id == START_NODE_ID:
                continue

            if node_id not in self.nodes:
                print(f"Warning: Node ID '{node_id}' found in execution order but not in graph nodes. Skipping.")
                continue

            node = self.nodes[node_id]
            # print(f"\nProcessing Node: {node.id} ({getattr(node.content, 'name', 'N/A')})")

            predecessors = node_predecessors.get(node_id, set())
            node_context: Any = None

            if START_NODE_ID in predecessors:
                if len(predecessors) > 1:
                    other_preds = predecessors - {START_NODE_ID}
                    print(f"Info: Node '{node_id}' connected to {START_NODE_ID} and other nodes ({other_preds}). Using initial graph input.")
                    # Decide if merging logic is needed here. For now, prioritize START input.
            elif not predecessors:
                # Should ideally not happen if _get_graph_definition is correct, unless the graph is disconnected and this node wasn't a start node.
                print(f"Warning: Node '{node_id}' has no predecessors in graph. Receiving None input.")
                node_context = None
            else:
                # Gather inputs from regular predecessors
                pred_outputs = {pred_id: node_outputs.get(pred_id) for pred_id in predecessors if pred_id != START_NODE_ID}
                if len(pred_outputs) == 1:
                    node_context = next(iter(pred_outputs.values()))
                else:
                    node_context = pred_outputs

            try:
                output = node.run(
                    inputs=inputs,
                    context=node_context,
                    max_iterations=max_iterations_per_node,
                )
                node_outputs[node_id] = output
            except Exception as e:
                print(f"ERROR running node '{node_id}': {e}")
                node_outputs[node_id] = f"ERROR: {e}"
                # Decide on error handling: continue or raise?
                # raise RuntimeError(f"Execution failed at node '{node_id}'") from e
        return node_outputs

    def plot(self, detailed: bool = False) -> graphviz.Digraph:
        """
        Generates a visual representation of the graph.

        Args:
            detailed: If False (default), shows groups as nodes.
                      If True, shows all individual agents.

        Returns:
            A graphviz.Digraph object visualization.
        """
        start_node_label = "START"
        start_node_attrs = {
            "shape": "Mdiamond",
            "style": "filled",
            "color": "mediumseagreen",
            "fontcolor": "white",
        }

        effective_edges, _ = self._get_graph_definition()
        has_start_node = START_NODE_ID in effective_edges or any(START_NODE_ID in preds for preds in effective_predecessors.values())

        if not detailed:
            dot = graphviz.Digraph(comment="Group Execution Graph")
            dot.attr(rankdir="LR", splines="ortho", nodesep="0.8", ranksep="1")

            if has_start_node:
                dot.node(START_NODE_ID, label=start_node_label, **start_node_attrs)

            # Add regular nodes
            for node_id, node in self.nodes.items():
                group = node.content
                workflow_type = group.workflow.__class__.__name__
                num_agents = len(group.agents)
                # label = f"<{node_id}<BR/>" f"<B>{group.name}</B><BR/>" f'<FONT POINT-SIZE="10">{workflow_type} ({num_agents} agents)</FONT>>'
                label = f"<{node_id}<BR/>" f"<B>{group.name}</B><BR/>" f'<FONT POINT-SIZE="10">{workflow_type} ({num_agents} agents)</FONT>>'
                dot.node(node_id, label=label, shape="box", style="rounded,filled", fillcolor="lightblue")  # Box shape for groups

            # Add edges from the graph definition
            for from_id, successors in effective_edges.items():
                for to_id in successors:
                    # Don't draw edges *to* the START node
                    if to_id == START_NODE_ID:
                        continue
                    # Don't draw edges from a node not in the plot (e.g., START if not connected)
                    if from_id == START_NODE_ID and not has_start_node:
                        continue
                    # Ensure target node exists in the plot
                    if to_id not in self.nodes and to_id != START_NODE_ID:  # Check against actual nodes
                        print(f"Warning (Plot): Target node '{to_id}' for edge from '{from_id}' not found in graph nodes. Skipping edge.")
                        continue

                    # Set edge color
                    edge_attrs = {}
                    if from_id == START_NODE_ID:
                        edge_attrs = {"color": "darkgreen", "penwidth": "2.0", "style": "bold"}
                    else:
                        edge_attrs = {"color": "black", "penwidth": "1.0"}
                    dot.edge(from_id, to_id, **edge_attrs)

            return dot

        else:
            dot = graphviz.Digraph(comment="Detailed Agent Graph")
            dot.attr(rankdir="LR")

            agents_info: Dict[str, Dict[str, str]] = {}
            groups_plot_info: Dict[str, Dict[str, Any]] = {}

            if has_start_node:
                dot.node(START_NODE_ID, label=start_node_label, **start_node_attrs)

            # Create nodes for all agents
            for node_id, node in self.nodes.items():
                content = node.content
                if isinstance(content, Group) and content.agents:
                    group_agents_list = []
                    first_agent_id, last_agent_id, broadcast_input_node_id = (
                        None,
                        None,
                        None,
                    )
                    if isinstance(content.workflow, BroadcastWorkflow):
                        broadcast_input_node_id = f"{node_id}_broadcast_entry"
                        dot.node(
                            broadcast_input_node_id,
                            "",
                            shape="point",
                            width="0.1",
                            height="0.1",
                        )
                    for i, agent in enumerate(content.agents):
                        global_id = f"{node_id}_{agent.name}"
                        if global_id in agents_info:
                            print(f"Warning: Duplicate global agent ID: {global_id}")
                        agents_info[global_id] = {
                            "label": agent.name,
                            "group_id": node_id,
                        }
                        dot.node(global_id, label=agent.name)
                        group_agents_list.append(global_id)
                        if i == 0:
                            first_agent_id = global_id
                        last_agent_id = global_id
                    groups_plot_info[node_id] = {
                        "workflow": content.workflow,
                        "agent_ids": group_agents_list,
                        "first_agent": first_agent_id,
                        "last_agent": last_agent_id,
                        "broadcast_input_node": broadcast_input_node_id,
                    }

            # Create workflow-specific edges within groups
            for node_id, info in groups_plot_info.items():
                workflow = info["workflow"]
                agent_ids = info["agent_ids"]
                if isinstance(workflow, SequentialWorkflow) and len(agent_ids) > 1:
                    for i in range(len(agent_ids) - 1):
                        dot.edge(agent_ids[i], agent_ids[i + 1])
                elif isinstance(workflow, BroadcastWorkflow) and agent_ids:
                    input_node = info["broadcast_input_node"]
                    if input_node:
                        for agent_id in agent_ids:
                            dot.edge(input_node, agent_id)

            # Connect groups with edges
            for from_node_id, successors in effective_edges.items():
                connect_from_id = None
                is_from_start_node = from_node_id == START_NODE_ID

                if is_from_start_node:
                    connect_from_id = START_NODE_ID
                elif from_node_id in groups_plot_info:
                    connect_from_id = groups_plot_info[from_node_id].get("last_agent")
                # Else: Source is not START and not a plottable group

                if not connect_from_id:
                    continue

                for to_node_id in successors:
                    if to_node_id == START_NODE_ID:
                        continue
                    if to_node_id not in groups_plot_info:
                        continue

                    to_info = groups_plot_info[to_node_id]
                    connect_to_id = None
                    target_workflow = to_info["workflow"]
                    if isinstance(target_workflow, BroadcastWorkflow):
                        connect_to_id = to_info.get("broadcast_input_node")
                    else:
                        connect_to_id = to_info.get("first_agent")

                    if not connect_to_id:
                        continue

                    # Add the edge with appropriate style
                    edge_attrs = {}
                    if is_from_start_node:
                        edge_attrs = {"color": "darkgreen", "penwidth": "1.5"}
                    else:
                        edge_attrs = {
                            "style": "dashed",
                            "color": "blue",
                            "constraint": "false",
                        }
                    dot.edge(connect_from_id, connect_to_id, **edge_attrs)

            return dot
