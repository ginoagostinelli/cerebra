import graphviz
from typing import Dict, Any, List, Set, Tuple, Union
from collections import defaultdict
from graphlib import TopologicalSorter
from cerebra.core.group import Group
from cerebra.core.workflow import SequentialWorkflow, ParallelWorkflow
import traceback

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
            raise TypeError(f"Node content must be an instance of Group. Received: {type(content)}")

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
        workflow_type = self.content.workflow.__class__.__name__
        return f"Node(id='{self.id}', group='{content_name}', workflow='{workflow_type}')"


class Graph:
    """Manages and executes a graph of connected Nodes."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)
        self._has_start_node_edges = False  # Track if the user explicitly added edges from START_NODE_ID

    def add_node(self, item: Union[Node, Group]):
        """Adds an item to the graph."""
        if isinstance(item, Group):
            node_id = item.name
            if not node_id:
                raise ValueError("Group provided to add_node must have a non-empty name.")
            if node_id == START_NODE_ID:
                raise ValueError(f"Group name cannot be the reserved ID '{START_NODE_ID}'.")
            if node_id in self.nodes:
                # Optional: Allow updating existing node content? For now, raise error.
                raise ValueError(f"Node or Group with name/ID '{node_id}' already exists.")
            node = Node(id=node_id, content=item)
        elif isinstance(item, Node):
            node = item
            if node.id == START_NODE_ID:
                raise ValueError(f"Node ID cannot be the reserved ID '{START_NODE_ID}'.")
            if node.id in self.nodes:
                raise ValueError(f"Node with ID '{node.id}' already exists.")
        else:
            raise TypeError("Can only add objects of type Node or Group to the graph.")

        self.nodes[node.id] = node
        self.edges.setdefault(node.id, set())
        self._predecessors.setdefault(node.id, set())

    def add_edge(self, from_item: Union[str, Group, Node], to_item: Union[str, Group, Node]):
        """
        Adds a directed edge connecting two items (Nodes, Groups, or string IDs).

        Args:
            from_item: The source item (Node, Group, or string ID, or START_NODE_ID).
            to_item: The target item (Node, Group, or string ID).
        """
        if from_item == START_NODE_ID:
            from_node_id = START_NODE_ID
            self._has_start_node_edges = True
        elif isinstance(from_item, (Group, Node)):
            self.add_node(from_item)
            from_node_id = from_item.name
        elif isinstance(from_item, str):
            from_node_id = from_item
            if from_node_id not in self.nodes and from_node_id != START_NODE_ID:
                raise ValueError(f"Source node ID '{from_node_id}' not found in graph.")
        else:
            raise TypeError("Edge source must be a string ID, Group, Node, or START_NODE_ID.")

        # --- Determine the 'to' node ID ---
        if to_item == START_NODE_ID:
            raise ValueError(f"Cannot add edge pointing *to* the reserved start node '{START_NODE_ID}'.")
        elif isinstance(to_item, (Group, Node)):
            self.add_node(to_item)
            to_node_id = to_item.name
        elif isinstance(to_item, str):
            to_node_id = to_item
            if to_node_id not in self.nodes:
                raise ValueError(f"Target node ID '{to_node_id}' not found in graph.")
        else:
            raise TypeError("Edge target must be a string ID, Group, or Node.")

        # ---
        if from_node_id == to_node_id:
            raise ValueError(f"Cannot add self-loop edge from '{from_node_id}' to itself.")

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
        edges = defaultdict(set, {k: v.copy() for k, v in self.edges.items()})
        predecessors = defaultdict(set, {k: v.copy() for k, v in self._predecessors.items()})

        # Ensure all nodes added via add_node exist as keys
        for node_id in self.nodes:
            edges.setdefault(node_id, set())
            predecessors.setdefault(node_id, set())

        # Determine starting nodes based on predecessors
        start_node_targets = set()
        if self._has_start_node_edges:
            start_node_targets = edges.get(START_NODE_ID, set()).intersection(self.nodes.keys())  # Ensure targets exist
            if not start_node_targets and self.nodes and START_NODE_ID in edges:
                print(f"Warning: Edges from {START_NODE_ID} were indicated, but none connect to existing nodes in the graph.")
        else:
            for node_id in self.nodes:
                node_preds = predecessors.get(node_id, set())
                if not any(pred in self.nodes for pred in node_preds):  # A node is a starting node if it has no predecessors within self.nodes
                    start_node_targets.add(node_id)

            if not start_node_targets and self.nodes:
                print("Warning: Could not automatically detect any starting nodes (no nodes without predecessors). Graph might be cyclic or empty.")
            elif start_node_targets:
                # Add implicit edges from START_NODE_ID
                edges[START_NODE_ID].update(start_node_targets)
                for target_node in start_node_targets:
                    predecessors[target_node].add(START_NODE_ID)

        # Ensure START_NODE_ID exists in the maps if it has connections
        if edges.get(START_NODE_ID):
            edges.setdefault(START_NODE_ID, set())
            predecessors.setdefault(START_NODE_ID, set())

        return dict(edges), dict(predecessors)

    def get_execution_order(self) -> List[str]:
        """Determines the execution order of nodes using topological sort."""
        edges, predecessors = self._get_graph_definition()

        graph_for_sort = defaultdict(set)
        all_nodes_in_graph = set(predecessors.keys()) | set(edges.keys())
        for source, targets in edges.items():
            all_nodes_in_graph.add(source)
            all_nodes_in_graph.update(targets)

        for node_id in all_nodes_in_graph:
            graph_for_sort[node_id] = predecessors.get(node_id, set())

        try:
            ts = TopologicalSorter(graph_for_sort)
            static_order = list(ts.static_order())
            return static_order
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

        for node_id in execution_order:
            if node_id == START_NODE_ID:
                continue

            if node_id not in self.nodes:
                print(f"Warning: Node ID '{node_id}' found in execution order but not in graph nodes. Skipping.")
                continue

            node = self.nodes[node_id]
            group_name = node.content.name

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
                print(f"ERROR executing node '{node_id}' (Group: {group_name}): {e}")
                print(traceback.format_exc())
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

        effective_edges, effective_predecessors = self._get_graph_definition()
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
                label = f"<<B>{group.name}</B><BR/>" f'<FONT POINT-SIZE="10">{workflow_type} ({num_agents} agents)</FONT>>'
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
            dot = graphviz.Digraph(comment="Agent Execution Graph")
            dot.attr(rankdir="LR", splines="ortho", nodesep="0.5", ranksep="0.8", compound="true")

            agents_info: Dict[str, Dict[str, str]] = {}  # global_agent_id -> {label, group_id}

            # group_node_id -> {workflow, agent_ids, first_agent, last_agent, parallel_input_node, parallel_output_node}
            groups_plot_info: Dict[str, Dict[str, Any]] = {}

            if has_start_node:
                dot.node(START_NODE_ID, label=start_node_label, **start_node_attrs)

            # Create nodes for all agents within each group's subgraph
            for node_id, node in self.nodes.items():
                group = node.content
                if isinstance(group, Group) and group.agents:
                    # Create a subgraph for the group to visually cluster agents
                    with dot.subgraph(name=f"cluster_{node_id}") as sub:
                        sub.attr(
                            label=f"{group.name}\n({group.workflow.__class__.__name__})",
                            style="rounded,filled",
                            color="lightgrey",
                            fillcolor="whitesmoke",
                        )
                        sub.attr(rank="same")  # Try to keep agents horizontally aligned if possible

                        group_agents_list = []  # List of global agent IDs in this group
                        first_agent_id, last_agent_id = None, None
                        parallel_input_node_id, parallel_output_node_id = None, None

                        # Create special points for parallel input/output visualization *inside* subgraph
                        if isinstance(group.workflow, ParallelWorkflow):
                            parallel_input_node_id = f"{node_id}_parallel_entry"
                            sub.node(
                                name=parallel_input_node_id,
                                label="",
                                shape="circle",
                                width="0.15",
                                height="0.15",
                                style="filled",
                                fillcolor="dodgerblue",
                            )
                            parallel_output_node_id = f"{node_id}_parallel_collector"
                            sub.node(
                                name=parallel_output_node_id,
                                label="",
                                shape="circle",
                                width="0.15",
                                height="0.15",
                                style="filled",
                                fillcolor="tomato",
                            )

                        # Create nodes for each agent *inside* subgraph
                        for i, agent in enumerate(group.agents):
                            global_id = f"{node_id}_{agent.name}"
                            # Handle potential duplicate agent names within the *same* group if necessary
                            count = 1
                            original_global_id = global_id
                            while global_id in agents_info:
                                count += 1
                                global_id = f"{original_global_id}_{count}"
                                print(f"Warning (Plot): Adjusting duplicate agent plot ID to {global_id}")

                            agents_info[global_id] = {
                                "label": agent.name + (f"_{count}" if count > 1 else ""),  # Adjust label if ID changed
                                "group_id": node_id,
                            }
                            # Add agent node to the subgraph
                            sub.node(global_id, label=agents_info[global_id]["label"], shape="ellipse", style="filled", fillcolor="white")
                            group_agents_list.append(global_id)
                            if i == 0:
                                first_agent_id = global_id
                            last_agent_id = global_id

                        groups_plot_info[node_id] = {
                            "workflow": group.workflow,
                            "agent_ids": group_agents_list,
                            "first_agent": first_agent_id,
                            "last_agent": last_agent_id,
                            "parallel_input_node": parallel_input_node_id,
                            "parallel_output_node": parallel_output_node_id,
                        }

            # Create workflow-specific edges *within* groups (connecting agents/points)
            for node_id, info in groups_plot_info.items():
                workflow = info["workflow"]
                agent_ids = info["agent_ids"]

                if isinstance(workflow, SequentialWorkflow) and len(agent_ids) > 1:
                    for i in range(len(agent_ids) - 1):
                        dot.edge(agent_ids[i], agent_ids[i + 1], color="black")

                elif isinstance(workflow, ParallelWorkflow) and agent_ids:
                    input_node = info["parallel_input_node"]
                    output_node = info["parallel_output_node"]
                    if input_node and output_node:
                        for agent_id in agent_ids:
                            dot.edge(input_node, agent_id, color="dodgerblue", arrowhead="none")
                        for agent_id in agent_ids:
                            dot.edge(agent_id, output_node, style="dashed", color="tomato", arrowhead="none")  # Connector

            # Connect groups/nodes based on the main graph edges (inter-group connections)
            for from_node_id, successors in effective_edges.items():
                connect_from_id = None
                is_from_start_node = from_node_id == START_NODE_ID

                if is_from_start_node:
                    if has_start_node:
                        connect_from_id = START_NODE_ID
                    else:
                        continue  # Skip if START isn't plotted
                elif from_node_id in groups_plot_info:
                    from_info = groups_plot_info[from_node_id]
                    if isinstance(from_info["workflow"], ParallelWorkflow):
                        connect_from_id = from_info.get("parallel_output_node")
                    else:  # Default (Sequential, or others)
                        connect_from_id = from_info.get("last_agent")
                # Else: Source is not START and not a plottable group

                if not connect_from_id:
                    continue

                # Connect to each successor
                for to_node_id in successors:
                    if to_node_id == START_NODE_ID:
                        continue
                    if to_node_id not in groups_plot_info:
                        continue

                    to_info = groups_plot_info[to_node_id]
                    connect_to_id = None

                    # Determine the connection point for the target node based on its workflow
                    target_workflow = to_info["workflow"]
                    if isinstance(target_workflow, ParallelWorkflow):
                        connect_to_id = to_info.get("parallel_input_node")
                    else:  # Default (Sequential, or others)
                        connect_to_id = to_info.get("first_agent")

                    if not connect_to_id:
                        continue

                    # Add the edge
                    edge_attrs = {}

                    lhead = f"cluster_{to_node_id}" if to_node_id in groups_plot_info else None
                    ltail = f"cluster_{from_node_id}" if from_node_id in groups_plot_info else None

                    if is_from_start_node:
                        edge_attrs = {"color": "darkgreen", "penwidth": "2.0", "style": "bold"}
                        if lhead:
                            edge_attrs["lhead"] = lhead
                    else:
                        # Style for edges between groups/nodes
                        edge_attrs = {
                            "style": "dashed",
                            "color": "darkblue",
                            "penwidth": "1.5",
                            "constraint": "true",
                        }
                        if ltail:
                            edge_attrs["ltail"] = ltail
                        if lhead:
                            edge_attrs["lhead"] = lhead

                    # Remove None values from edge_attrs
                    edge_attrs = {k: v for k, v in edge_attrs.items() if v is not None}

                    dot.edge(connect_from_id, connect_to_id, **edge_attrs)

            return dot
