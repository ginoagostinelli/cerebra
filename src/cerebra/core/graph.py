import graphviz
from typing import Dict, Any, List, Set, Tuple, Union
from collections import defaultdict
from graphlib import TopologicalSorter
from cerebra.core.agent import Agent
from cerebra.core.group import Group
from cerebra.core.workflow import SequentialWorkflow, ParallelWorkflow
import traceback

START_NODE_ID = "__START__"


class Node:
    """Represents a node in the execution graph, typically containing a Group."""

    def __init__(self, id: str, content: Union[Agent, Group]):
        """
        Initializes a Node.

        Args:
            id: A unique identifier for this node within the graph.
            content: The functional unit this node represents (e.g., a Group instance).
                     Currently designed for Group, but could be extended.
        """
        if not isinstance(id, str) or not id:
            raise ValueError("Node ID must be a non-empty string.")

        if not isinstance(content, (Agent, Group)):
            raise TypeError(f"Node content must be an Agent or a Group. Received: {type(content)}")

        # The name is mandatory for now
        if not hasattr(content, "name") or not content.name:
            raise TypeError(f"Node content {content} must have a non-empty 'name' attribute.")

        if id != content.name:
            raise ValueError("Node ID must match content name.")

        self.id: str = id
        self.content: Union[Agent, Group] = content

    @property
    def name(self) -> str:
        """Returns the name of the content (Agent or Group name)."""
        return self.content.name

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
        content_type = self.content.__class__.__name__
        return f"Node(id='{self.id}', type='{content_type}', name='{self.name}')"


class Graph:
    """Manages and executes a graph of connected Nodes."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)
        self._has_start_node_edges = False  # Track if the user explicitly added edges from START_NODE_ID
        self.start_node = START_NODE_ID

    def add_edge(
        self,
        from_items: Union[str, Agent, Group, Node, List[Union[str, Agent, Group, Node]]],
        to_items: Union[str, Agent, Group, Node, List[Union[str, Agent, Group, Node]]],
    ) -> None:
        """
        Adds one or many directed edges between sources and targets.

        Args:
            from_items: A single source item or list of source items.
            to_items: A single target item or list of target items.
        """
        sources = from_items if isinstance(from_items, (list, tuple)) else [from_items]
        targets = to_items if isinstance(to_items, (list, tuple)) else [to_items]

        for src in sources:
            for tgt in targets:
                self._add_single_edge(src, tgt)

    def _add_single_edge(self, from_item: Union[str, Agent, Group, Node], to_item: Union[str, Agent, Group, Node]):
        """
        Adds a single directed edge connecting two items (Nodes, Agents, Groups, or string IDs).

        Args:
            from_item: The source item or its ID (or START_NODE_ID).
            to_item: The target item or its ID.
        """
        try:
            from_node_id = self._resolve_item_to_id(from_item)
        except (ValueError, TypeError) as e:
            raise type(e)(f"Source node resolution failed: {e}") from e

        if to_item == START_NODE_ID:
            raise ValueError(f"Cannot connect *to* the reserved start node '{START_NODE_ID}'.")

        try:
            to_node_id = self._resolve_item_to_id(to_item)
        except (ValueError, TypeError) as e:
            raise type(e)(f"Target node resolution failed: {e}") from e

        if from_node_id == to_node_id:
            raise ValueError(f"Cannot add self-loop connection from '{from_node_id}' to itself.")

        # Add the edge if it doesn't exist already
        if to_node_id not in self.edges[from_node_id]:
            self.edges[from_node_id].add(to_node_id)
            self._predecessors[to_node_id].add(from_node_id)

        if from_node_id == START_NODE_ID:
            self._has_start_node_edges = True

    def _resolve_item_to_id(self, item: Union[str, Agent, Group, Node]) -> str:
        """Resolves an item (Agent, Group, Node, str) to its node ID string."""
        if item == START_NODE_ID:
            return START_NODE_ID

        elif isinstance(item, Node):
            node_id = item.id
            if node_id not in self.nodes:
                print(f"Warning: Node object {item} passed but not found in graph nodes. Adding it.")
                self._add_node(item)
            return node_id

        elif isinstance(item, (Agent, Group)):
            node_id = item.name
            if not node_id:
                raise ValueError(f"{type(item).__name__} must have a non-empty name.")
            if node_id not in self.nodes:
                self._add_node(item)
            return node_id

        elif isinstance(item, str):
            if item != START_NODE_ID and item not in self.nodes:
                raise ValueError(f"Node ID '{item}' not found in graph.")
            return item

        else:
            raise TypeError(f"Invalid item type for connection: {type(item)}. Must be str, Agent, Group, or Node.")

    def _add_node(self, item: Union[Node, Agent, Group]):
        """Adds an Agent, Group, or Node object to the graph."""

        if isinstance(item, Node):
            node = item
            node_id = node.id
            if node.id == START_NODE_ID:
                raise ValueError(f"Node ID cannot be the reserved ID '{START_NODE_ID}'.")
            if node.id in self.nodes:
                raise ValueError(f"Node with ID '{node.id}' already exists.")

        elif isinstance(item, (Agent, Group)):
            node_id = item.name
            if not node_id:
                raise ValueError(f"{type(item).__name__} must have a non-empty name to be added as a node.")
            try:
                node = Node(id=node_id, content=item)
            except ValueError as e:
                raise ValueError(f"Failed to create Node for {item}: {e}") from e

        else:
            raise TypeError("Can only add objects of type Node or Group to the graph.")

        if node_id == START_NODE_ID:
            raise ValueError(f"Node ID cannot be the reserved ID '{START_NODE_ID}'.")

        if node_id in self.nodes:
            if self.nodes[node_id].content is node.content:
                return  # Already exists, do nothing

        self.nodes[node.id] = node
        self.edges.setdefault(node.id, set())
        self._predecessors.setdefault(node.id, set())

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

    # TODO: refactor (split function)
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
        start_node_attrs = {"shape": "Mdiamond", "style": "filled", "color": "mediumseagreen", "fontcolor": "white"}
        agent_node_attrs = {"shape": "ellipse", "style": "filled", "fillcolor": "lightyellow"}
        group_node_attrs = {"shape": "box", "style": "rounded,filled", "fillcolor": "lightblue"}

        effective_edges, effective_predecessors = self._get_graph_definition()
        has_start_node = START_NODE_ID in effective_edges or any(START_NODE_ID in preds for preds in effective_predecessors.values())

        if not detailed:
            dot = graphviz.Digraph(comment="Execution Graph")
            dot.attr(rankdir="LR", splines="ortho", nodesep="0.6", ranksep="0.8")

            if has_start_node:
                dot.node(START_NODE_ID, label=start_node_label, **start_node_attrs)

            for node_id, node in self.nodes.items():
                if isinstance(node.content, Agent):
                    label = f"<<B>{node.name}</B><BR/><FONT POINT-SIZE='10'>Agent</FONT>>"
                    dot.node(node_id, label=label, **agent_node_attrs)

                elif isinstance(node.content, Group):
                    group = node.content

                    workflow_type = getattr(group, "workflow", "Unknown Workflow")
                    if not isinstance(workflow_type, str):  # Handle case where it's an object
                        workflow_type = workflow_type.__class__.__name__
                    num_agents = len(getattr(group, "agents", []))
                    label = f"<<B>{group.name}</B><BR/><FONT POINT-SIZE='10'>Group ({workflow_type}, {num_agents} agents)</FONT>>"
                    dot.node(node_id, label=label, **group_node_attrs)

                else:  # Should not happen with current Node validation
                    dot.node(node_id, label=f"{node_id}\n(Unknown Type)", shape="egg")

            # Add edges from the graph definition
            for from_id, successors in effective_edges.items():
                for to_id in successors:
                    # Ensure target node exists in the plot (or is START)
                    if to_id not in self.nodes and to_id != START_NODE_ID:
                        print(f"Warning (Plot): Target node '{to_id}' for edge from '{from_id}' not found in graph nodes. Skipping edge.")
                        continue

                    # Ensure source node exists in the plot's nodes (or is START)
                    if from_id not in self.nodes and from_id != START_NODE_ID:
                        print(f"Warning (Plot): Source node '{from_id}' for edge to '{to_id}' not found in graph nodes. Skipping edge.")
                        continue

                    edge_attrs = {}
                    if from_id == START_NODE_ID:
                        edge_attrs = {"color": "darkgreen", "penwidth": "2.0", "style": "bold"}
                    else:
                        edge_attrs = {"color": "black", "penwidth": "1.0"}
                    dot.edge(from_id, to_id, **edge_attrs)

            return dot

        else:
            dot = graphviz.Digraph(comment="Execution Graph")
            dot.attr(rankdir="LR", splines="ortho", nodesep="0.5", ranksep="0.8", compound="true")

            # Map: global_agent_plot_id -> {label: str, group_cluster_id: Optional[str]}
            agents_plot_info: Dict[str, Dict[str, Any]] = {}
            # Map: group_node_id -> {workflow_obj: Any, agent_plot_ids: List[str], first_agent_plot_id: str, last_agent_plot_id: str, parallel_input_node: str, parallel_output_node: str}
            groups_plot_info: Dict[str, Dict[str, Any]] = {}
            # Map: standalone_agent_node_id -> plot_id (usually same)
            standalone_agents_plot_ids: Dict[str, str] = {}

            if has_start_node:
                dot.node(START_NODE_ID, label=start_node_label, **start_node_attrs)

            # --- Phase 1: Create nodes and subgraphs ---
            for node_id, node in self.nodes.items():
                content = node.content

                if isinstance(content, Group):
                    group = content
                    group_cluster_id = f"cluster_{node_id}"
                    workflow_obj = getattr(group, "workflow", None)  # Get actual workflow object if possible
                    workflow_type_name = workflow_obj.__class__.__name__ if workflow_obj else "UnknownWorkflow"

                    # Create a subgraph for the group
                    with dot.subgraph(name=group_cluster_id) as sub:
                        sub.attr(
                            label=f"{group.name}\n({workflow_type_name})",
                            style="rounded,filled",
                            color="lightgrey",
                            fillcolor="whitesmoke",
                            rank="same",  # Attempt to align agents
                        )

                        group_agent_plot_ids = []
                        first_agent_plot_id, last_agent_plot_id = None, None
                        parallel_input_node_id, parallel_output_node_id = None, None

                        is_parallel = workflow_type_name == "ParallelWorkflow"

                        if is_parallel:
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

                        # Create nodes for each agent *inside* the group's subgraph
                        group_agents = getattr(group, "agents", [])
                        for i, agent in enumerate(group_agents):
                            agent_name = getattr(agent, "name", f"agent_{i}")
                            # Create a unique plot ID for this agent *instance* within this graph plot
                            agent_plot_id = f"{node_id}_{agent_name}"  # Agents within group use group prefix
                            # Handle duplicate agent names within the same group for plotting
                            count = 1
                            original_agent_plot_id = agent_plot_id
                            while agent_plot_id in agents_plot_info:
                                count += 1
                                agent_plot_id = f"{original_agent_plot_id}_{count}"
                                print(f"Warning (Plot): Adjusting duplicate agent plot ID within group '{node_id}' to {agent_plot_id}")

                            agents_plot_info[agent_plot_id] = {
                                "label": agent_name + (f"_{count}" if count > 1 else ""),
                                "group_cluster_id": group_cluster_id,
                            }
                            # Add agent node to the subgraph
                            sub.node(agent_plot_id, label=agents_plot_info[agent_plot_id]["label"], **agent_node_attrs)
                            group_agent_plot_ids.append(agent_plot_id)
                            if i == 0:
                                first_agent_plot_id = agent_plot_id
                            last_agent_plot_id = agent_plot_id

                        groups_plot_info[node_id] = {
                            "workflow_obj": workflow_obj,
                            "workflow_type_name": workflow_type_name,
                            "agent_plot_ids": group_agent_plot_ids,
                            "first_agent_plot_id": first_agent_plot_id,
                            "last_agent_plot_id": last_agent_plot_id,
                            "parallel_input_node": parallel_input_node_id,
                            "parallel_output_node": parallel_output_node_id,
                        }

                elif isinstance(content, Agent):
                    agent = content
                    agent_plot_id = node_id
                    if agent_plot_id in agents_plot_info or agent_plot_id in standalone_agents_plot_ids:
                        # Handle potential ID clash if an agent has same name as a group
                        count = 1
                        original_agent_plot_id = agent_plot_id
                        while agent_plot_id in agents_plot_info or agent_plot_id in standalone_agents_plot_ids:
                            count += 1
                            agent_plot_id = f"{original_agent_plot_id}_agent_{count}"
                            print(f"Warning (Plot): Adjusting standalone agent plot ID to avoid conflict: {agent_plot_id}")

                    agents_plot_info[agent_plot_id] = {"label": agent.name, "group_cluster_id": None}
                    standalone_agents_plot_ids[node_id] = agent_plot_id
                    dot.node(agent_plot_id, label=agent.name, **agent_node_attrs)

                else:  # Should not happen
                    dot.node(node_id, label=f"{node_id}\n(Unknown Type)", shape="egg")

            # --- Phase 2: Create edges *within* groups based on workflow ---
            for node_id, info in groups_plot_info.items():
                agent_plot_ids = info["agent_plot_ids"]
                is_sequential = info["workflow_type_name"] == "SequentialWorkflow"
                is_parallel = info["workflow_type_name"] == "ParallelWorkflow"

                if is_sequential and len(agent_plot_ids) > 1:
                    for i in range(len(agent_plot_ids) - 1):
                        dot.edge(agent_plot_ids[i], agent_plot_ids[i + 1], color="black")
                elif is_parallel and agent_plot_ids:
                    input_node = info["parallel_input_node"]
                    output_node = info["parallel_output_node"]
                    if input_node and output_node:
                        # Connect entry point to each agent
                        for agent_plot_id in agent_plot_ids:
                            dot.edge(input_node, agent_plot_id, color="dodgerblue", arrowhead="odot", arrowtail="none", dir="forward")
                        # Connect each agent to collector point
                        for agent_plot_id in agent_plot_ids:
                            dot.edge(agent_plot_id, output_node, style="dashed", color="tomato", arrowhead="none")

            # --- Phase 3: Create edges *between* nodes/groups based on main graph edges ---
            for from_node_id, successors in effective_edges.items():
                connect_from_plot_id = None  # The actual source point for the edge in the plot

                is_from_start_node = from_node_id == START_NODE_ID
                if is_from_start_node:
                    if has_start_node:
                        connect_from_plot_id = START_NODE_ID
                    else:
                        continue  # START node isn't plotted

                elif from_node_id in groups_plot_info:  # Source is a Group
                    from_group_info = groups_plot_info[from_node_id]
                    is_parallel = from_group_info["workflow_type_name"] == "ParallelWorkflow"
                    # Connect from the group's exit point
                    connect_from_plot_id = from_group_info.get("parallel_output_node") if is_parallel else from_group_info.get("last_agent_plot_id")
                    if not connect_from_plot_id and from_group_info.get("agent_plot_ids"):  # Handle single-agent group
                        connect_from_plot_id = from_group_info["agent_plot_ids"][0]

                elif from_node_id in standalone_agents_plot_ids:  # Source is a Agent
                    connect_from_plot_id = standalone_agents_plot_ids[from_node_id]

                if not connect_from_plot_id:
                    if not is_from_start_node:
                        print(f"Warning (Plot): Could not find plot source point for node ID '{from_node_id}'. Skipping outgoing edges.")
                    continue

                # Connect to each successor
                for to_node_id in successors:
                    if to_node_id == START_NODE_ID:
                        continue

                    connect_to_plot_id = None
                    target_cluster_id = None

                    if to_node_id in groups_plot_info:  # Target is a Group
                        to_group_info = groups_plot_info[to_node_id]
                        is_parallel = to_group_info["workflow_type_name"] == "ParallelWorkflow"
                        target_cluster_id = f"cluster_{to_node_id}"
                        # Connect to the group's entry point
                        connect_to_plot_id = to_group_info.get("parallel_input_node") if is_parallel else to_group_info.get("first_agent_plot_id")
                        if not connect_to_plot_id and to_group_info.get("agent_plot_ids"):  # Handle single-agent group
                            connect_to_plot_id = to_group_info["agent_plot_ids"][0]

                    elif to_node_id in standalone_agents_plot_ids:  # Target is a Agent
                        connect_to_plot_id = standalone_agents_plot_ids[to_node_id]
                        target_cluster_id = None  # No cluster for standalone agent

                    # If target couldn't be resolved in plot, skip edge to it
                    if not connect_to_plot_id:
                        print(f"Warning (Plot): Could not find plot target point for node ID '{to_node_id}'. Skipping edge from '{from_node_id}'.")
                        continue

                    # Determine edge attributes (style, color, logical head/tail)
                    edge_attrs = {}
                    source_cluster_id = (
                        agents_plot_info.get(connect_from_plot_id, {}).get("group_cluster_id")
                        if from_node_id not in groups_plot_info and from_node_id in standalone_agents_plot_ids
                        else f"cluster_{from_node_id}" if from_node_id in groups_plot_info else None
                    )

                    lhead = target_cluster_id
                    ltail = source_cluster_id  # Logical tail pointing to the cluster *containing* the source node

                    if is_from_start_node:
                        edge_attrs = {"color": "darkgreen", "penwidth": "2.0", "style": "bold"}
                        if lhead:
                            edge_attrs["lhead"] = lhead
                    else:
                        edge_attrs = {"style": "dashed", "color": "darkblue", "penwidth": "1.5", "constraint": "true"}
                        if ltail:
                            edge_attrs["ltail"] = ltail
                        if lhead:
                            edge_attrs["lhead"] = lhead

                    # Remove None values from edge_attrs before adding edge
                    edge_attrs = {k: v for k, v in edge_attrs.items() if v is not None}
                    dot.edge(connect_from_plot_id, connect_to_plot_id, **edge_attrs)

            return dot
