from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.checkpoint.memory import MemorySaver

from .nodes import agent, rewrite, generate, grade_documents 
from .state import AgentState
from .tools import retriever_tool  

# Define the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

# Define edges
workflow.add_edge(START, "agent")

# Conditional edges from agent
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

# Conditional edges from retrieve
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "rewrite": "rewrite",
        "generate": "generate",
    }
)

workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)

# Compile the graph
# checkpointer = MemorySaver()
# graph = workflow.compile(checkpointer=checkpointer)
graph = workflow.compile()

