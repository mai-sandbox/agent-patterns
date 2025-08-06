"""
Two-Stage Review Agent using LangGraph

This module implements a generic two-stage review system where:
1. A ReAct agent generates an initial response
2. A review agent evaluates the output quality
3. If approved, the process ends; if not, it loops back for revision

The implementation is designed to be generic and accept arbitrary ReAct agent configurations.
"""

from typing import Annotated, Any, Callable, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class ReviewState(TypedDict):
    """State schema for the two-stage review agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_agent: Literal["react", "review"]
    iteration_count: int
    review_decision: Optional[Literal["approve", "reject"]]
    max_iterations: int
    react_agent_config: Dict[str, Any]
    review_criteria: str


def create_react_agent_node(llm: BaseChatModel, tools: Optional[List[BaseTool]] = None) -> callable:
    """
    Creates a generic ReAct agent node that can work with any LLM and tools.
    
    Args:
        llm: The language model to use for the ReAct agent
        tools: Optional list of tools the agent can use
        
    Returns:
        A node function that implements the ReAct pattern
    """
    def react_agent_node(state: ReviewState) -> Dict[str, Any]:
        """
        ReAct agent node that processes the current messages and generates a response.
        Implements the Reasoning + Acting pattern.
        """
        messages = state["messages"]
        iteration = state["iteration_count"]
        
        # Add system message for ReAct pattern if this is the first iteration
        if iteration == 0 or not any(isinstance(msg, SystemMessage) for msg in messages):
            react_system_prompt = """You are a helpful AI assistant that uses the ReAct (Reasoning + Acting) pattern.

For each user query:
1. Think step by step about what you need to do
2. If you have tools available, decide if you need to use them
3. Take action by using tools if necessary
4. Observe the results
5. Provide a comprehensive final answer

Be thorough in your reasoning and provide clear, helpful responses."""
            
            messages = [SystemMessage(content=react_system_prompt)] + messages
        
        # If tools are available, bind them to the model
        if tools:
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke(messages)
        else:
            response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "current_agent": "review",
            "iteration_count": iteration + 1
        }
    
    return react_agent_node


def create_review_agent_node(llm: BaseChatModel) -> callable:
    """
    Creates a review agent node that evaluates the quality of responses.
    
    Args:
        llm: The language model to use for review
        
    Returns:
        A node function that reviews responses
    """
    def review_agent_node(state: ReviewState) -> Dict[str, Any]:
        """
        Review agent node that evaluates the quality of the ReAct agent's response.
        """
        messages = state["messages"]
        review_criteria = state.get("review_criteria", "")
        
        # Get the last AI message (the response to review)
        last_ai_message = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                last_ai_message = msg
                break
        
        if not last_ai_message:
            return {
                "review_decision": "reject",
                "messages": [AIMessage(content="No response found to review.")]
            }
        
        # Create review prompt
        review_prompt = f"""Please review the following AI response for quality and completeness.

Review Criteria:
{review_criteria if review_criteria else "The response should be helpful, accurate, complete, and well-reasoned."}

AI Response to Review:
{last_ai_message.content}

Based on your review, respond with either:
- "APPROVE: [brief reason]" if the response meets the criteria
- "REJECT: [specific feedback for improvement]" if the response needs improvement

Your decision:"""
        
        review_messages = [HumanMessage(content=review_prompt)]
        review_response = llm.invoke(review_messages)
        
        # Parse the review decision
        review_content = review_response.content.strip()
        if review_content.startswith("APPROVE"):
            decision = "approve"
        else:
            decision = "reject"
        
        return {
            "review_decision": decision,
            "messages": [review_response],
            "current_agent": "react" if decision == "reject" else "review"
        }
    
    return review_agent_node


def should_continue(state: ReviewState) -> Literal["react_agent", "end"]:
    """
    Conditional routing function that determines whether to continue or end.
    
    Args:
        state: Current state of the review process
        
    Returns:
        Next node to execute or END
    """
    review_decision = state.get("review_decision")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    # End if approved or max iterations reached
    if review_decision == "approve" or iteration_count >= max_iterations:
        return "end"
    
    # Continue with ReAct agent if rejected
    return "react_agent"


def create_two_stage_review_agent(
    react_llm: Optional[BaseChatModel] = None,
    review_llm: Optional[BaseChatModel] = None,
    tools: Optional[List[BaseTool]] = None,
    max_iterations: int = 3,
    review_criteria: str = ""
) -> StateGraph:
    """
    Creates a two-stage review agent graph.
    
    Args:
        react_llm: Language model for the ReAct agent (defaults to GPT-4)
        review_llm: Language model for the review agent (defaults to same as react_llm)
        tools: Optional tools for the ReAct agent
        max_iterations: Maximum number of review iterations
        review_criteria: Specific criteria for review evaluation
        
    Returns:
        Compiled StateGraph ready for execution
    """
    # Initialize default models if not provided
    if react_llm is None:
        react_llm = init_chat_model("gpt-4", temperature=0.1)
    
    if review_llm is None:
        review_llm = react_llm
    
    # Create the state graph
    graph_builder = StateGraph(ReviewState)
    
    # Create node functions
    react_node = create_react_agent_node(react_llm, tools)
    review_node = create_review_agent_node(review_llm)
    
    # Add nodes to the graph
    graph_builder.add_node("react_agent", react_node)
    graph_builder.add_node("review_agent", review_node)
    
    # Add edges
    graph_builder.add_edge(START, "react_agent")
    graph_builder.add_edge("react_agent", "review_agent")
    
    # Add conditional edge from review_agent
    graph_builder.add_conditional_edges(
        "review_agent",
        should_continue,
        {
            "react_agent": "react_agent",
            "end": END
        }
    )
    
    return graph_builder.compile()


# Default configuration for deployment
def create_default_agent() -> StateGraph:
    """
    Creates a default two-stage review agent for deployment.
    Uses environment variables for configuration.
    """
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Determine which LLM to use based on available API keys
    llm = None
    if os.getenv("OPENAI_API_KEY"):
        llm = init_chat_model("gpt-4", temperature=0.1)
    elif os.getenv("ANTHROPIC_API_KEY"):
        llm = init_chat_model("claude-3-sonnet-20240229", temperature=0.1)
    elif os.getenv("GOOGLE_API_KEY"):
        llm = init_chat_model("gemini-pro", temperature=0.1)
    else:
        # Fallback - will require API key to be set
        llm = init_chat_model("gpt-4", temperature=0.1)
    
    # Default review criteria
    default_criteria = """
    The response should be:
    1. Accurate and factually correct
    2. Complete and addresses all aspects of the question
    3. Well-structured and easy to understand
    4. Helpful and actionable
    5. Free from harmful or inappropriate content
    """
    
    return create_two_stage_review_agent(
        react_llm=llm,
        review_llm=llm,
        tools=None,  # No tools by default - can be customized
        max_iterations=3,
        review_criteria=default_criteria
    )


# Create the app instance for deployment
app = create_default_agent()


# Example usage function for testing
def run_example():
    """Example of how to use the two-stage review agent."""
    
    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content="Explain the concept of machine learning in simple terms.")],
        "current_agent": "react",
        "iteration_count": 0,
        "review_decision": None,
        "max_iterations": 3,
        "react_agent_config": {},
        "review_criteria": "The explanation should be clear, accurate, and suitable for beginners."
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    
    print("Final result:")
    for message in result["messages"]:
        print(f"{message.__class__.__name__}: {message.content}")
        print("-" * 50)


if __name__ == "__main__":
    run_example()


