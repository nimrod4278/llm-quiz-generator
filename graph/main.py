from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph, START

from graph.state import InterviewState
from graph.dialog import generate_question
from graph.answer import gen_answer
from graph.quiz import gen_quiz_chain
from graph.perspective import survey_subjects

from graph.quiz import ResearchState

max_num_turns = 5

def route_messages(state: InterviewState, name: str = "Subject_Matter_Expert"):
    messages = state["messages"]
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )
    if num_responses >= max_num_turns:
        return END
    last_question = messages[-2]
    if last_question.content.endswith("Thank you so much for your help!"):
        return END
    return "ask_question"

def initialize_research(state: ResearchState):
    topic = state["topic"]
    perspectives = survey_subjects.invoke(topic),
 
    return {
        **state,
        "editors": perspectives[0].editors,
    }


def gen_quiz(
    state: ResearchState,
):
    quiz = gen_quiz_chain.invoke({"article": state["interview_results"], "level": state["level"], "length": 10})
    return {
        **state,
        "quiz": quiz
    }

def conduct_interviews(state: ResearchState):
    topic = state["topic"]
    initial_states = [
        {
            "editor": editor,
            "messages": [
                AIMessage(
                    content=f"So you said you were writing an article on {topic}?",
                    name="Subject_Matter_Expert",
                )
            ],
        }
        for editor in state["editors"]
    ]
    # We call in to the sub-graph here to parallelize the interviews
    interview_results = interview_graph.batch(initial_states)

    all_interviews = ""
    for interview in interview_results:
        interview_text = "\n".join([msg.content for msg in interview["messages"]])
        all_interviews += f"\n\n{interview_text}"

    return {
        **state,
        "interview_results": all_interviews,
    }

builder = StateGraph(InterviewState)

builder.add_node("ask_question", generate_question)
builder.add_node("answer_question", gen_answer)
builder.add_conditional_edges("answer_question", route_messages)
builder.add_edge("ask_question", "answer_question")

builder.add_edge(START, "ask_question")
interview_graph = builder.compile().with_config(run_name="Conduct Interviews")

builder_of_storm = StateGraph(ResearchState)
builder_of_storm.add_node("initialize_research", initialize_research)
builder_of_storm.add_node("conduct_interviews", conduct_interviews)
builder_of_storm.add_node("gen_quiz", gen_quiz)
builder_of_storm.add_edge(START, "initialize_research")
builder_of_storm.add_edge("initialize_research", "conduct_interviews")
builder_of_storm.add_edge("conduct_interviews", "gen_quiz")
builder_of_storm.add_edge("gen_quiz", END)
storm = builder_of_storm.compile()

if __name__ == "__main__":
    example_topic = "how to make the perfect nopolian pizza?"
    quiz = storm.invoke({"topic": example_topic})
    print(quiz)