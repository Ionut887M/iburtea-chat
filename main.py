import os

from datasets import load_dataset
from haystack.agents import Tool
from haystack.agents.base import ToolsManager
from haystack.agents.memory import ConversationSummaryMemory
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode
from haystack.nodes import PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack.utils import convert_files_to_docs
key = os.getenv("HF_TOKEN", None)

from haystack.agents import Agent

path = r"path_to_dataset"
doc_docx = convert_files_to_docs(dir_path=path)


document_store = InMemoryDocumentStore(use_bm25=True)
document_store.write_documents(doc_docx)

retriever = BM25Retriever(document_store=document_store, top_k=3)

prompt_template = PromptTemplate(
    prompt="""
    Based on the client's query provided below and the knowledge from the documents, generate a detailed offer for the client. The offer should include:
    1. A description of the requested application.
    2. The technologies used for the application development (e.g., technology stack - front-end, back-end, databases, etc.).
    3. Concrete and detailed tasks required for the application development, including those not directly mentioned by the client but necessary (e.g., financial sections, invoicing, etc.).
    
    
    Documents: {join(documents)}
    query: {query}
    Offer:
    """,
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(
    model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=key,
    default_prompt_template=prompt_template,
    max_length=500,
    model_kwargs={"model_max_length": 10000}
)
generative_pipeline = Pipeline()
generative_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
generative_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

search_tool = Tool(
    name="generate_offer",
    pipeline_or_node=generative_pipeline,
    description="Use this tool to generate a detailed offer based on client requirements",
    output_variable="answers",
)

agent_prompt_node = PromptNode(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    api_key=key,
    max_length=256,
    stop_words=["Observation:"],
    model_kwargs={"temperature": 0.5},
)

memory_prompt_node = PromptNode(
    "philschmid/bart-large-cnn-samsum", max_length=256, model_kwargs={"task_name": "text2text-generation"}
)
memory = ConversationSummaryMemory(memory_prompt_node, prompt_template="{chat_transcript}")

agent_prompt = """
In the following conversation, a human user interacts with an AI Agent. The human user poses a request, and the AI Agent goes through several steps to generate a detailed offer based on the client's requirements.
The AI Agent must use the available tools to find up-to-date information and generate the offer. The final offer should be based solely on the output of the tools and the provided documents. The AI Agent should ignore its knowledge when generating the offer.
The AI Agent has access to these tools:
{tool_names_with_descriptions}

The following is the previous conversation between a human and The AI Agent:
{memory}

AI Agent responses must start with one of the following:

Thought: [the AI Agent's reasoning process]
Tool: [tool names] (on a new line) Tool Input: [input as a question for the selected tool WITHOUT quotation marks and on a new line] (These must always be provided together and on separate lines.)
Observation: [tool's result]
The final answer should contain the description of application, technologies used and concrete and detalied tasks required for application development
Final Answer: [final answer to the human user's request]
When selecting a tool, the AI Agent must provide both the "Tool:" and "Tool Input:" pair in the same response, but on separate lines.

The AI Agent should not ask the human user for additional information, clarification, or context.
If the AI Agent cannot find specific information after exhausting available tools and approaches, it answers with Final Answer: inconclusive

Request: {query}
Thought:
{transcript}
"""


def resolver_function(query, agent, agent_step):
    return {
        "query": query,
        "tool_names_with_descriptions": agent.tm.get_tool_names_with_descriptions(),
        "transcript": agent_step.transcript,
        "memory": agent.memory.load(),
    }


conversational_agent = Agent(
    agent_prompt_node,
    prompt_template=agent_prompt,
    prompt_parameters_resolver=resolver_function,
    memory=memory,
    tools_manager=ToolsManager([search_tool]),
)
conversational_agent.run("dezvoltarea unei aplicatii de tip glovo")
