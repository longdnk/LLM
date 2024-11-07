from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from typing import Dict, TypedDict
from langchain_core.messages import BaseMessage
import json
import operator
from typing import Annotated, Sequence, TypedDict
from fastapi import APIRouter, status, Request
from langchain import hub
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.output_parsers import StrOutputParser

# from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import ToolInvocation
import pprint
import os
from langgraph.graph import END, StateGraph

# OS environ
os.environ["OPENAI_API_KEY"] = ""
os.environ["TAVILY_API_KEY"] = ""


class RequestItem(BaseModel):
    text: str

rag_router = APIRouter(prefix="/rags", tags=["rags"])

# urls = [
#     "https://finance.yahoo.com/news/chinese-stocks-rebound-ministry-hints-013837670.html",
#     "https://finance.yahoo.com/news/mortgage-rates-jump-most-since-160000104.html",
#     "https://finance.yahoo.com/news/ibm-poised-apos-solid-apos-174213005.html",
#     "https://www.example.com/news/fed-rate-cut",
#     "https://finance.yahoo.com/news/why-pro-trader-tom-sosnoff-says-to-bet-on-yourself-instead-of-passive-index-funds-190057935.html",
#     "https://finance.yahoo.com/news/northvolt-founder-next-big-green-040038431.html",
#     "https://finance.yahoo.com/news/1-stock-split-stock-set-join-nvidia-apple-microsoft-amazon-alphabet-meta-1-trillion-club-19470000.html",
#     "https://finance.yahoo.com/news/sri-lanka-begins-voting-presidential-013152309.html",
#     "https://finance.yahoo.com/news/why-strong-earnings-and-cash-flow-are-key-to-navigating-value-traps-181838014.html",
#     "https://finance.yahoo.com/news/tesla-robotaxi-event-analysts-weigh-in-on-what-to-expect-from-ceo-elon-musks-big-moment-164748104.html",
#     "https://finance.yahoo.com/news/own-home-hope-buy-one-090020148.html",
#     "https://finance.yahoo.com/news/fake-names-lured-billionaire-salinas-211738755.html",
#     "https://finance.yahoo.com/news/us-30-year-fixed-rate-mortgage-falls-6-09-22103212.html",
#     "https://finance.yahoo.com/news/q2-earnings-roundup-lamb-weston-nyse-lw-rest-shelf-stable-food-segment-00000000.html",
#     "https://finance.yahoo.com/news/three-mile-island-nuclear-reactor-to-restart-under-microsoft-deal-154812499.html",
#     "https://finance.yahoo.com/news/x-names-brazil-legal-representative-003638875.html",
#     "https://www.example.com/news/broadcom-stock-split",
#     "https://finance.yahoo.com/news/want-safe-dividend-income-2024-084800477.html",
#     "https://finance.yahoo.com/news/j-j-subsidiary-files-bankruptcy-193903276.html",
#     "https://finance.yahoo.com/news/billionaire-bond-king-bill-gross-005809707.html?guccounter=1&guce_referrer=aHR0cHM6Ly9maW5hbmNlLnlhaG9vLmNvbS9uZXdzLw&guce_referrer_sig=AQAAAEVfPVEASIT_EVG7Ar3pZP3EFIb7yCyfXZREkyo7-1b5dWIKzVBmgjFfwhaeKV7SIkNetwpNNPgFLPMpbl6g0kuXE7b5o0zS7FnvNfVKfAzP9AyUzqS8og85cBkWxxDWYZgvueIRl4OBSgE41R2jbhZsfZjOCCOPPu-qJObxPy9V",
#     "https://finance.yahoo.com/news/espn-stephen-smith-picks-sports-000822252.html?guccounter=1&guce_referrer=aHR0cHM6Ly9maW5hbmNlLnlhaG9vLmNvbS9uZXdzLw&guce_referrer_sig=AQAAAEVfPVEASIT_EVG7Ar3pZP3EFIb7yCyfXZREkyo7-1b5dWIKzVBmgjFfwhaeKV7SIkNetwpNNPgFLPMpbl6g0kuXE7b5o0zS7FnvNfVKfAzP9AyUzqS8og85cBkWxxDWYZgvueIRl4OBSgE41R2jbhZsfZjOCCOPPu-qJObxPy9V",
#     "https://finance.yahoo.com/news/us-mortgage-rates-fall-further-stoking-housing-optimism-22192212.html",
#     "https://finance.yahoo.com/news/elon-musk-reacts-mark-cuban-says-23000000.html",
#     "https://finance.yahoo.com/news/caused-by-crowdstrike-delta-ceo-cites-tech-disruption-as-earnings-miss-113034365.html",
#     "https://finance.yahoo.com/news/crypto-investment-firm-deus-x-capital-unveils-defi-unit-23000000.html",
#     "https://www.example.com/news/apple-iphone-16-launch",
#     "https://finance.yahoo.com/news/dominos-nyse-dpz-reports-sales-101824438.html",
#     "https://finance.yahoo.com/news/milton-surprise-damage-unleashed-powerful-215914303.html",
#     "https://finance.yahoo.com/news/existing-home-sales-fall-in-august-despite-lower-mortgage-rates-171246859.html",
#     "https://finance.yahoo.com/news/stock-market-today-dow-sp-500-close-record-highs-nasdaq-surges-amid-rate-cut-euphoria-14530400.html",
#     "https://www.example.com/news/base-metals-demand",
#     "https://finance.yahoo.com/news/stock-market-today-dow-ekes-out-another-record-amid-winning-week-for-stocks-200133466.html",
#     "https://finance.yahoo.com/news/intel-gains-report-qualcomm-made-192727186.html",
#     "https://www.example.com/news/kids-facing-sudden-wealth-syndrome",
#     "https://finance.yahoo.com/news/elad-gil-latest-ai-bet-231412988.html",
#     "https://finance.yahoo.com/news/meituan-63-stock-surge-faces-risks-china-consumer-malaise-00000000.html",
#     "https://finance.yahoo.com/news/warren-buffett-could-bought-379-090600668.html",
#     "https://finance.yahoo.com/news/sri-lankans-vote-tight-race-013000137.html",
#     "https://finance.yahoo.com/news/look-back-perishable-food-stocks-q2-earnings-tyson-foods-vs-rest-pack-124148838.html",
#     "https://finance.yahoo.com/news/fedex-reports-sales-below-analyst-estimates-q3-earnings-19480000.html",
#     "https://finance.yahoo.com/news/millennial-fire-couple-shares-moving-180302476.html",
#     "https://finance.yahoo.com/news/bank-korea-pivots-rate-cut-010705411.html",
#     "https://finance.yahoo.com/news/summers-sees-fed-rate-projections-upended-higher-mortgage-rates-22102212.html",
#     "https://finance.yahoo.com/news/september-apartment-rents-edge-higher-184341698.html",
#     "https://finance.yahoo.com/news/fintech-market-stirs-celero-seeks-222143615.html",
#     "https://finance.yahoo.com/news/gold-hovers-near-record-highs-heres-where-analysts-say-its-headed-next-165252182.html?guccounter=1&guce_referrer=aHR0cHM6Ly9maW5hbmNlLnlhaG9vLmNvbS8&guce_referrer_sig=AQAAAHIFMgfHo755iL5l7Zcn_xIsEMebkV8hKNWIBgHORjWZwdHjmGvk3GO-H0wAZRgdy40-AHGndzsPQbpR8AvPp-wiAvm2gmHz7QBz682z7c4EhVbPq3f1ZkBIWuuQm2bKYkNuSJ1_ynMO7oB__HB8F7rvDOGT7I1RFNwPw-oTo9JA",
#     "https://finance.yahoo.com/news/us-postal-not-hike-stamp-182738612.html",
#     "https://finance.yahoo.com/news/elon-musk-says-warren-buffett-positioning-kamala-harris-win-23000000.html",
#     "https://finance.yahoo.com/news/mortgage-rates-fall-lowest-level-160400784.html?guccounter=1&guce_referrer=aHR0cHM6Ly9maW5hbmNlLnlhaG9vLmNvbS8&guce_referrer_sig=AQAAAHIFMgfHo755iL5l7Zcn_xIsEMebkV8hKNWIBgHORjWZwdHjmGvk3GO-H0wAZRgdy40-AHGndzsPQbpR8AvPp-wiAvm2gmHz7QBz682z7c4EhVbPq3f1ZkBIWuuQm2bKYkNuSJ1_ynMO7oB__HB8F7rvDOGT7I1RFNwPw-oTo9JA",
#     "https://www.example.com/news/jim-cramer-stock-picks",
#     "https://www.example.com/news/truth-social-stock",
#     "https://finance.yahoo.com/news/why-have-mortgage-rates-gone-4-57-analysts-say-22160203.html",
#     "https://finance.yahoo.com/news/china-unexpectedly-leaves-lending-rates-steady-markets-expect-cuts-soon-22384707.html",
#     "https://finance.yahoo.com/news/sri-lankans-vote-presidential-election-013301889.html",
#     "https://finance.yahoo.com/news/trump-media-nosedives-record-low-161159651.html",
#     "https://finance.yahoo.com/news/mortgage-rates-inch-closer-to-6-following-fed-rate-cut-160309788.html",
#     "https://finance.yahoo.com/news/defi-lender-sky-ratifies-plan-offboard-wrapped-bitcoin-223355875.html",
#     "https://finance.yahoo.com/news/reflecting-hr-software-stocks-q2-earnings-paylocity-124149985.html",
#     "https://www.example.com/news/russia-exit-tax-hike",
#     "https://www.example.com/news/nike-ceo-appointment",
#     "https://www.example.com/news/african-currencies-pressure",
#     "https://finance.yahoo.com/news/billionaire-stanley-druckenmiller-sold-88-085100444.html",
#     "https://finance.yahoo.com/news/why-social-media-companies-keep-183000571.html",
#     "https://finance.yahoo.com/news/protocol-village-nexus-launches-worlds-234023194.html",
#     "https://finance.yahoo.com/news/rich-enough-top-1-heres-213018631.html?guccounter=1&guce_referrer=aHR0cHM6Ly9maW5hbmNlLnlhaG9vLmNvbS9uZXdzLw&guce_referrer_sig=AQAAAEVfPVEASIT_EVG7Ar3pZP3EFIb7yCyfXZREkyo7-1b5dWIKzVBmgjFfwhaeKV7SIkNetwpNNPgFLPMpbl6g0kuXE7b5o0zS7FnvNfVKfAzP9AyUzqS8og85cBkWxxDWYZgvueIRl4OBSgE41R2jbhZsfZjOCCOPPu-qJObxPy9V",
#     "https://finance.yahoo.com/news/sumitomo-hires-juan-toro-ceo-143000964.html",
#     "https://finance.yahoo.com/news/brazil-trims-2024-spending-freeze-002356049.html",
#     "https://finance.yahoo.com/news/mixed-mortgage-signals-inflation-continues-145424235.html",
# ]

# docs = [WebBaseLoader(url).load() for url in urls]
# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=500, chunk_overlap=100
# )
# doc_splits = text_splitter.split_documents(docs_list)

# # Add to vectorDB
# vectorstore = Chroma.from_documents(
#     documents=doc_splits,
#     collection_name="yahoo-news",
#     embedding=OpenAIEmbeddings(),
#     persist_directory="yf-persist",
# )

client = Chroma(persist_directory="yf-persist", embedding_function=OpenAIEmbeddings(), collection_name="yahoo-news")
retriever = client.as_retriever(search_kwargs={"k": 3})

ids = client._collection.get()['ids']
print(f"number of documents: {len(ids)}")

class GraphState(TypedDict):
    """
    Represents the state of an agent in the conversation.

    Attributes:
        keys: A dictionary where each key is a string and the value is expected to be a list or another structure
              that supports addition with `operator.add`. This could be used, for instance, to accumulate messages
              or other pieces of data throughout the graph.
    """

    keys: Dict[str, any]


### Nodes ##
def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        dict: New key added to state, documents, that contains documents.
    """
    print("---RETRIEVE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = retriever.invoke(question)
    return {"keys": {"documents": documents, "question": question}}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        dict: New key added to state, generation, that contains generation.
    """
    print("---GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    add_quest = "(Make sure to answer the question as shortly and concisely as possible. The answer should not be more than 100 characters and should not be more than 12 words.)"
    quest = f"{add_quest}\n\n{question}"
    generation = rag_chain.invoke({"context": documents, "question": quest})
    return {
        "keys": {"documents": documents, "question": question, "generation": generation}
    }


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        dict: New key added to state, filtered_documents, that contains relevant documents.
    """

    print("---CHECK RELEVANCE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Tool
    grade_tool_oai = convert_to_openai_tool(grade)

    # LLM with tool and enforce invocation
    llm_with_tool = model.bind(
        tools=[convert_to_openai_tool(grade_tool_oai)],
        tool_choice={"type": "function", "function": {"name": "grade"}},
    )

    # Parser
    parser_tool = PydanticToolsParser(tools=[grade])

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool | parser_tool

    # Score
    filtered_docs = []
    search = "No"  # Default do not opt for web search to supplement retrieval
    for d in documents:
        score = chain.invoke({"question": question, "context": d.page_content})
        grade = score[0].binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            search = "Yes"  # Perform web search
            continue

    return {
        "keys": {
            "documents": filtered_docs,
            "question": question,
            "run_web_search": search,
        }
    }


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        dict: New value saved to question.
    """

    print("---TRANSFORM QUERY---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    # Create a prompt template with format instructions and the query
    prompt = PromptTemplate(
        template="""You are generating questions that is well optimized for retrieval. \n
        Look at the input and try to reason about the underlying sematic intent / meaning. \n
        Here is the initial question:
        \n ------- \n
        {question}
        \n ------- \n
        Formulate an improved question: """,
        input_variables=["question"],
    )

    # Grader
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # Prompt
    chain = prompt | model | StrOutputParser()
    better_question = chain.invoke({"question": question})

    return {"keys": {"documents": documents, "question": better_question}}


def web_search(state):
    """
    Web search using Tavily.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        state (dict): Web results appended to documents.
    """

    print("---WEB SEARCH---")
    state_dict = state["keys"]
    question = state_dict["question"]
    documents = state_dict["documents"]

    tool = TavilySearchResults()
    docs = tool.invoke({"query": question})

    # Kiểm tra xem docs có phải là một danh sách không và không rỗng
    if isinstance(docs, list) and docs:
        web_results = "\n".join([d["content"] for d in docs if "content" in d])
        web_results = Document(page_content=web_results)
        documents.append(web_results)
    else:
        print("No results found or docs is not a list.")

    return {"keys": {"documents": documents, "question": question}}


### Edges
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current state of the agent, including all keys.

    Returns:
        dict: New key added to state, filtered_documents, that contains relevant documents.
    """

    print("---DECIDE TO GENERATE---")
    state_dict = state["keys"]
    question = state_dict["question"]
    filtered_documents = state_dict["documents"]
    search = state_dict["run_web_search"]

    if search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TRANSFORM QUERY and RUN WEB SEARCH---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generatae
workflow.add_node("transform_query", transform_query)  # transform_query
workflow.add_node("web_search", web_search)  # web search

# Build graph
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "web_search")
workflow.add_edge("web_search", "generate")
workflow.add_edge("generate", END)

# Compile
app = workflow.compile()


def retrieval_in_rag(question: str):
    try:
        # Correction for question not present in context
        inputs = {"keys": {"question": f"{question}"}}
        item = app.stream(inputs)
        generation_content = ""
        for output in app.stream(inputs):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value["keys"], indent=2, width=80, depth=None)
            if "generate" in output and "generation" in output["generate"]["keys"]:
                generation_content = output["generate"]["keys"]["generation"]
            pprint.pprint("\n---\n")

        return generation_content

    except Exception as e:
        # Log or print the exception if needed
        print(f"Lỗi xảy ra: {e}")
        return "Cannot find information, please try again"


@rag_router.post("", status_code=status.HTTP_201_CREATED)
async def create_chat(request: Request, request_item: RequestItem):
    try:
        print(request_item)
        result = retrieval_in_rag(request_item.text)
        return {
            "message": "Rag Retrieval success",
            "code": status.HTTP_201_CREATED,
            "data": result,
        }

    except SQLAlchemyError as e:
        # Xử lý ngoại lệ nếu có lỗi xảy ra
        return {
            "message": "Error",
            "code": status.HTTP_500_INTERNAL_SERVER_ERROR,
            "error": str(e),
        }
