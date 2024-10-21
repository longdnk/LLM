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
from langchain_chroma import Chroma
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

urls = [
    "https://finance.yahoo.com/news/chinese-stocks-rebound-ministry-hints-013837670.html",
    # "https://finance.yahoo.com/news/tsmc-third-quarter-profit-seen-000227274.html",
    # "https://finance.yahoo.com/news/boeing-endless-doom-loop-gives-180000720.html",
    # "https://finance.yahoo.com/news/asian-investors-wary-china-briefing-222159266.html",
    # "https://finance.yahoo.com/news/how-trump-could-exert-new-controls-over-the-fed--even-without-firing-powell-130015225.html",
    # "https://finance.yahoo.com/news/did-exxon-lie-about-recycling-california-widens-climate-fight-with-kind-of-new-legal-strategy-133006452.html",
    # "https://finance.yahoo.com/news/european-carmakers-descend-paris-showcase-062006706.html",
    # "https://finance.yahoo.com/news/the-bull-market-is-2-years-old-heres-where-wall-street-thinks-stocks-go-next-100050648.html",
    # "https://finance.yahoo.com/news/retail-sales-big-banks-results-and-netflix-earnings-what-to-know-this-week-114506799.html",
    # "https://finance.yahoo.com/news/tesla-stock-selloff-after-robotaxi-event-could-be-just-the-beginning-pros-warn-155630131.html",
    # "https://finance.yahoo.com/news/does-warren-buffett-know-something-121500342.html",
    # "https://finance.yahoo.com/m/13a394cc-c8b2-3eda-936d-f7ee95db4db6/is-the-stock-market-open.html",
    # "https://finance.yahoo.com/news/oppenheimer-predicts-740-rally-2-100113020.html",
    # "https://finance.yahoo.com/news/tsmc-third-quarter-profit-seen-000227274.html",
    # "https://finance.yahoo.com/news/chinese-stocks-rebound-ministry-hints-013837670.html",
    # "https://finance.yahoo.com/news/asian-investors-wary-china-briefing-222159266.html",
    # "https://finance.yahoo.com/news/analyst-adjusts-amd-stock-price-231700331.html",
    # "https://finance.yahoo.com/news/black-swan-author-really-afraid-172433830.html",
    # "https://finance.yahoo.com/news/bitcoin-jumps-traders-weigh-china-055859218.html",
    # "https://finance.yahoo.com/news/tesla-stock-selloff-after-robotaxi-event-could-be-just-the-beginning-pros-warn-155630131.html",
    # "https://finance.yahoo.com/news/billionaire-bill-gates-81-48-220100456.html",
    # "https://finance.yahoo.com/news/possible-stock-splits-2025-2-124000524.html",
    # "https://www.yahoo.com/tech/columbus-day-trading-netflix-bank-090000300.html",
    # "https://finance.yahoo.com/news/1-top-artificial-intelligence-ai-180000360.html",
    # "https://finance.yahoo.com/news/donald-trump-kamala-harris-better-090600766.html",
    # "https://finance.yahoo.com/news/retail-sales-big-banks-results-and-netflix-earnings-what-to-know-this-week-114506799.html",
    # "https://finance.yahoo.com/news/analyst-forecast-palantirs-rally-makes-133700707.html",
    # "https://finance.yahoo.com/m/0caeb92f-0a11-3ecd-aa75-9a1a0b2238dc/stock-market-holidays-2024-.html",
    # "https://finance.yahoo.com/news/super-micro-computer-shares-surge-211100590.html",
    # "https://finance.yahoo.com/m/69a2fb68-524a-3d7e-854d-754de3137fa9/dow-jones-futures-chinese.html",
    # "https://finance.yahoo.com/news/1-stock-buy-1-stock-130500437.html",
    # "https://finance.yahoo.com/news/legendary-investor-unveils-updated-stock-000300093.html",
    # "https://finance.yahoo.com/news/stock-market-today-asian-shares-050730590.html",
    # "https://finance.yahoo.com/news/p-500s-bull-run-looks-115550131.html",
    # "https://finance.yahoo.com/m/82702796-8574-337c-b563-34e16dbf111e/goldman-sachs-netflix-.html",
    # "https://finance.yahoo.com/news/expect-markets-week-100000721.html",
    # "https://finance.yahoo.com/news/where-palantir-stock-1-181500767.html",
    # "https://finance.yahoo.com/news/morning-bid-china-stimulus-gets-043332660.html",
    # "https://finance.yahoo.com/news/whats-best-magnificent-seven-stock-084900888.html",
    # "https://finance.yahoo.com/news/2-best-artificial-intelligence-stocks-073500247.html",
    # "https://finance.yahoo.com/news/oil-prices-fall-more-1-223856975.html",
    # "https://finance.yahoo.com/m/3702fa24-061b-3e29-8a73-cd0f3a79e218/stocks-poised-for-lower-open.html",
    # "https://finance.yahoo.com/news/dollar-extends-gains-while-investors-010931316.html",
    # "https://finance.yahoo.com/news/jamie-dimon-warns-global-risks-190516735.html",
    # "https://finance.yahoo.com/news/oil-could-fall-to-25-per-barrel-by-2050-if-net-zero-emissions-goals-met-iea-says-183256015.html",
    # "https://finance.yahoo.com/news/commentary-trumps-tariff-threats-are-derailing-his-campaign-160625515.html",
    # "https://finance.yahoo.com/news/us-agency-adopts-rule-easier-170016023.html",
    # "https://finance.yahoo.com/news/live/stock-market-today-dow-jumps-over-300-points-to-hit-fresh-record-lead-stock-rebound-200534422.html",
    # "https://finance.yahoo.com/news/nvidia-is-set-to-dominate-another-big-tech-earnings-season-194525244.html",
    # "https://finance.yahoo.com/news/oil-company-phillips-66-says-235117166.html",
    # "https://finance.yahoo.com/news/stock-market-flashed-sell-signal-234118409.html",
    # "https://finance.yahoo.com/news/elon-musk-goes-big-in-2024-campaign-with-75-million-for-trump-and-other-republicans-132345924.html",
    # "https://finance.yahoo.com/news/robinhood-reveals-new-look-will-start-offering-futures-and-index-options-trading-223116254.html",
    # "https://finance.yahoo.com/news/asml-earnings-disappointment-isnt-a-disaster-for-the-ai-chip-sector-211412268.html",
    # "https://finance.yahoo.com/news/asian-stocks-advance-amid-us-222029442.html",
    # "https://finance.yahoo.com/news/hyundai-record-indian-ipo-draws-042530048.html",
    # "https://finance.yahoo.com/news/china-boost-financing-approved-housing-040829155.html",
    # "https://finance.yahoo.com/news/germanys-vay-gets-37-million-040733734.html",
    # "https://finance.yahoo.com/news/argentine-province-creative-solution-president-040536876.html",
    # "https://finance.yahoo.com/news/italy-winemakers-face-yet-another-040017452.html",
    # "https://finance.yahoo.com/news/ecb-step-pace-back-back-040000538.html",
    # "https://finance.yahoo.com/news/turkey-hold-interest-rates-steady-040000050.html",
    # "https://finance.yahoo.com/personal-finance/where-to-keep-emergency-fund-222513004.html",
    # "https://finance.yahoo.com/personal-finance/flood-insurance-cost-155459950.html",
    # "https://finance.yahoo.com/personal-finance/how-much-is-homeowners-insurance-201445503.html",
    # "https://finance.yahoo.com/personal-finance/cash-to-close-180256531.html",
    # "https://finance.yahoo.com/personal-finance/va-construction-loan-195032285.html",
    # "https://finance.yahoo.com/personal-finance/when-will-mortgage-rates-go-down-164144910.html",
    # "https://finance.yahoo.com/personal-finance/zelle-scams-193209132.html",
    # "https://finance.yahoo.com/personal-finance/sent-money-wrong-person-cash-app-zelle-venmo-210059447.html",
    # "https://finance.yahoo.com/personal-finance/sneaky-tricks-credit-card-issuers-use-175600159.html",
    # "https://finance.yahoo.com/personal-finance/save-on-travel-shoulder-season-230049463.html",
    # "https://finance.yahoo.com/personal-finance/can-you-pause-credit-card-payments-183228901.html",
    # "https://finance.yahoo.com/news/guess-percent-people-4-million-193048041.html",
    # "https://finance.yahoo.com/news/warren-buffett-buying-shares-legal-085100537.html",
    # "https://finance.yahoo.com/news/chinese-stocks-rise-eyes-housing-014731600.html",
    # "https://finance.yahoo.com/news/china-life-says-profit-jump-105656982.html",
    # "https://finance.yahoo.com/news/60-old-canadian-earning-9-180018507.html",
    # "https://finance.yahoo.com/news/nvidias-biggest-problem-might-even-164207019.html",
    # "https://finance.yahoo.com/news/asian-stocks-advance-amid-us-222029442.html",
    # "https://finance.yahoo.com/m/224fc26a-0d0d-32d8-82aa-4250fccf039a/dow-jones-futures-taiwan.html",
    # "https://finance.yahoo.com/news/rmds-start-soon-want-convert-123106570.html",
    # "https://finance.yahoo.com/news/jim-cramer-says-earnings-season-000026465.html",
    # "https://finance.yahoo.com/news/3-tenacious-stocks-could-millionaire-133700075.html",
    # "https://finance.yahoo.com/news/chinese-stocks-extend-decline-peak-013258907.html",
    # "https://finance.yahoo.com/news/stock-market-today-dow-jumps-040825975.html",
    # "https://finance.yahoo.com/news/artificial-intelligence-ai-energy-consumption-220000409.html",
    # "https://finance.yahoo.com/news/billionaire-steven-cohen-sold-87-090600030.html",
    # "https://finance.yahoo.com/m/223df73d-6961-3694-a32a-fb10a7086788/blowout-growth-from-3-stocks.html",
    # "https://finance.yahoo.com/news/chinese-stocks-attractive-investment-managers-192028893.html",
    # "https://finance.yahoo.com/news/iwp-532m-inflows-212910889.html",
    # "https://finance.yahoo.com/news/jpmorgan-kicks-off-post-earnings-125053312.html",
    # "https://finance.yahoo.com/news/bond-markets-heaving-fed-rate-141700298.html",
    # "https://finance.yahoo.com/news/asian-stocks-track-us-selloff-220638744.html",
    # "https://finance.yahoo.com/news/nvidia-is-set-to-dominate-another-big-tech-earnings-season-194525244.html",
    # "https://finance.yahoo.com/news/donald-trump-is-set-for-a-mcdonalds-stop-in-latest-2024-campaign-effort-to-claim-the-golden-arches-080055004.html",
    # "https://finance.yahoo.com/news/samsung-electronics-estimates-274-jump-235615435.html",
    # "https://finance.yahoo.com/news/oil-prices-extend-gains-as-markets-wait-for-israels-retaliation-against-iran-154717391.html",
    # "https://finance.yahoo.com/news/firing-cylinders-mrc-global-nyse-081847099.html",
    # "https://finance.yahoo.com/news/q2-earnings-highlights-wayfair-nyse-081916819.html",
    # "https://finance.yahoo.com/news/analyst-reboots-reddit-stock-price-000300546.htm",
    # "https://finance.yahoo.com/news/lockheed-martin-remains-apos-prime-163912207.html",
    # "https://finance.yahoo.com/news/asian-traders-cautious-ahead-china-223430584.html",
    # "https://finance.yahoo.com/news/ares-said-agree-acquisition-gcp-021839259.html",
    # "https://finance.yahoo.com/news/middle-east-conflict-keeps-markets-004300395.html",
    # "https://finance.yahoo.com/news/the-ai-stock-surge-signals-investors-will-be-patient-for-profit-morning-brief-100500699.html",
    # "https://finance.yahoo.com/news/live/stock-market-today-nasdaq-futures-jump-as-tsmc-outlook-eases-ai-worries-105447024.html",
    # "https://finance.yahoo.com/news/tsmc-profit-beats-estimates-during-054637777.html",
    # "https://finance.yahoo.com/news/kamala-harris-has-a-new-campaign-trail-partner-mark-cuban-090033175.html",
    # "https://finance.yahoo.com/news/gold-rallies-record-tight-us-064949081.html",
    # "https://finance.yahoo.com/news/donald-trump-is-set-for-a-mcdonalds-stop-in-latest-2024-campaign-effort-to-claim-the-golden-arches-080055004.html",
    # "https://finance.yahoo.com/news/the-ai-stock-surge-signals-investors-will-be-patient-for-profit-morning-brief-100500699.html",
    # "https://finance.yahoo.com/news/nvidia-is-set-to-dominate-another-big-tech-earnings-season-194525244.html",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

# Add to vectorDB
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()


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
    generation = rag_chain.invoke({"context": documents, "question": question})
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
            if 'generate' in output and 'generation' in output['generate']['keys']:
                generation_content = output['generate']['keys']['generation']
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
