

import os
from dotenv import load_dotenv

# Add references
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import (
    ConnectedAgentTool,
    MessageRole,
    ListSortOrder
)
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient

# Clear console
os.system('cls' if os.name == 'nt' else 'clear')

# Load env variables
load_dotenv()
project_endpoint = os.getenv("https://agentic-ai-training-batch2.openai.azure.com/") #PROJECT_ENDPOINT
model_deployment = os.getenv("gpt-4o")
search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
search_index = os.getenv("AZURE_SEARCH_INDEX")

# Connect to Agents Client
agents_client = AgentsClient(
    endpoint=project_endpoint,
    credential=DefaultAzureCredential(
        exclude_environment_credential=True,
        exclude_managed_identity_credential=True
    ),
)

# 🔍 Azure AI Search client (RAG)
search_client = SearchClient(
    endpoint=search_endpoint,
    index_name=search_index,
    credential=DefaultAzureCredential()
)

# -------------------------------
# RAG TOOL FUNCTION
# -------------------------------
def retrieve_legal_references(query: str) -> str:
    """
    Retrieves relevant legal or regulatory references from Azure AI Search
    """
    results = search_client.search(
        search_text=query,
        top=5
    )

    documents = []
    for result in results:
        documents.append(result["content"])

    return "\n\n".join(documents)


with agents_client:

    # -----------------------------
    # Compliance Risk Agent
    # -----------------------------
    compliance_agent = agents_client.create_agent(
        model=model_deployment,
        name="compliance_risk_agent",
        instructions="""
Assess legal compliance risk in the document.
Return:
- Risk Level (High / Medium / Low)
- Short explanation
"""
    )

    # -----------------------------
    # Clause Classification Agent
    # -----------------------------
    clause_agent = agents_client.create_agent(
        model=model_deployment,
        name="clause_classification_agent",
        instructions="""
Identify and classify clauses:
Termination, Liability, Data Privacy, Confidentiality, Payment.
Return concise results.
"""
    )

    # -----------------------------
    # Legal Complexity Agent
    # -----------------------------
    complexity_agent = agents_client.create_agent(
        model=model_deployment,
        name="legal_complexity_agent",
        instructions="""
Estimate legal review complexity:
Low / Medium / High
Provide short justification.
"""
    )

    # -----------------------------
    # 🔍 RAG Retrieval Agent
    # -----------------------------
    rag_agent = agents_client.create_agent(
        model=model_deployment,
        name="legal_rag_agent",
        instructions="""
Retrieve relevant laws, regulations, or precedents
based on the provided legal document.
Return only grounded reference text.
"""
    )

    # -----------------------------
    # Connected Agent Tools
    # -----------------------------
    compliance_tool = ConnectedAgentTool(
        id=compliance_agent.id,
        name="compliance_risk_agent",
        description="Evaluates compliance risk"
    )

    clause_tool = ConnectedAgentTool(
        id=clause_agent.id,
        name="clause_classification_agent",
        description="Classifies legal clauses"
    )

    complexity_tool = ConnectedAgentTool(
        id=complexity_agent.id,
        name="legal_complexity_agent",
        description="Estimates legal complexity"
    )

    rag_tool = ConnectedAgentTool(
        id=rag_agent.id,
        name="legal_rag_agent",
        description="Retrieves grounded legal references"
    )

    # -----------------------------
    # Orchestrator Agent (WITH RAG)
    # -----------------------------
    legal_orchestrator = agents_client.create_agent(
        model=model_deployment,
        name="legal_orchestrator_agent",
        instructions="""
Review the legal document.

Steps:
1. Identify clause types
2. Assess compliance risk
3. Estimate legal complexity
4. Retrieve relevant legal references using RAG
5. Produce a grounded summary with references

Rules:
- Do NOT invent laws
- Cite retrieved references explicitly
""",
        tools=[
            clause_tool.definitions[0],
            compliance_tool.definitions[0],
            complexity_tool.definitions[0],
            rag_tool.definitions[0]
        ]
    )

    # -----------------------------
    # Run the Review
    # -----------------------------
    print("Creating agent thread...")
    thread = agents_client.threads.create()

    legal_text = input("\nPaste the legal document text:\n\n")

    agents_client.messages.create(
        thread_id=thread.id,
        role=MessageRole.USER,
        content=legal_text
    )

    print("\nProcessing legal document with RAG...\n")

    run = agents_client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=legal_orchestrator.id
    )

    if run.status == "failed":
        print(f"Run failed: {run.last_error}")

    messages = agents_client.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING
    )

    for message in messages:
        if message.role == MessageRole.AGENT and message.text_messages:
            print("LEGAL COMPLIANCE OUTPUT:\n")
            print(message.text_messages[-1].text.value)

    # -----------------------------
    # Clean up
    # -----------------------------
    agents_client.delete_agent(legal_orchestrator.id)
    agents_client.delete_agent(compliance_agent.id)
    agents_client.delete_agent(clause_agent.id)
    agents_client.delete_agent(complexity_agent.id)
    agents_client.delete_agent(rag_agent.id)

    print("\nAll agents cleaned up.")
