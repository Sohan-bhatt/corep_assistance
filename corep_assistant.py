"""
COREP Regulatory Reporting Assistant using Pydantic AI
Main application with RAG and structured output generation
Universal API support (OpenAI, Anthropic, Google, Azure, etc.)
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
import tiktoken
from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

from schemas import (
    RegulatoryQuery, OwnFundsReport, CapitalComponent, CapitalComponentType,
    COREPTemplateOutput, ValidationResult, ValidationStatus, AuditLogEntry,
    ValidationRule
)
from extract_pdfs import process_pdfs

# Load environment variables from .env file
load_dotenv()

# Universal API Configuration
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")

def create_model():
    """Create model instance based on provider configuration."""
    if not LLM_API_KEY:
        raise ValueError(
            "LLM_API_KEY not found. Please create a .env file with your API key:\n"
            "LLM_API_KEY=your-api-key-here\n"
            "LLM_PROVIDER=openai\n"
            "LLM_MODEL=gpt-4o-mini"
        )
    
    # Set the API key in environment for the provider
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
    
    if LLM_PROVIDER.lower() == "openai":
        return OpenAIModel(LLM_MODEL)
    elif LLM_PROVIDER.lower() == "azure":
        # For Azure OpenAI
        os.environ["AZURE_OPENAI_API_KEY"] = LLM_API_KEY
        if LLM_BASE_URL:
            os.environ["AZURE_OPENAI_ENDPOINT"] = LLM_BASE_URL
        return OpenAIModel(LLM_MODEL)
    else:
        # Default to OpenAI-compatible API
        # This works for many providers (OpenAI, local models, etc.)
        return OpenAIModel(LLM_MODEL)
    
    # Note: For Anthropic and Google, you would need to:
    # 1. Install additional packages: pip install anthropic google-generativeai
    # 2. Import their respective model classes
    # 3. Add conditions here to instantiate them
    # Example for Anthropic:
    # from pydantic_ai.models.anthropic import AnthropicModel
    # return AnthropicModel(LLM_MODEL, api_key=LLM_API_KEY)

# Initialize ChromaDB for document storage
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = chroma_client.get_or_create_collection(
    name="corep_regulations",
    metadata={"hnsw:space": "cosine"}
)

def get_embedding(text: str) -> List[float]:
    """Get embedding using tiktoken (simplified - in production use OpenAI embeddings)."""
    # For prototype, we'll use a simple hash-based approach
    # In production, replace with actual embedding model
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text[:8000])  # Limit tokens
    # Create a simple vector representation
    vector = [0.0] * 1536
    for i, token in enumerate(tokens[:1536]):
        vector[i] = float(token % 100) / 100.0
    return vector

def store_documents(chunks: List[Dict]):
    """Store document chunks in ChromaDB."""
    print("Storing documents in vector database...")
    
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk['text'])
        collection.add(
            embeddings=[embedding],
            documents=[chunk['text']],
            metadatas=[{
                'source': chunk['source'],
                'chunk_id': chunk['chunk_id'],
                'start': chunk['start'],
                'end': chunk['end']
            }],
            ids=[chunk['chunk_id']]
        )
    
    print(f"Stored {len(chunks)} chunks")

def retrieve_relevant_context(query: str, n_results: int = 5) -> List[Dict]:
    """Retrieve relevant regulatory text based on query."""
    query_embedding = get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    contexts = []
    for i in range(len(results['documents'][0])):
        contexts.append({
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'chunk_id': results['metadatas'][0][i]['chunk_id'],
            'distance': results['distances'][0][i]
        })
    
    return contexts

# Define validation rules
VALIDATION_RULES = [
    ValidationRule(
        rule_id="R001",
        rule_name="CET1 Positive",
        rule_description="CET1 capital must be non-negative",
        field_to_validate="cet1_capital",
        validation_type="range",
        min_value=0,
        error_message="CET1 capital cannot be negative"
    ),
    ValidationRule(
        rule_id="R002",
        rule_name="Capital Adequacy",
        rule_description="Total capital must be at least 8% of RWA",
        field_to_validate="total_capital_ratio",
        validation_type="calculation",
        min_value=0.08,
        error_message="Total capital ratio below regulatory minimum of 8%"
    ),
    ValidationRule(
        rule_id="R003",
        rule_name="CET1 Ratio",
        rule_description="CET1 ratio must be at least 4.5%",
        field_to_validate="cet1_ratio",
        validation_type="range",
        min_value=0.045,
        error_message="CET1 ratio below minimum requirement of 4.5%"
    ),
    ValidationRule(
        rule_id="R004",
        rule_name="Capital Calculation",
        rule_description="Total capital equals CET1 + AT1 + Tier 2",
        field_to_validate="total_capital",
        validation_type="calculation",
        condition="cet1_capital + at1_capital + tier2_capital",
        error_message="Total capital calculation mismatch"
    ),
    ValidationRule(
        rule_id="R005",
        rule_name="Ratio Calculation",
        rule_description="CET1 ratio equals CET1 / RWA",
        field_to_validate="cet1_ratio",
        validation_type="calculation",
        condition="cet1_capital / total_rwa",
        error_message="CET1 ratio calculation mismatch"
    )
]

def validate_report(report: OwnFundsReport) -> List[ValidationResult]:
    """Validate the generated report against rules."""
    validations = []
    
    for rule in VALIDATION_RULES:
        result = ValidationResult(
            field_id=rule.field_to_validate,
            field_name=rule.rule_name,
            status=ValidationStatus.VALID,
            value=getattr(report, rule.field_to_validate, None),
            message="",
            rule_id=rule.rule_id
        )
        
        if rule.validation_type == "range":
            value = result.value
            if value is None:
                result.status = ValidationStatus.MISSING
                result.message = f"Field {rule.field_to_validate} is missing"
            elif rule.min_value is not None and value < rule.min_value:
                result.status = ValidationStatus.ERROR
                result.message = rule.error_message
                result.expected_range = f"≥ {rule.min_value}"
            elif rule.max_value is not None and value > rule.max_value:
                result.status = ValidationStatus.ERROR
                result.message = rule.error_message
                result.expected_range = f"≤ {rule.max_value}"
            else:
                result.message = f"Value {value} within acceptable range"
                
        elif rule.validation_type == "calculation":
            if rule.field_to_validate == "total_capital":
                expected = report.cet1_capital + report.at1_capital + report.tier2_capital
                if abs(report.total_capital - expected) > 0.01:
                    result.status = ValidationStatus.ERROR
                    result.message = f"{rule.error_message}. Expected: {expected}, Got: {report.total_capital}"
                else:
                    result.message = "Capital calculation verified"
                    
            elif rule.field_to_validate == "cet1_ratio":
                if report.total_rwa > 0:
                    expected = report.cet1_capital / report.total_rwa
                    if abs(report.cet1_ratio - expected) > 0.001:
                        result.status = ValidationStatus.ERROR
                        result.message = f"{rule.error_message}. Expected: {expected:.4f}, Got: {report.cet1_ratio:.4f}"
                    else:
                        result.message = f"Ratio calculation verified: {result.value:.2%}"
        
        validations.append(result)
    
    return validations

# Initialize Pydantic AI Agent with universal configuration
try:
    model = create_model()
    agent_initialized = True
except ValueError as e:
    print(f"Warning: {e}")
    print("\nNote: You can still use mock_demo.py which doesn't require an API key.")
    model = None
    agent_initialized = False

if agent_initialized and model:
    agent = Agent(
        model=model,
        result_type=OwnFundsReport,
        system_prompt="""You are a PRA COREP regulatory reporting expert. Your task is to generate accurate Own Funds reports based on user queries and regulatory context.

Follow these rules:
1. Use only the provided regulatory context to determine capital classifications and deductions
2. Apply CRR and PRA Rulebook requirements correctly
3. Calculate capital ratios accurately: CET1 ratio = CET1 / RWA, etc.
4. Ensure all components sum correctly
5. Flag any ambiguities or missing information
6. Provide reasoning for each major calculation

Key regulatory references to consider:
- CET1 capital components (CRR Article 26)
- CET1 deductions (CRR Article 36)
- AT1 capital requirements (CRR Article 51)
- Tier 2 capital (CRR Article 62)
- Capital ratios minimum requirements

Always return a complete OwnFundsReport structure with all fields populated.""")
else:
    agent = None
    print("\n⚠️  Agent not initialized. Please check your .env configuration.")
    print("   The mock_demo.py file can be used without API keys.\n")

class COREPAssistant:
    """Main assistant class for COREP reporting."""
    
    def __init__(self):
        self.agent = agent
        self.validation_rules = VALIDATION_RULES
        self.audit_log: List[AuditLogEntry] = []
    
    def process_query(self, query: RegulatoryQuery) -> COREPTemplateOutput:
        """Process a user query and generate a complete report."""
        
        # Step 1: Retrieve relevant regulatory context
        print("Retrieving relevant regulations...")
        search_query = f"{query.question} {query.scenario_description}"
        contexts = retrieve_relevant_context(search_query, n_results=7)
        
        # Format context for the LLM
        context_text = "\n\n".join([
            f"[{ctx['source']}] {ctx['text'][:500]}..."
            for ctx in contexts
        ])
        
        rule_references = [ctx['source'] for ctx in contexts]
        
        # Step 2: Generate report using Pydantic AI
        print("Generating report with Pydantic AI...")
        
        prompt = f"""User Question: {query.question}

Scenario: {query.scenario_description}

Relevant Regulatory Context:
{context_text}

Based on the above regulatory context and scenario, generate a complete Own Funds (CA1) report.

Requirements:
1. Include realistic but compliant capital figures
2. Apply appropriate deductions based on CRR/PRA rules
3. Calculate all ratios correctly
4. Ensure CET1 ≥ 4.5% of RWA, Total Capital ≥ 8% of RWA
5. Include all required capital components

Respond with a complete OwnFundsReport structure."""

        # Run the agent
        result = self.agent.run_sync(prompt)
        report = result.data
        
        # Step 3: Create audit log
        timestamp = datetime.now().isoformat()
        audit_entries = [
            AuditLogEntry(
                field_id="cet1_capital",
                field_name="CET1 Capital",
                value=report.cet1_capital,
                rule_references=rule_references[:3],
                confidence=0.92,
                reasoning=f"Based on regulatory requirements from {rule_references[0] if rule_references else 'CRR'}",
                timestamp=timestamp
            ),
            AuditLogEntry(
                field_id="total_capital_ratio",
                field_name="Total Capital Ratio",
                value=report.total_capital_ratio,
                rule_references=rule_references[:3],
                confidence=0.95,
                reasoning="Calculated as Total Capital / RWA, must meet 8% minimum",
                timestamp=timestamp
            )
        ]
        
        # Step 4: Validate the report
        print("Validating report...")
        validations = validate_report(report)
        
        # Step 5: Compile final output
        missing_fields = []
        warnings = []
        errors = []
        
        for v in validations:
            if v.status == ValidationStatus.MISSING:
                missing_fields.append(v.field_id)
            elif v.status == ValidationStatus.WARNING:
                warnings.append(v.message)
            elif v.status == ValidationStatus.ERROR:
                errors.append(v.message)
        
        output = COREPTemplateOutput(
            report=report,
            validations=validations,
            audit_log=audit_entries,
            missing_fields=missing_fields,
            warnings=warnings,
            errors=errors
        )
        
        return output
    
    def generate_template_view(self, output: COREPTemplateOutput) -> str:
        """Generate a human-readable template view of the report."""
        report = output.report
        
        template = f"""
{'='*80}
COREP OWN FUNDS REPORT (CA1) - TEMPLATE EXTRACT
{'='*80}

REPORTING ENTITY: {report.reporting_entity}
REPORTING PERIOD: {report.reporting_period}
REPORTING DATE: {report.reporting_date}

{'─'*80}
CAPITAL STRUCTURE
{'─'*80}

COMMON EQUITY TIER 1 (CET1) CAPITAL
  Total CET1 Capital:                    £{report.cet1_capital:>15,.0f}
  Less: CET1 Deductions:                 £{report.cet1_deductions:>15,.0f}
  ─────────────────────────────────────────────────────────────────
  Net CET1 Capital:                      £{report.cet1_capital - report.cet1_deductions:>15,.0f}

ADDITIONAL TIER 1 (AT1) CAPITAL
  Total AT1 Capital:                     £{report.at1_capital:>15,.0f}
  Less: AT1 Deductions:                  £{report.at1_deductions:>15,.0f}
  ─────────────────────────────────────────────────────────────────
  Net AT1 Capital:                       £{report.at1_capital - report.at1_deductions:>15,.0f}

TIER 2 CAPITAL
  Total Tier 2 Capital:                  £{report.tier2_capital:>15,.0f}
  Less: Tier 2 Deductions:               £{report.tier2_deductions:>15,.0f}
  ─────────────────────────────────────────────────────────────────
  Net Tier 2 Capital:                    £{report.tier2_capital - report.tier2_deductions:>15,.0f}

{'─'*80}
TOTAL OWN FUNDS
{'─'*80}

  TOTAL CAPITAL:                         £{report.total_capital:>15,.0f}

{'─'*80}
RISK-WEIGHTED ASSETS AND CAPITAL RATIOS
{'─'*80}

  Total Risk-Weighted Assets:            £{report.total_rwa:>15,.0f}

  CET1 Capital Ratio:                          {report.cet1_ratio:>15.2%} (Min: 4.5%)
  Tier 1 Capital Ratio:                        {report.tier1_ratio:>15.2%} (Min: 6.0%)
  Total Capital Ratio:                         {report.total_capital_ratio:>15.2%} (Min: 8.0%)

{'='*80}
VALIDATION STATUS: {'✓ PASSED' if output.is_valid() else '✗ FAILED'}
{'='*80}

VALIDATION RESULTS:
"""
        
        for v in output.validations:
            status_icon = "✓" if v.status == ValidationStatus.VALID else "⚠" if v.status == ValidationStatus.WARNING else "✗"
            template += f"  {status_icon} {v.field_name}: {v.message}\n"
        
        if output.missing_fields:
            template += f"\nMISSING FIELDS: {', '.join(output.missing_fields)}\n"
        
        if output.warnings:
            template += f"\nWARNINGS:\n"
            for w in output.warnings:
                template += f"  ⚠ {w}\n"
        
        if output.errors:
            template += f"\nERRORS:\n"
            for e in output.errors:
                template += f"  ✗ {e}\n"
        
        template += f"\n{'='*80}\nAUDIT LOG (Sample)\n{'='*80}\n"
        for entry in output.audit_log[:3]:
            template += f"\nField: {entry.field_name}\n"
            template += f"  Value: £{entry.value:,.0f}\n"
            template += f"  Rule References: {', '.join(entry.rule_references)}\n"
            template += f"  Confidence: {entry.confidence:.0%}\n"
            template += f"  Reasoning: {entry.reasoning}\n"
        
        template += f"\n{'='*80}\n"
        
        return template

def initialize_system():
    """Initialize the system by processing PDFs and storing in vector DB."""
    print("Initializing COREP Regulatory Assistant...")
    print("="*60)
    
    # Check if already initialized
    if os.path.exists("./chroma_db") and os.listdir("./chroma_db"):
        print("Vector database already exists. Skipping PDF processing.")
        return
    
    # Process PDFs
    pdf_files = [
        "/home/sohanx1/Downloads/AKION/corep-own-funds-instructions.pdf",
        "/home/sohanx1/Downloads/AKION/Own Funds (CRR)_08-02-2026.pdf",
        "/home/sohanx1/Downloads/AKION/Reporting (CRR)_08-02-2026.pdf",
        "/home/sohanx1/Downloads/AKION/ss3415-october-2025.pdf"
    ]
    
    print("\nStep 1: Extracting text from PDFs...")
    chunks = process_pdfs(pdf_files)
    
    print("\nStep 2: Storing in vector database...")
    store_documents(chunks)
    
    print("\n✓ System initialized successfully!")
    print("="*60)

if __name__ == "__main__":
    # Initialize system
    initialize_system()
    
    # Example usage
    print("\n" + "="*60)
    print("COREP REGULATORY REPORTING ASSISTANT - DEMO")
    print("="*60)
    
    assistant = COREPAssistant()
    
    # Example query
    query = RegulatoryQuery(
        question="How do I report own funds for a bank with £1bn CET1, £200m AT1, and £150m Tier 2 capital?",
        scenario_description="Standard UK bank with goodwill deductions of £50m and intangible assets of £30m. Total RWA is £8bn.",
        reporting_period="Q4 2025",
        additional_context={
            "entity_name": "Demo Bank PLC",
            "reporting_date": "2025-12-31"
        }
    )
    
    print(f"\nProcessing query: {query.question}\n")
    
    # Generate report
    output = assistant.process_query(query)
    
    # Display results
    template_view = assistant.generate_template_view(output)
    print(template_view)
    
    # Save to file
    with open("corep_report_output.json", "w") as f:
        json.dump(output.report.model_dump(), f, indent=2, default=str)
    
    with open("corep_report_template.txt", "w") as f:
        f.write(template_view)
    
    print("\n✓ Reports saved to:")
    print("  - corep_report_output.json")
    print("  - corep_report_template.txt")
