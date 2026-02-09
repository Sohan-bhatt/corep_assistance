# COREP Regulatory Reporting Assistant

**LLM-assisted PRA COREP reporting prototype using Pydantic AI framework.**

*Assignment Submission: Demonstrates end-to-end regulatory reporting automation for PRA COREP Own Funds (CA1) template.*

---

## Assignment Context

This prototype implements an LLM-assisted regulatory reporting assistant that:
- **Parses natural language queries** about capital reporting scenarios
- **Retrieves relevant regulatory text** from PRA/CRR documents (RAG)
- **Generates structured COREP reports** using Pydantic AI
- **Validates against CRR/PRA rules** (minimum ratios, deductions)
- **Creates audit trails** with rule paragraph references

**Two implementations provided:**

1. **`mock_demo.py`** ← **PRIMARY FOR GRADING** (standalone, no external dependencies)
   - Fully functional without API keys or PDFs
   - Demonstrates all core features
   - Runs immediately for assignment review

2. **`corep_assistant.py`** (full version with RAG + LLM, requires setup)
   - Vector database with regulatory documents
   - Live LLM reasoning with OpenAI/Anthropic/etc.
   - Advanced retrieval capabilities

---

## Features

- ✅ **Natural Language Processing** - Convert scenarios to structured data
- ✅ **Pydantic Schemas** - Type-safe COREP report structures
- ✅ **RAG Implementation** - Document retrieval from regulatory texts
- ✅ **Validation Engine** - Automatic compliance checking
- ✅ **Audit Logging** - Complete traceability with rule references
- ✅ **Template Generation** - Human-readable COREP CA1 format
- ✅ **Universal API Support** - Works with OpenAI, Anthropic, Google, etc.

---

## Quick Start

### Option 1: Mock Demo (Recommended for Assignment Review)

**No setup required - runs immediately:**

```bash
cd /home/sohanx1/corep-assistant
source venv/bin/activate
python mock_demo.py
```

This runs 3 demonstration scenarios showing:
- Standard bank capital reporting
- High capital buffer scenarios
- Borderline compliance detection

**Output files generated:**
- `scenario_1_output.json` - Structured data
- `scenario_1_output.txt` - Human-readable report
- Validation results with pass/fail status
- Audit logs with CRR rule references

### Option 2: Full Version with RAG + LLM (Requires Setup)

**Step 1: Configure Environment**

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY="Your_key_here"
```

**Step 2: Install Dependencies**

```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Step 3: Process Regulatory Documents (First Run Only)**

Place your regulatory PDFs in the project directory and run:

```bash
python corep_assistant.py
```

This extracts text and builds the vector database.

**Step 4: Run the Assistant**

```bash
python corep_assistant.py
```

---

## Regulatory Sources

This prototype references the following publicly available regulatory documents:

### Primary Sources:

1. **Capital Requirements Regulation (CRR) - EU Regulation No 575/2013**
   - Article 25: Definition of own funds
   - Article 26: Common Equity Tier 1 (CET1) capital
   - Article 36: Deductions from CET1
   - Article 51: Additional Tier 1 (AT1) capital
   - Article 62: Tier 2 capital
   - Article 92: Capital ratio requirements
   - *Source: [EUR-Lex](https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/regulatory-reporting/banking/corep-own-funds-instructions.pdf)*

2. **PRA Rulebook - Part: Capital Requirements**
   - Implementation of CRR in UK
   - *Source: [Bank of England](https://www.prarulebook.co.uk/pra-rules/own-funds-and-eligible-liabilities-crr/09-02-2026)*

3. **EBA COREP Reporting Framework**
   - Template CA1: Own Funds
   - Instructions for completion
   - *Source: [EBA](https://www.prarulebook.co.uk/pra-rules/reporting-crr/09-02-2026)*

4. **PRA Supervisory Statement 3/15**
   - Guidance on capital reporting
   - *Source: [PRA Publications](https://www.bankofengland.co.uk/-/media/boe/files/prudential-regulation/supervisory-statement/2025/ss3415-october-2025.pdf)*

### Note on PDFs:
Original PDF documents are **not included** in this submission due to copyright/licensing restrictions. The code demonstrates capability to process them if provided. The `mock_demo.py` version operates without requiring these documents.

---

## Usage Examples

### Basic Usage (Mock Version)

```python
from mock_demo import MockCOREPAssistant
from schemas import RegulatoryQuery

# Initialize
assistant = MockCOREPAssistant()

# Create query
query = RegulatoryQuery(
    question="How do I report own funds for a bank with specific capital amounts?",
    scenario_description="Bank with £1bn CET1, £200m AT1, £150m Tier 2. Total RWA is £8bn with £50m goodwill deductions.",
    reporting_period="Q4 2025",
    additional_context={
        "entity_name": "Demo Bank PLC",
        "reporting_date": "2025-12-31"
    }
)

# Generate report
output = assistant.process_query(query)

# Display results
template = assistant.generate_template_view(output)
print(template)

# Check validation
if output.is_valid():
    print("✓ Report passes all validations")
else:
    print("✗ Report has errors:", output.errors)
```

### Full Version with Live LLM

```python
from corep_assistant import COREPAssistant
from schemas import RegulatoryQuery

# Initialize (loads API key from .env automatically)
assistant = COREPAssistant()

# Create query - same as above
query = RegulatoryQuery(
    question="How do I classify hybrid securities for Tier 1 capital?",
    scenario_description="Bank issued £100m of perpetual subordinated debt with 5.5% coupon, callable after 10 years.",
    reporting_period="Q4 2025"
)

# Process with RAG + LLM
output = assistant.process_query(query)

# Results include regulatory citations from PDFs
print(assistant.generate_template_view(output))
```

---

## Output Format

### Structured JSON Report (`scenario_1_output.json`)

```json
{
  "reporting_entity": "Standard Bank UK PLC",
  "reporting_period": "Q4 2025",
  "reporting_date": "2025-12-31",
  "cet1_capital": 920000000,
  "at1_capital": 200000000,
  "tier2_capital": 150000000,
  "total_capital": 1270000000,
  "total_rwa": 8000000000,
  "cet1_ratio": 0.115,
  "tier1_ratio": 0.14,
  "total_capital_ratio": 0.1588,
  "cet1_components": [...],
  "audit_log": [...]
}
```

### Human-Readable Template (`scenario_1_output.txt`)

```
================================================================================
COREP OWN FUNDS REPORT (CA1) - TEMPLATE EXTRACT
================================================================================

REPORTING ENTITY: Standard Bank UK PLC
REPORTING PERIOD: Q4 2025
REPORTING DATE: 2025-12-31

────────────────────────────────────────────────────────────────────────────────
CAPITAL STRUCTURE
────────────────────────────────────────────────────────────────────────────────

COMMON EQUITY TIER 1 (CET1) CAPITAL
  Gross CET1 Capital:                    £  1,000,000,000
  Less: CET1 Deductions:                 £     80,000,000
  ─────────────────────────────────────────────────────────────────
  Net CET1 Capital:                      £    920,000,000

ADDITIONAL TIER 1 (AT1) CAPITAL
  Total AT1 Capital:                     £    200,000,000
  Net AT1 Capital:                       £    200,000,000

TIER 2 CAPITAL
  Total Tier 2 Capital:                  £    150,000,000
  Net Tier 2 Capital:                    £    150,000,000

────────────────────────────────────────────────────────────────────────────────
TOTAL OWN FUNDS:                         £  1,270,000,000
────────────────────────────────────────────────────────────────────────────────

RISK-WEIGHTED ASSETS AND CAPITAL RATIOS

  Total Risk-Weighted Assets:            £  8,000,000,000

  ┌─────────────────────────────────────────────────────────────┐
  │  CET1 Capital Ratio:                11.50%  (Min: 4.5%)   │
  │  Tier 1 Capital Ratio:              14.00%  (Min: 6.0%)   │
  │  Total Capital Ratio:               15.88%  (Min: 8.0%)   │
  └─────────────────────────────────────────────────────────────┘

================================================================================
VALIDATION STATUS: ✓ PASSED
================================================================================
```

---

## Architecture

```
User Query (Natural Language)
    ↓
RegulatoryQuery (Pydantic Model)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  MOCK VERSION                    │  FULL VERSION            │
│  (mock_demo.py)                  │  (corep_assistant.py)    │
├──────────────────────────────────┼──────────────────────────┤
│  • Regex parsing                 │  • Document Retrieval    │
│  • Rule-based logic              │  • Vector DB (ChromaDB)  │
│  • Static validation             │  • Embeddings            │
│  • No external deps              │  • Live LLM Agent        │
│  • Runs offline                  │  • Dynamic reasoning     │
└──────────────────────────────────┴──────────────────────────┘
    ↓
OwnFundsReport (Pydantic Schema)
    ↓
Validation Engine (5 CRR Rules)
    ↓
COREPTemplateOutput (Final Result)
    ↓
├── JSON Report (machine-readable)
├── Template View (human-readable)
├── Validation Results (pass/fail)
└── Audit Log (rule references)
```

---

## Validation Rules

The system validates reports against CRR requirements:

| Rule ID | Description | Requirement | Validation |
|---------|-------------|-------------|------------|
| R001 | CET1 Positive | CET1 ≥ 0 | ✓ Pass / ✗ Fail |
| R002 | Capital Adequacy | Total Ratio ≥ 8% | ✓ Pass / ✗ Fail |
| R003 | CET1 Ratio | CET1 Ratio ≥ 4.5% | ✓ Pass / ✗ Fail |
| R004 | Tier 1 Ratio | Tier 1 Ratio ≥ 6% | ✓ Pass / ✗ Fail |
| R005 | Calculation | CET1 + AT1 + T2 = Total | ✓ Pass / ✗ Fail |

---

## Project Structure

```
corep-assistant/
│
├── schemas.py              # Pydantic data models
├── extract_pd.py
├── mock_demo.py            # Standalone demo 
├── corep_assistant.py      # Full version with RAG
├── requirements.txt        # Dependencies
├── README.md              # This file
│
├── .env                   
│
├── scenario_1_output.txt  #Human-Readable Template
└── scenario_1_output.*    # Example outputs

```

---

## Configuration

### Environment Variables (.env)

The system uses a universal configuration approach supporting multiple LLM providers:

```bash
# Required
LLM_API_KEY=your-api-key

# Provider Selection
LLM_PROVIDER=openai        # Options: openai, anthropic, google, azure, etc.
LLM_MODEL=gpt-4o-mini      # Model name
LLM_BASE_URL=<optional>    # Custom endpoint (for Azure, local models, etc.)
```

**Supported Providers:**
- OpenAI (`openai`)
- Anthropic Claude (`anthropic`)
- Google Gemini (`google`)
- Azure OpenAI (`azure`)
- Local models via compatible APIs

**To switch providers:** Simply change `LLM_PROVIDER` and `LLM_MODEL` in `.env`.

---

## Dependencies

```
pdfplumber>=0.11.0      # PDF text extraction
pydantic-ai>=0.0.20     # AI agent framework
openai>=1.0.0          # OpenAI API client
chromadb>=0.4.0        # Vector database
tiktoken>=0.5.0        # Token counting
pydantic>=2.0.0        # Data validation
python-dotenv>=1.0.0   # Environment variables
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Assignment Demonstration

### Key Capabilities Demonstrated:

1. **Structured Data Modeling** (`schemas.py`)
   - Pydantic models for regulatory reporting
   - Type safety and validation
   - Complex nested structures

2. **LLM Integration** (`corep_assistant.py`)
   - Pydantic AI agent framework
   - Universal API provider support
   - Environment-based configuration

3. **RAG Implementation** (`extract_pdfs.py` + vector DB)
   - Document chunking
   - Embedding-based retrieval
   - Contextual query answering

4. **Business Logic** (`mock_demo.py`)
   - Scenario parsing
   - Capital calculations
   - Regulatory rule application

5. **Validation & Audit** (both versions)
   - Automated compliance checking
   - Audit trail generation
   - Rule citation tracking

6. **Output Generation** (`generate_template_view`)
   - JSON for systems integration
   - Human-readable reports
   - Professional formatting

---

## 📝 Message from Me(Developer)

1. **Primary File:** `mock_demo.py` contains complete working implementation
2. **No External Dependencies Required:** Mock version runs standalone
3. **Example Outputs:** See `scenario_1_output.txt` for expected results
4. **Full Version Available:** `corep_assistant.py` shows advanced RAG capabilities
5. **Easy to Test:** Run `python mock_demo.py` for immediate demonstration

---

## 🔒 Security

- API keys stored in `.env` (excluded from git via `.gitignore`)
- No hardcoded credentials
- Environment variables for sensitive data
- Universal provider support prevents vendor lock-in

---



