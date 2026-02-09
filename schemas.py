"""
Pydantic schemas for COREP Own Funds reporting
Based on CRR and PRA Rulebook requirements
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import date

class CapitalComponentType(str, Enum):
    """Types of capital components."""
    CET1 = "Common Equity Tier 1"
    AT1 = "Additional Tier 1"
    TIER2 = "Tier 2"

class DeductionType(str, Enum):
    """Types of deductions from capital."""
    GOODWILL = "Goodwill"
    INTANGIBLES = "Intangible assets"
    DEFERRED_TAX = "Deferred tax assets"
    PENSION_DEFICIT = "Pension fund deficit"
    INVESTMENTS = "Significant investments"

class ValidationStatus(str, Enum):
    """Validation status for a field."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    MISSING = "missing"

class AuditLogEntry(BaseModel):
    """Audit log entry for each populated field."""
    field_id: str = Field(..., description="Unique identifier for the field")
    field_name: str = Field(..., description="Human-readable field name")
    value: Any = Field(..., description="Populated value")
    rule_references: List[str] = Field(..., description="PRA/CRR rule paragraphs used")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    reasoning: str = Field(..., description="Explanation of how the value was derived")
    timestamp: str = Field(..., description="Timestamp of the decision")

class CapitalComponent(BaseModel):
    """Individual capital component."""
    component_type: CapitalComponentType
    description: str
    amount: float = Field(..., ge=0)
    currency: str = "GBP"
    is_eligible: bool = True
    deductions: List[Dict[str, Any]] = []

class OwnFundsReport(BaseModel):
    """COREP Own Funds (CA1) report structure."""
    reporting_entity: str = Field(..., description="Name of the reporting institution")
    reporting_date: date
    reporting_period: str = Field(..., description="e.g., 'Q4 2025'")
    
    # Common Equity Tier 1 (CET1) Capital
    cet1_capital: float = Field(..., ge=0, description="Total CET1 capital")
    cet1_components: List[CapitalComponent] = []
    cet1_deductions: float = Field(default=0, ge=0)
    
    # Additional Tier 1 (AT1) Capital
    at1_capital: float = Field(..., ge=0, description="Total AT1 capital")
    at1_components: List[CapitalComponent] = []
    at1_deductions: float = Field(default=0, ge=0)
    
    # Tier 2 Capital
    tier2_capital: float = Field(..., ge=0, description="Total Tier 2 capital")
    tier2_components: List[CapitalComponent] = []
    tier2_deductions: float = Field(default=0, ge=0)
    
    # Total Capital
    total_capital: float = Field(..., ge=0, description="Total own funds")
    
    # Risk Weighted Assets
    total_rwa: float = Field(..., ge=0, description="Total risk-weighted assets")
    
    # Capital Ratios
    cet1_ratio: float = Field(..., ge=0, description="CET1 capital ratio")
    tier1_ratio: float = Field(..., ge=0, description="Tier 1 capital ratio")
    total_capital_ratio: float = Field(..., ge=0, description="Total capital ratio")
    
    # Metadata
    prepared_by: str
    review_status: str = "draft"
    
    class Config:
        json_schema_extra = {
            "example": {
                "reporting_entity": "Example Bank Ltd",
                "reporting_date": "2025-12-31",
                "reporting_period": "Q4 2025",
                "cet1_capital": 1000000000,
                "at1_capital": 200000000,
                "tier2_capital": 150000000,
                "total_capital": 1350000000,
                "total_rwa": 8000000000,
                "cet1_ratio": 0.125,
                "tier1_ratio": 0.15,
                "total_capital_ratio": 0.16875
            }
        }

class ValidationRule(BaseModel):
    """Validation rule for COREP reporting."""
    rule_id: str
    rule_name: str
    rule_description: str
    field_to_validate: str
    validation_type: str  # "range", "presence", "calculation", "consistency"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    condition: Optional[str] = None
    error_message: str

class ValidationResult(BaseModel):
    """Result of validating a field."""
    field_id: str
    field_name: str
    status: ValidationStatus
    value: Any
    expected_range: Optional[str] = None
    message: str
    rule_id: str

class COREPTemplateOutput(BaseModel):
    """Complete output including report, validation, and audit."""
    report: OwnFundsReport
    validations: List[ValidationResult] = []
    audit_log: List[AuditLogEntry] = []
    missing_fields: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []
    
    def is_valid(self) -> bool:
        """Check if the report passes all validations."""
        return all(v.status in [ValidationStatus.VALID, ValidationStatus.WARNING] 
                  for v in self.validations) and len(self.errors) == 0

class RegulatoryQuery(BaseModel):
    """User query for the regulatory assistant."""
    question: str = Field(..., description="Natural language question about reporting")
    scenario_description: str = Field(..., description="Description of the reporting scenario")
    reporting_period: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
