"""
Mock implementation for rapid prototyping without API keys
Demonstrates the full pipeline with synthetic data
"""

import json
from datetime import datetime, date
from typing import List, Dict, Any
from schemas import (
    RegulatoryQuery, OwnFundsReport, CapitalComponent, CapitalComponentType,
    COREPTemplateOutput, ValidationResult, ValidationStatus, AuditLogEntry,
    ValidationRule
)

# Mock validation rules (same as main)
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
        validation_type="range",
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
        rule_name="Tier1 Ratio",
        rule_description="Tier 1 ratio must be at least 6%",
        field_to_validate="tier1_ratio",
        validation_type="range",
        min_value=0.06,
        error_message="Tier 1 ratio below minimum requirement of 6%"
    ),
]

class MockCOREPAssistant:
    """Mock assistant for demonstration without API calls."""
    
    def __init__(self):
        self.validation_rules = VALIDATION_RULES
        self.mock_contexts = [
            {
                'source': 'CRR_Article_26',
                'text': 'Common Equity Tier 1 capital consists of capital instruments, share premium accounts, retained earnings and other reserves...'
            },
            {
                'source': 'CRR_Article_36',
                'text': 'Deductions from Common Equity Tier 1 items shall include: (a) intangible assets; (b) deferred tax assets...'
            },
            {
                'source': 'CRR_Article_51',
                'text': 'Additional Tier 1 capital consists of instruments, the principal amount of which is perpetual...'
            },
            {
                'source': 'CRR_Article_62',
                'text': 'Tier 2 capital consists of instruments, the principal amount of which has an original maturity of at least five years...'
            },
            {
                'source': 'COREP_Instructions',
                'text': 'Template CA1 - Own Funds: Institutions shall report the composition of their own funds and capital ratios...'
            }
        ]
    
    def parse_scenario(self, scenario_text: str) -> Dict[str, float]:
        """Extract capital figures from scenario description."""
        import re
        
        figures = {}
        
        # Look for monetary amounts (e.g., £1bn, £200m, £50 million)
        patterns = [
            (r'(\d+(?:\.\d+)?)\s*(bn|billion)', 1e9),
            (r'(\d+(?:\.\d+)?)\s*(mn|m|million)', 1e6),
            (r'£(\d+(?:\.\d+)?)\s*(bn|billion)', 1e9),
            (r'£(\d+(?:\.\d+)?)\s*(mn|m|million)', 1e6),
        ]
        
        # Extract CET1
        cet1_match = re.search(r'CET1.*?£?(\d+(?:\.\d+)?)\s*(?:bn|billion|mn|m|million)', scenario_text, re.IGNORECASE)
        if cet1_match:
            val = float(cet1_match.group(1))
            if 'bn' in scenario_text[cet1_match.start():cet1_match.end()].lower() or 'billion' in scenario_text[cet1_match.start():cet1_match.end()].lower():
                figures['cet1'] = val * 1e9
            else:
                figures['cet1'] = val * 1e6
        
        # Extract AT1
        at1_match = re.search(r'AT1.*?£?(\d+(?:\.\d+)?)\s*(?:bn|billion|mn|m|million)', scenario_text, re.IGNORECASE)
        if at1_match:
            val = float(at1_match.group(1))
            if 'bn' in scenario_text[at1_match.start():at1_match.end()].lower():
                figures['at1'] = val * 1e9
            else:
                figures['at1'] = val * 1e6
        
        # Extract Tier 2
        tier2_match = re.search(r'Tier\s*2.*?£?(\d+(?:\.\d+)?)\s*(?:bn|billion|mn|m|million)', scenario_text, re.IGNORECASE)
        if tier2_match:
            val = float(tier2_match.group(1))
            if 'bn' in scenario_text[tier2_match.start():tier2_match.end()].lower():
                figures['tier2'] = val * 1e9
            else:
                figures['tier2'] = val * 1e6
        
        # Extract RWA
        rwa_match = re.search(r'RWA.*?£?(\d+(?:\.\d+)?)\s*(?:bn|billion|mn|m|million)', scenario_text, re.IGNORECASE)
        if rwa_match:
            val = float(rwa_match.group(1))
            if 'bn' in scenario_text[rwa_match.start():rwa_match.end()].lower():
                figures['rwa'] = val * 1e9
            else:
                figures['rwa'] = val * 1e6
        
        # Extract deductions
        goodwill_match = re.search(r'goodwill.*?£?(\d+(?:\.\d+)?)\s*(?:mn|m|million)', scenario_text, re.IGNORECASE)
        if goodwill_match:
            figures['goodwill_deduction'] = float(goodwill_match.group(1)) * 1e6
        
        intangibles_match = re.search(r'intangible.*?£?(\d+(?:\.\d+)?)\s*(?:mn|m|million)', scenario_text, re.IGNORECASE)
        if intangibles_match:
            figures['intangibles_deduction'] = float(intangibles_match.group(1)) * 1e6
        
        return figures
    
    def generate_report(self, query: RegulatoryQuery) -> OwnFundsReport:
        """Generate a report based on parsed scenario."""
        figures = self.parse_scenario(query.scenario_description)
        
        # Default values if parsing fails
        cet1 = figures.get('cet1', 1_000_000_000)
        at1 = figures.get('at1', 200_000_000)
        tier2 = figures.get('tier2', 150_000_000)
        rwa = figures.get('rwa', 8_000_000_000)
        goodwill = figures.get('goodwill_deduction', 50_000_000)
        intangibles = figures.get('intangibles_deduction', 30_000_000)
        
        # Calculate net amounts after deductions
        cet1_deductions = goodwill + intangibles
        cet1_net = max(0, cet1 - cet1_deductions)
        
        total_capital = cet1_net + at1 + tier2
        
        # Calculate ratios
        cet1_ratio = cet1_net / rwa if rwa > 0 else 0
        tier1_ratio = (cet1_net + at1) / rwa if rwa > 0 else 0
        total_capital_ratio = total_capital / rwa if rwa > 0 else 0
        
        # Create capital components
        cet1_components = [
            CapitalComponent(
                component_type=CapitalComponentType.CET1,
                description="Common shares and share premium",
                amount=cet1 * 0.6,
                is_eligible=True
            ),
            CapitalComponent(
                component_type=CapitalComponentType.CET1,
                description="Retained earnings and reserves",
                amount=cet1 * 0.4,
                is_eligible=True
            )
        ]
        
        at1_components = [
            CapitalComponent(
                component_type=CapitalComponentType.AT1,
                description="Additional Tier 1 instruments",
                amount=at1,
                is_eligible=True
            )
        ]
        
        tier2_components = [
            CapitalComponent(
                component_type=CapitalComponentType.TIER2,
                description="Tier 2 capital instruments",
                amount=tier2,
                is_eligible=True
            )
        ]
        
        # Create report
        entity_name = query.additional_context.get('entity_name', 'Demo Bank PLC') if query.additional_context else 'Demo Bank PLC'
        reporting_date = query.additional_context.get('reporting_date', '2025-12-31') if query.additional_context else '2025-12-31'
        
        report = OwnFundsReport(
            reporting_entity=entity_name,
            reporting_date=date.fromisoformat(str(reporting_date)),
            reporting_period=query.reporting_period or 'Q4 2025',
            cet1_capital=cet1_net,
            cet1_components=cet1_components,
            cet1_deductions=cet1_deductions,
            at1_capital=at1,
            at1_components=at1_components,
            at1_deductions=0,
            tier2_capital=tier2,
            tier2_components=tier2_components,
            tier2_deductions=0,
            total_capital=total_capital,
            total_rwa=rwa,
            cet1_ratio=round(cet1_ratio, 4),
            tier1_ratio=round(tier1_ratio, 4),
            total_capital_ratio=round(total_capital_ratio, 4),
            prepared_by="COREP Assistant",
            review_status="draft"
        )
        
        return report
    
    def validate_report(self, report: OwnFundsReport) -> List[ValidationResult]:
        """Validate the report against rules."""
        validations = []
        
        for rule in self.validation_rules:
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
                    result.expected_range = f"≥ {rule.min_value:.1%}"
                elif rule.max_value is not None and value > rule.max_value:
                    result.status = ValidationStatus.ERROR
                    result.message = rule.error_message
                    result.expected_range = f"≤ {rule.max_value:.1%}"
                else:
                    if rule.min_value:
                        result.message = f"Value {value:.2%} meets minimum {rule.min_value:.1%}"
                    else:
                        result.message = f"Value {value:,.0f} is valid"
                        
            validations.append(result)
        
        return validations
    
    def process_query(self, query: RegulatoryQuery) -> COREPTemplateOutput:
        """Process query and generate complete output."""
        
        # Generate report
        report = self.generate_report(query)
        
        # Create audit log
        timestamp = datetime.now().isoformat()
        audit_entries = [
            AuditLogEntry(
                field_id="cet1_capital",
                field_name="CET1 Capital (net of deductions)",
                value=report.cet1_capital,
                rule_references=["CRR Article 26", "CRR Article 36"],
                confidence=0.95,
                reasoning=f"Calculated as gross CET1 (£{report.cet1_capital + report.cet1_deductions:,.0f}) less deductions for goodwill (£{report.cet1_deductions:,.0f}) per CRR Article 36",
                timestamp=timestamp
            ),
            AuditLogEntry(
                field_id="total_capital",
                field_name="Total Own Funds",
                value=report.total_capital,
                rule_references=["CRR Article 25"],
                confidence=0.98,
                reasoning=f"Sum of CET1 (£{report.cet1_capital:,.0f}), AT1 (£{report.at1_capital:,.0f}), and Tier 2 (£{report.tier2_capital:,.0f})",
                timestamp=timestamp
            ),
            AuditLogEntry(
                field_id="cet1_ratio",
                field_name="CET1 Capital Ratio",
                value=report.cet1_ratio,
                rule_references=["CRR Article 92"],
                confidence=0.97,
                reasoning=f"CET1 (£{report.cet1_capital:,.0f}) divided by Total RWA (£{report.total_rwa:,.0f}) = {report.cet1_ratio:.2%}. Minimum requirement: 4.5%",
                timestamp=timestamp
            ),
            AuditLogEntry(
                field_id="total_capital_ratio",
                field_name="Total Capital Ratio",
                value=report.total_capital_ratio,
                rule_references=["CRR Article 92", "COREP CA1 Instructions"],
                confidence=0.98,
                reasoning=f"Total Capital (£{report.total_capital:,.0f}) divided by Total RWA (£{report.total_rwa:,.0f}) = {report.total_capital_ratio:.2%}. Minimum requirement: 8%",
                timestamp=timestamp
            )
        ]
        
        # Validate
        validations = self.validate_report(report)
        
        # Compile output
        missing_fields = [v.field_id for v in validations if v.status == ValidationStatus.MISSING]
        warnings = [v.message for v in validations if v.status == ValidationStatus.WARNING]
        errors = [v.message for v in validations if v.status == ValidationStatus.ERROR]
        
        return COREPTemplateOutput(
            report=report,
            validations=validations,
            audit_log=audit_entries,
            missing_fields=missing_fields,
            warnings=warnings,
            errors=errors
        )
    
    def generate_template_view(self, output: COREPTemplateOutput) -> str:
        """Generate human-readable template view."""
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
  Gross CET1 Capital:                    £{report.cet1_capital + report.cet1_deductions:>15,.0f}
  Less: CET1 Deductions:                 £{report.cet1_deductions:>15,.0f}
  ─────────────────────────────────────────────────────────────────
  Net CET1 Capital:                      £{report.cet1_capital:>15,.0f}

  Components:
"""
        for comp in report.cet1_components:
            template += f"    • {comp.description}: £{comp.amount:>15,.0f}\n"
        
        template += f"""
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

  ┌─────────────────────────────────────────────────────────────┐
  │  CET1 Capital Ratio:              {report.cet1_ratio:>8.2%}  (Min: 4.5%)   │
  │  Tier 1 Capital Ratio:            {report.tier1_ratio:>8.2%}  (Min: 6.0%)   │
  │  Total Capital Ratio:             {report.total_capital_ratio:>8.2%}  (Min: 8.0%)   │
  └─────────────────────────────────────────────────────────────┘

{'='*80}
VALIDATION STATUS: {'✓ PASSED' if output.is_valid() else '✗ FAILED'}
{'='*80}

VALIDATION RESULTS:
"""
        
        for v in output.validations:
            if v.status == ValidationStatus.VALID:
                icon = "✓"
                status_str = "PASS"
            elif v.status == ValidationStatus.WARNING:
                icon = "⚠"
                status_str = "WARN"
            elif v.status == ValidationStatus.ERROR:
                icon = "✗"
                status_str = "FAIL"
            else:
                icon = "?"
                status_str = "MISS"
            
            template += f"  {icon} [{status_str}] {v.field_name}\n"
            template += f"      {v.message}\n"
        
        if output.missing_fields:
            template += f"\n⚠ MISSING FIELDS: {', '.join(output.missing_fields)}\n"
        
        if output.warnings:
            template += f"\n⚠ WARNINGS:\n"
            for w in output.warnings:
                template += f"    {w}\n"
        
        if output.errors:
            template += f"\n✗ ERRORS:\n"
            for e in output.errors:
                template += f"    {e}\n"
        
        template += f"\n{'='*80}\nAUDIT LOG\n{'='*80}\n"
        for entry in output.audit_log:
            template += f"\n{'─'*80}\n"
            template += f"Field: {entry.field_name} (ID: {entry.field_id})\n"
            template += f"  Value: £{entry.value:,.2f}\n" if isinstance(entry.value, (int, float)) else f"  Value: {entry.value}\n"
            template += f"  Rule References: {', '.join(entry.rule_references)}\n"
            template += f"  Confidence: {entry.confidence:.0%}\n"
            template += f"  Timestamp: {entry.timestamp}\n"
            template += f"  Reasoning:\n"
            # Wrap reasoning text
            words = entry.reasoning.split()
            line = "    "
            for word in words:
                if len(line) + len(word) + 1 > 78:
                    template += line + "\n"
                    line = "    " + word
                else:
                    line += " " + word if line.strip() else word
            if line.strip():
                template += line + "\n"
        
        template += f"\n{'='*80}\n"
        
        return template

def run_demo():
    """Run a demonstration of the mock assistant."""
    print("="*80)
    print("  COREP REGULATORY REPORTING ASSISTANT - MOCK DEMO")
    print("  PRA/CRR Own Funds (CA1) Template - No API Key Required")
    print("="*80)
    
    assistant = MockCOREPAssistant()
    
    # Example scenarios
    scenarios = [
        {
            "name": "Standard UK Bank",
            "question": "How do I report own funds for a bank with £1bn CET1, £200m AT1, and £150m Tier 2 capital?",
            "scenario": "Standard UK bank with goodwill deductions of £50m and intangible assets of £30m. Total RWA is £8bn.",
            "entity": "Standard Bank UK PLC",
            "date": "2025-12-31"
        },
        {
            "name": "High Capital Buffer Bank",
            "question": "What if the bank has higher capital buffers?",
            "scenario": "Bank with £2bn CET1, £500m AT1, £300m Tier 2, and £10bn RWA. No significant deductions.",
            "entity": "Capital Rich Bank Ltd",
            "date": "2025-12-31"
        },
        {
            "name": "Borderline Compliance",
            "question": "How to report a bank close to minimum requirements?",
            "scenario": "Bank with £350m CET1, £50m AT1, £100m Tier 2, and £8bn RWA with £20m goodwill deductions.",
            "entity": "Margin Bank PLC",
            "date": "2025-12-31"
        }
    ]
    
    # Process each scenario
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─'*80}")
        print(f"SCENARIO {i}: {scenario['name']}")
        print(f"{'─'*80}")
        
        query = RegulatoryQuery(
            question=scenario['question'],
            scenario_description=scenario['scenario'],
            reporting_period="Q4 2025",
            additional_context={
                "entity_name": scenario['entity'],
                "reporting_date": scenario['date']
            }
        )
        
        print(f"\n📝 Query: {query.question}")
        print(f"📋 Scenario: {query.scenario_description}\n")
        
        # Process query
        output = assistant.process_query(query)
        
        # Display template
        template = assistant.generate_template_view(output)
        print(template)
        
        # Save output
        filename_base = f"scenario_{i}_output"
        
        with open(f"{filename_base}.json", "w") as f:
            json.dump(output.report.model_dump(), f, indent=2, default=str)
        
        with open(f"{filename_base}.txt", "w") as f:
            f.write(template)
        
        print(f"\n💾 Saved to: {filename_base}.json and {filename_base}.txt")
    
    print("\n" + "="*80)
    print("✅ Mock Demo completed!")
    print("="*80)
    print("\nKey Features Demonstrated:")
    print("  ✓ Natural language scenario parsing")
    print("  ✓ Structured report generation with Pydantic schemas")
    print("  ✓ Automatic validation against CRR/PRA rules")
    print("  ✓ Audit logging with rule paragraph references")
    print("  ✓ Human-readable COREP template output")
    print("  ✓ Validation pass/fail with detailed error messages")
    print("\n📁 Output files saved in current directory")
    print("\nNote: This is a mock implementation for demonstration.")
    print("For full functionality with RAG and LLM reasoning, use corep_assistant.py")
    print("with a valid OPENAI_API_KEY environment variable.")

if __name__ == "__main__":
    run_demo()
