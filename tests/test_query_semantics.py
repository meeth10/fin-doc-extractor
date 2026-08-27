from agent.query_semantics import compute_ebitda_change, expense_candidates, normalize_question


def sample_data():
    return {
        "summary": {"source_name": "test.pdf", "metadata": {"currency": "INR", "unit": "crore"}},
        "document": {"pages": []},
        "statement_tables": {
            "income_statement": {"tables": [{
                "page_number": 10,
                "page_number_human": 11,
                "table_title": "Statement of Profit and Loss",
                "source": "test",
                "score": 1.0,
                "validated": True,
                "statement_assignment": "title",
                "table": [
                    ["Particulars", "2025", "2024"],
                    ["Operating income", "12,000", "10,000"],
                    ["Depreciation and amortisation", "500", "400"],
                    ["EBITDA", "2,500", "2,000"],
                    ["Total expenses", "9,000", "7,500"],
                ],
            }]},
            "balance_sheet": {"tables": []},
            "cash_flow": {"tables": []},
        },
    }


def test_operational_income_normalizes_to_operating_income():
    assert normalize_question("What was operational income?") == "What was operating income?"


def test_ebitda_change_uses_two_periods():
    result = compute_ebitda_change("Did EBITDA increase or decrease?", sample_data())
    assert result is not None
    assert result["latest_value"] == 2500.0
    assert result["prior_value"] == 2000.0
    assert result["change"] == 500.0
    assert result["percent_change"] == 25.0
    assert result["formula"] == "latest − prior"


def test_ebitda_percentage_change():
    result = compute_ebitda_change("What was the percentage increase in EBITDA?", sample_data())
    assert result is not None
    assert result["answer"] == 25.0
    assert result["formula"] == "(latest − prior) / prior × 100"


def test_expense_candidate_is_income_statement_aggregate():
    candidates = expense_candidates(sample_data())
    assert len(candidates) == 1
    assert candidates[0]["metric"] == "total_expenses"
    assert candidates[0]["values"] == [9000.0, 7500.0]
    assert candidates[0]["page"] == 11
