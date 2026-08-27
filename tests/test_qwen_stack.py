from agent.financial_agent import _total_debt
from agent.qwen_retrieval import _rank_facts, _simple_value_question


def apple_facts():
    return [
        {"fact_id":"f1","metric":"total_debt","label":"Commercial paper","value":7979.0,"period":"2025","page":34,"statement":"balance_sheet","validated":True,"is_flow_candidate":False},
        {"fact_id":"f2","metric":"total_debt","label":"Term debt","value":12350.0,"period":"2025","page":34,"statement":"balance_sheet","validated":True,"section_context":"Current liabilities","is_flow_candidate":False},
        {"fact_id":"f3","metric":"total_debt","label":"Term debt","value":78328.0,"period":"2025","page":34,"statement":"balance_sheet","validated":True,"section_context":"Non-current liabilities","is_flow_candidate":False},
        {"fact_id":"f4","metric":"total_debt","label":"Proceeds from issuance of term debt, net","value":4481.0,"period":"2025","page":36,"statement":"cash_flow","validated":True,"is_flow_candidate":True},
    ]


def test_apple_total_debt():
    result = _total_debt(apple_facts())
    assert result["answer"] == 98657.0
    assert result["status"] == "derived"
    assert len(result["inputs"]) == 3


def test_debt_flow_is_ranked_below_balance_sheet():
    ranked = _rank_facts("What was total debt?", apple_facts())
    pages = [fact["page"] for fact in ranked]
    assert pages.count(34) >= 3
    assert 36 not in pages or pages.index(34) < pages.index(36)


def test_simple_question_does_not_require_semantic_retrieval():
    assert _simple_value_question("What was total debt?") is True
    assert _simple_value_question("Why did leverage increase?") is False
