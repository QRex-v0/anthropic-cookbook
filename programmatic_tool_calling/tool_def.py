# Download from: https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/utils/team_expense_api.py
from team_expense_api import get_custom_budget, get_expenses, get_team_members

 
# Tool definitions for the team expense API
tools = [
    {
        "name": "get_team_members",
        "description": 'Returns a list of team members for a given department. Each team member includes their ID, name, role, level (junior, mid, senior, staff, principal), and contact information. Use this to get a list of people whose expenses you want to analyze. Available departments are: engineering, sales, and marketing.\n\nRETURN FORMAT: Returns a JSON string containing an ARRAY of team member objects (not wrapped in an outer object). Parse with json.loads() to get a list. Example: [{"id": "ENG001", "name": "Alice", ...}, {"id": "ENG002", ...}]',
        "input_schema": {
            "type": "object",
            "properties": {
                "department": {
                    "type": "string",
                    "description": "The department name. Case-insensitive.",
                }
            },
            "required": ["department"],
        },
        "input_examples": [
            {"department": "engineering"},
            {"department": "sales"},
            {"department": "marketing"},
        ],
    },
    {
        "name": "get_expenses",
        "description": "Returns all expense line items for a given employee in a specific quarter. Each expense includes extensive metadata: date, category, description, amount (in USD), currency, status (approved, pending, rejected), receipt URL, approval chain, merchant name and location, payment method, and project codes. An employee may have 20-50+ expense line items per quarter, and each line item contains substantial metadata for audit and compliance purposes. Categories include: 'travel' (flights, trains, rental cars, taxis, parking), 'lodging' (hotels, airbnb), 'meals', 'software', 'equipment', 'conference', 'office', and 'internet'. IMPORTANT: Only expenses with status='approved' should be counted toward budget limits.\n\nRETURN FORMAT: Returns a JSON string containing an ARRAY of expense objects (not wrapped in an outer object with an 'expenses' key). Parse with json.loads() to get a list directly. Example: [{\"expense_id\": \"ENG001_Q3_001\", \"amount\": 1250.50, \"category\": \"travel\", ...}, {...}]",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {
                    "type": "string",
                    "description": "The unique employee identifier",
                },
                "quarter": {
                    "type": "string",
                    "description": "Quarter identifier: 'Q1', 'Q2', 'Q3', or 'Q4'",
                },
            },
            "required": ["employee_id", "quarter"],
        },
        "input_examples": [
            {"employee_id": "ENG001", "quarter": "Q3"},
            {"employee_id": "SAL002", "quarter": "Q1"},
            {"employee_id": "MKT001", "quarter": "Q4"},
        ],
    },
    {
        "name": "get_custom_budget",
        "description": 'Get the custom quarterly travel budget for a specific employee. Most employees have a standard $5,000 quarterly travel budget. However, some employees have custom budget exceptions based on their role requirements. This function checks if a specific employee has a custom budget assigned.\n\nRETURN FORMAT: Returns a JSON string containing a SINGLE OBJECT (not an array). Parse with json.loads() to get a dict. Example: {"user_id": "ENG001", "has_custom_budget": false, "travel_budget": 5000, "reason": "Standard", "currency": "USD"}',
        "input_schema": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "The unique employee identifier",
                }
            },
            "required": ["user_id"],
        },
        "input_examples": [
            {"user_id": "ENG001"},
            {"user_id": "SAL002"},
            {"user_id": "MKT001"},
        ],
    },
]
 
tool_functions = {
    "get_team_members": get_team_members,
    "get_expenses": get_expenses,
    "get_custom_budget": get_custom_budget,
}

import copy
 
ptc_tools = copy.deepcopy(tools)
for tool in ptc_tools:
    tool["allowed_callers"] = ["code_execution_20250825"]  # type: ignore
 
 
# Add the code execution tool
ptc_tools.append(
    {
        "type": "code_execution_20250825",  # type: ignore
        "name": "code_execution",
    }
)
