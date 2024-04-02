PROMPT = """\
Your task is to analyze the TEXT below and determine if it contains information that companies or individuals do not want to disclose outside.

Answer "Yes" when TEXT contains company financials, strategies, operations, sales account, customer or partner information, meeting minutes, and other business sensitive information. 

Answer "Yes" when TEXT contains performance and skill evaluation of a person, areas of improvement, or personal identifiable information.

You first reason step by step about your answer and then answer with "Yes" or "No".

You ALWAYS respond only in the following JSON format: {{"reason": "...", "answer": "..."}}
You only respond with one single JSON response. 

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
JSON response: {{"reason": "The text contains profit numbers of a company", "answer": "Yes"}}
Text: We expanded our sales pipeline, adding Computex and SynWin.
JSON response: {{"reason": "Names of sales accounts are mentioned", "answer": "Yes"}}
Text: John showed strong performance over the last quarter and we recommend his promotion.
JSON response: {{"reason": "The text discusses the performance evaluation of a person named John.", "answer":"Yes"}}
Text: We have seen tremendous user growth of more than 2x in the emerging markets.
JSON response: {{"reason": "The text states the growth of the user base. It does not contain business sensitive data", "answer":"No"}}

Your TEXT to analyse:
TEXT: {text}
JSON response:
"""

PROMPT_BUSINESS_SENSITIVE = """\
Your task is to analyze the TEXT below and determine if it contains sensitive information that companies definitely do not want to disclose outside.

You first reason step by step about your answer and then answer with "Yes" or "No".

Answer "No" if TEXT only talks about user growth or adoption.

You ALWAYS respond only in the following JSON format: {{"reason": "...", "answer": "..."}}
You only respond with one single JSON response. 

Examples:
Text: Operating profit increased, from EUR 7m to 9m compared to the previous reporting period.
JSON response: {{"reason": "The text contains profit numbers of a company", "answer": "Yes"}}
Text: We expanded our sales pipeline, adding Computex and SynWin.
JSON response: {{"reason": "Names of sales accounts are mentioned", "answer": "Yes"}}
Text: Joe Smith, Account Manager, APAC region
JSON response: {{"reason": "The text disclosed the name of an account manager. This is confidential information.", "answer":"Yes"}}
Text: We have seen tremendous user growth of more than 2x in the emerging markets.
JSON response: {{"reason": "The text only states the growth of the user base. It does not contain business sensitive data", "answer":"No"}}

Your TEXT to analyse:
TEXT: {text}
JSON response:
"""

PROMPT_PERSONAL_SENSITIVE = """\
Your task is to analyze the TEXT below and determine if it contains evaluations or personal information of a person.

You first reason step by step about your answer and then answer with "Yes" or "No".

You ALWAYS respond only in the following JSON format: {{"reason": "...", "answer": "..."}}
You only respond with one single JSON response. 

Examples:
Text: John showed strong performance over the last quarter and we recommend his promotion.
JSON response: {{"reason": "The text discusses the performance evaluation of a person named John.", "answer":"Yes"}}
Text: Joe Smith, Account Manager, APAC region
JSON response: {{"reason": "The text disclosed the job title and the name of a person. This is personal information.", "answer":"Yes"}}

Your TEXT to analyse:
TEXT: {text}
JSON response:
"""