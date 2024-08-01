SQL_PROMPT="""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- NEVER self-assuming any value, instead, you should join tables to get relevant data or key constraints.
- Remember that users' financial figures are located in the `expense` table entirely.
- Eventually, if you cannot answer the question with the available database schema, return 'I do not know'.

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
SQL Query: [SQL]
"""
SQL_RESPONSE_QUERY="""
### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Instructions
- NEVER self-assuming any value, instead, you should join tables to get relevant data or key constraints.
- Remember that users' financial figures are located in the `expense` table entirely.
- Eventually, if you cannot answer the question with the available database schema, return 'I do not know'.

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
SQL Query: {query}
SQL Response: {response}
"""

POSTGRES_PROMPT = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 10 results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: "Question here"
SQLQuery: "SQL Query to run"
SQLResult: "Result of the SQLQuery"

Only use the following tables:
{table_info}

Question: {question}

{agent_scratchpad}
"""

TEAM_SUPERVIOR_PROMPT="""You are a supervisor tasked with managing a conversation between the following workers:  {team_members}.
Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status.
When finished, respond with FINISH.
"""

TOP_SUPERVIOR_PROMPT="""You are a supervisor tasked with managing a conversation between the following workers:  {team_members}.
Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status.
You should try to answer the question by yourself before routing the question to a worker.
When finished, respond with FINISH.
"""