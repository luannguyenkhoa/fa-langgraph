from typing import Any, Type, Optional
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool
from langchain_core.runnables.base import RunnableSerializable
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

class SQLInput(BaseModel):
    question: str = Field(description="a user's question in natural language")

class SQLTool(BaseTool):
    name = "sql_query"
    description = "useful for when you need to retrieve data from the database"
    args_schema: Type[BaseModel] = SQLInput

    sql_chain: RunnableSerializable[Any, str]

    def _run(
      self, question: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        "Use the tool."
        print(f"Received question: {question}")
        return self.sql_chain.invoke({ "question": question })

    async def _arun(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("sql_query does not support async")
