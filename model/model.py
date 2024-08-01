from os.path import exists
from langchain_openai import ChatOpenAI
from langchain_community.llms.llamacpp import LlamaCpp

# For general purposes of Generative AI
chat_model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=512, temperature=0.5, streaming=True)

# For minimized SQL query result without creative tokens
# non_temp_model = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=512, temperature=0, streaming=True)

# For higher accuracy of SQL query generation without creative tokens
sql_llm = ChatOpenAI(model="gpt-4-0125-preview", max_tokens=512, temperature=0, streaming=True)

# https://huggingface.co/MaziyarPanahi/sqlcoder-7b-2-GGUF/blob/main/sqlcoder-7b-2.Q4_K_M.gguf
# Download the quantized model then place in the path ./gguf if wanna use the local model for SQL queries generation.
if exists("../gguf/sqlcoder-7b-2.Q4_K_M.gguf"):
    sql_llm = LlamaCpp(
        model_path="../gguf/sqlcoder-7b-2.Q4_K_M.gguf",
        cache=True,
        temperature=0,
        max_tokens=512,
        n_ctx=1024,
        n_threads=4,
        f16_kv=True,
        model_kwargs={
            "num_beams": 4,
            "do_sample": False,
            "num_return_sequences": 1,
        }
    )
