import hashlib
from gptcache import Cache
from langchain.cache import GPTCache
from gptcache.adapter.api import init_similar_cache

def get_hashed_name(name):
   return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
   hashed_llm = get_hashed_name(llm)
   init_similar_cache(cache_obj=cache_obj, 
                      data_dir=f"./cache/similar_cache_{hashed_llm}/")

cache = GPTCache(init_gptcache)