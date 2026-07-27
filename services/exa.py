from exa_py import Exa
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class ExaClient:
	def __init__(self):
		self.client = Exa(os.getenv("EXA_API"))

	def search(self, query):
		result = self.client.search(
			query,
			num_results = 10,
			type = "auto",
			contents = {
			"highlights": True,
			"summary" : { "query" : "key findings and conclusions" }
			}
		)
		return  [{k: v for k, v in vars(r).items() if k != "text"}
    				for r in result.results]
