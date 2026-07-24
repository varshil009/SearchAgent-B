from exa_py import Exa
import os

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
		print()
		return [vars(r) for r in result.results]
