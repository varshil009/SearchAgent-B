import requests

class WikiClient:
    def __init__(self):
        self.url = "https://en.wikipedia.org/w/api.php"
        self.headers = {"User-Agent" : "SearchAgentApp/1.0 (lmao99@gmail.com)"}

    def search(self, query):
        # get titles
        r = requests.get(
                        self.url, 
                        headers=self.headers,
                        params=self.params_get_pages(query)
                    )
        articles = []
        for x in r.json()["query"]["search"][:3]:
            pageid = x["pageid"]
            response = requests.get(
                                    self.url, 
                                    headers=self.headers, 
                                    params=self.params_get_articles(pageid)
                                )
            data = response.json()
            articles.append(data["query"]["pages"][str(pageid)]["extract"])
            #print(data["query"]["pages"][str(pageid)]["extract"], end="\n_____________________________________________________________________________\n")
        return articles

    def params_get_articles(self, pageids):
        return {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "pageids": pageids
            }

    def params_get_pages(self, query):
        return {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json"
            }
