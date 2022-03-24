# 🧠 NeuralSearch
> Fixing the search function of my broker's knowledge base

# The Problem
Soooo... my broker has a "Knowledge Base" – a collection of articles detailing their platform, fees, and processes. Many of these articles are highly informative and well written. **The problem: you can't find them** because the search function works... poorly. I took this as an inspiration to scrape all 773 articles and build a small search engine that is able to surface relevant articles for a query.

## Status Quo
The problem with the provided search functionality is that it only matches words in the query against titles. To make matters worse, the words have to match *exactly*, so you better not make any typos. I like to imagine that their search looks something like this:

```python
def search(query: str, articles: list) -> list:
  return [a for a in articles if query in a.title]
```

Now, to be fair, their platform is actually quite good and the main landing page of the brokerage has a great search function. But this example made me realize that there are loads of websites out there with poorly designed search functionality.

| <img width=600 src="https://user-images.githubusercontent.com/58488209/159924491-ad7c0f54-82a3-4230-87af-1842c439bacf.png">| <img width=600 src="https://user-images.githubusercontent.com/58488209/159925114-8c631a50-4930-472a-9bd8-3f3f3885e441.png"> |
|:---:|:---:|
|Loads of articles|The search results for "inactivity fee"|

