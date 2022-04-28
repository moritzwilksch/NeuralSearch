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

# The Solution
## TFIDF-based sub-word similarity
A simple solution based on similarity matching of the TFIDF representation of subwords in the articles yields quite an improvement over the status quo:

**Results for query: "inactivity fee"**

*Does IBKR provide for a dormant or inactive account status?*  
- While there is no provision for dormant or inactive account status, there is no monthly minimum activity requirement or inactivity fee in your IBKR account. While we have no minimum account balance, [...]

*Overview of Fees*  
- Clients and as well as prospective clients are encouraged to review our website where fees are outlined in detail. [...]

## Encoder Model
Use `SentenceTransformer` (multi-qa-MiniLM-L6-cos-v1) to embed each sentence into latetent space. Score each sentence's similarity to the query embedding. The similarity score for a document is calculated as the mean over all sentence embeddings in the document.

**Results for query: "inactivity fee"**

*How do I change the fees which I charge my clients?*  
-  Overview: Advisor accounts have the ability to configure the fees which they assess to their clients. [...]

*Options Regulatory Fee (ORF)*
- The ORF is an exchange fee which OCC collects from its clearing members, including IBKR. [...]

## Elastic Search

**Results for query: "inactivity fee"**

*Does IBKR provide for a dormant or inactive account status?*  
- While there is no provision for dormant or inactive account status, there is no monthly minimum activity requirement or inactivity fee in your IBKR account. While we have no minimum account balance, [...]

*Overview of Fees*  
- Clients and as well as prospective clients are encouraged to review our website where fees are outlined in detail. [...]