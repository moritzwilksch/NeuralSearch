from src.scraping.scrapers import ArticleBodyCollector, ArticleLinkCollector

# scraper = ArticleLinkCollector(base_url="https://ibkr.info/index?title=&type=article")
# scraper.run()

body_collector = ArticleBodyCollector()
body_collector.run()
