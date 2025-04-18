import os
import logging
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime, timedelta
import time
import dateparser

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)

class NewsCrawler:
    def __init__(self, csv_filename="news_data.csv"):
        """
        Initialize the crawler.
        - `csv_filename`: Name of the CSV file to store the crawled data.
        """
        self.csv_filename = csv_filename
        self.visited_urls = set()  # Track visited URLs to avoid duplicates
        self.initialize_csv()
        self.one_month_ago = datetime.now() - timedelta(days=30)  # Articles must be newer than this

    def initialize_csv(self):
        """
        Initialize the CSV file with headers if it doesn't exist.
        """
        if not os.path.exists(self.csv_filename):
            df = pd.DataFrame(columns=["title", "text", "subject", "date", "label"])
            df.to_csv(self.csv_filename, index=False)
            logging.info(f"Initialized CSV file: {self.csv_filename}")

    def load_existing_data(self):
        """
        Load existing data from the CSV file to check for duplicates.
        """
        if os.path.exists(self.csv_filename):
            df = pd.read_csv(self.csv_filename)
            return set(df["title"]), len(df)  # Use title to avoid duplicates
        return set(), 0

    def parse_date(self, date_str):
        """
        Parse date string into datetime object.
        Returns None if parsing fails.
        """
        try:
            # Try to parse the date using dateparser (handles multiple formats)
            parsed_date = dateparser.parse(date_str)
            if parsed_date:
                return parsed_date
            return None
        except Exception as e:
            logging.warning(f"Could not parse date: {date_str} - {str(e)}")
            return None

    def is_recent(self, article_date):
        """
        Check if the article date is within the last month.
        """
        # Skip date checking - accept all articles regardless of date
        return True

    def crawl_website(self, url, label, limit=10):
        """
        Crawl a specific website and collect articles.
        - `url`: URL of the website.
        - `label`: Label for the articles (`1` for fake, `0` for real).
        - `limit`: Maximum number of articles to collect.
        """
        logging.info(f"Crawling website: {url} (Label: {'Fake' if label == 1 else 'Real'})")
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find all article links
            articles = soup.find_all('a', href=True)[:limit*2]  # Get more links as we'll filter by date
            collected_count = 0

            for article in articles:
                if collected_count >= limit:
                    break
                    
                article_url = urljoin(url, article['href'])
                if article_url in self.visited_urls:
                    continue

                try:
                    article_response = requests.get(article_url, timeout=10)
                    article_response.raise_for_status()
                    article_soup = BeautifulSoup(article_response.content, 'html.parser')

                    # Extract title and text
                    title = article_soup.title.string.strip() if article_soup.title else "No Title"
                    paragraphs = [p.get_text(strip=True) for p in article_soup.find_all('p')]
                    # Limit text to maximum 250 lines
                    paragraphs = paragraphs[:100]
                    text = " ".join(paragraphs)

                    # Extract subject
                    subject_tag = article_soup.find('meta', attrs={'name': 'section'}) or \
                                article_soup.find('meta', attrs={'property': 'article:section'})
                    subject = subject_tag['content'] if subject_tag and 'content' in subject_tag.attrs else "General"

                    # Extract date - try multiple methods
                    date = None
                    date_tag = article_soup.find('meta', attrs={'name': 'date'}) or \
                             article_soup.find('meta', attrs={'property': 'article:published_time'}) or \
                             article_soup.find('meta', attrs={'name': 'pubdate'}) or \
                             article_soup.find('meta', attrs={'property': 'og:pubdate'}) or \
                             article_soup.find('time')
                    
                    if date_tag:
                        if 'content' in date_tag.attrs:
                            date_str = date_tag['content']
                        elif 'datetime' in date_tag.attrs:
                            date_str = date_tag['datetime']
                        elif date_tag.text:
                            date_str = date_tag.text
                        else:
                            date_str = None
                            
                        if date_str:
                            date = self.parse_date(date_str)
                    
                    # If no date found in meta tags, use current date as fallback
                    if not date:
                        date = datetime.now()
                        logging.warning(f"No date found for article {article_url}, using current date")

                    # Skip if article is too old
                    if not self.is_recent(date):
                        logging.info(f"Skipping old article from {date}: {title}")
                        continue

                    # Format date for storage
                    formatted_date = date.strftime("%Y-%m-%d")

                    # Skip empty articles
                    if not text.strip():
                        logging.warning(f"Skipping empty article: {article_url}")
                        continue

                    # Add article to CSV
                    self.add_article_to_csv(title, text, subject, formatted_date, label)
                    self.visited_urls.add(article_url)
                    collected_count += 1
                    logging.info(f"Collected article {collected_count}: {title} ({formatted_date})")

                except Exception as e:
                    logging.error(f"Error processing article {article_url}: {str(e)}")

        except Exception as e:
            logging.error(f"Error crawling website {url}: {str(e)}")

    def add_article_to_csv(self, title, text, subject, date, label):
        """
        Add an article to the CSV file.
        - `title`: Title of the article.
        - `text`: Content of the article.
        - `subject`: Category or section of the article.
        - `date`: Publication date.
        - `label`: Label (`0` for fake, `1` for real).
        """
        new_data = {
            "title": [title],
            "text": [text],
            "subject": [subject],
            "date": [date],
            "label": [label]
        }
        df_new = pd.DataFrame(new_data)
        df_new.columns = ['title', 'text', 'subject', 'date','label']
        df_new.to_csv(self.csv_filename, mode='a', header=False, index=False)
        logging.info(f"Added article to CSV: {title}")

    def continuous_crawl(self, real_sites, fake_sites, interval_minutes=10, limit_per_site=5):
        """
        Continuously crawl real and fake news websites at regular intervals.
        - `real_sites`: List of real news website URLs.
        - `fake_sites`: List of fake news website URLs.
        - `interval_minutes`: Time interval between crawls (in minutes).
        - `limit_per_site`: Maximum articles to collect per site per crawl.
        """
        while True:
            logging.info("Starting a new crawl cycle...")
            # Update the one_month_ago timestamp for each new crawl
            self.one_month_ago = datetime.now() - timedelta(days=30)
            
            # Crawl real news sites
            for site in real_sites:
                self.crawl_website(site, label=1, limit=limit_per_site)
            # Crawl fake news sites
            for site in fake_sites:
                self.crawl_website(site, label=0, limit=limit_per_site)
            logging.info(f"Sleeping for {interval_minutes} minutes before the next crawl...")
            time.sleep(interval_minutes * 60)

# Define real and fake news websites
real_sites = [
            "https://www.bbc.com/news",
            "https://www.reuters.com/",
            "https://apnews.com/",
            "https://www.npr.org/sections/news/",
            "https://www.aljazeera.com/",
            "https://www.pbs.org/newshour/",
            "https://www.dw.com/en/top-stories/s-9097",
            "https://www.france24.com/en/",
            "https://www.voanews.com/",
            "https://www.scmp.com/news",
            "https://www.africanews.com/",
            "https://www.japantimes.co.jp/",
            "https://www.haaretz.com/",
            "https://www.thenationalnews.com/",
            "https://www.sbs.com.au/news/",
            "https://www.cbc.ca/news/",
            "https://phys.org/",
            "https://www.scientificamerican.com/",
            "https://www.medicalnewstoday.com/",
            "https://www.space.com/",
            "https://www.climate.gov/news-features",
            "https://www.techcrunch.com/",
            "https://www.euronews.com/green",
            "https://www.straitstimes.com/",
            "https://www.timesofindia.indiatimes.com/",
            "https://www.thehindu.com/",
            "https://www.abc.net.au/news/",
            "https://www.ynetnews.com/",
            "https://www.jpost.com/",
            "https://www.irishtimes.com/",
            "https://www.channelnewsasia.com/",
            "https://www.upi.com/",
            "https://www.newsweek.com/",
            "https://www.csmonitor.com/",
            "https://www.thehill.com/",
            "https://www.politico.com/",
            "https://www.rollcall.com/",
            "https://www.defensenews.com/",
            "https://www.militarytimes.com/",
            "https://www.marinecorpstimes.com/",
            "https://www.airforcetimes.com/",
            "https://www.navytimes.com/",
            "https://www.stripes.com/",
            "https://www.armytimes.com/",
            "https://www.spaceflightnow.com/",
            "https://www.spacepolicyonline.com/",
            "https://www.spacewar.com/",
            "https://www.space-travel.com/",
            "https://www.universetoday.com/",
            "https://www.earthsky.org/",
            "https://www.sciencedaily.com/",
            "https://www.livescience.com/",
            "https://www.nature.com/",
            "https://www.sciencemag.org/",
            "https://www.newscientist.com/",
            "https://www.popularmechanics.com/",
            "https://www.popularscience.com/",
            "https://www.discovermagazine.com/",
            "https://www.smithsonianmag.com/",
            "https://www.nationalgeographic.com/",
            "https://www.history.com/news",
            "https://www.archaeology.org/",
            "https://www.sciencenews.org/",
            "https://www.futurity.org/",
            "https://www.sciencenewsforstudents.org/",
        ]
fake_sites= [
            "https://www.theonion.com/",
            "https://babylonbee.com/",
            "https://worldnewsdailyreport.com/",
            "https://www.thedailymash.co.uk/",
            "https://www.theshovel.com.au/",
            "https://duffelblog.com/",
            "https://www.newsbiscuit.com/",
            "https://www.southendnewsNetwork.com/",
            "https://www.empirenews.net/",
            "https://www.thespoof.com/",
            "https://www.waterfordwhispersnews.com/",
            "https://www.thepoke.co.uk/",
            "https://www.reductress.com/",
            "https://www.clickhole.com/",
            "https://www.borowitzreport.com/",
            "https://www.thehardtimes.net/",
            "https://www.thebeaverton.com/",
            "https://www.theoutandout.com/",
            "https://www.thelastlineofdefense.org/",
            "https://www.beforeitsnews.com/",
            "https://www.infowars.com/",
            "https://www.naturalnews.com/",
            "https://www.newsbusters.org/",
            "https://www.thepoliticalinsider.com/",
            "https://www.wnd.com/",
            "https://www.breitbart.com/",
            "https://www.dailywire.com/",
            "https://www.theblaze.com/",
            "https://www.oann.com/",
            "https://www.globalresearch.ca/",
            "https://www.zerohedge.com/",
            "https://www.activistpost.com/",
            "https://www.collective-evolution.com/",
            "https://www.davidicke.com/",
            "https://www.prisonplanet.com/",
            "https://www.thedailysheeple.com/",
            "https://www.truthdig.com/",
            "https://www.truth-out.org/",
            "https://www.counterpunch.org/",
            "https://www.alternet.org/",
            "https://www.commondreams.org/",
            "https://www.democracynow.org/",
            "https://www.theintercept.com/",
            "https://www.mintpressnews.com/",
            "https://www.truthandaction.org/",
            "https://www.trueactivist.com/",
            "https://www.anonews.co/",
            "https://www.disclose.tv/",
            "https://www.sgtreport.com/",
            "https://www.thefreethoughtproject.com/",
            "https://www.activistpost.com/",
            "https://www.naturalnews.com/",
            "https://www.healthnutnews.com/",
            "https://www.greenmedinfo.com/",
            "https://www.mercola.com/"
        ]

# Initialize the crawler
crawler = NewsCrawler(csv_filename="news_data.csv")

# Start continuous crawling
crawler.continuous_crawl(real_sites, fake_sites, interval_minutes=1, limit_per_site=100)