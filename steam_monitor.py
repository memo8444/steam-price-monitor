# Steam Market Price Monitor Bot
import asyncio
import aiohttp
import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64
from dataclasses import dataclass
from telegram import Bot
from telegram.error import TelegramError
import sqlite3
from bs4 import BeautifulSoup
import time
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class MarketItem:
    name: str
    url: str
    current_price: float
    price_history: List[tuple]  # [(timestamp, price), ...]
    image_url: str
    market_hash_name: str

@dataclass
class PriceAlert:
    item: MarketItem
    price_change_percent: float
    old_price: float
    new_price: float
    chart_image: bytes

class SteamMarketScraper:
    def __init__(self):
        self.base_url = "https://steamcommunity.com"
        self.market_url = "https://steamcommunity.com/market/search"
        self.session = None
        
    async def init_session(self):
        """Initialize aiohttp session with proper headers"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=3)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            headers=headers, 
            connector=connector,
            timeout=timeout
        )
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def get_market_items(self, max_items: int = 100) -> List[Dict]:
        """Scrape items from Steam Market search page"""
        params = {
            'descriptions': '1',
            'category_730_ItemSet[0]': 'any',
            'category_730_Weapon[0]': 'any',
            'category_730_Quality[0]': 'any',
            'category_730_ProPlayer[0]': 'any',
            'category_730_StickerCapsule[0]': 'any',
            'appid': '730',
            'q': '',
            'count': str(min(max_items, 100)),
            'start': '0'
        }
        
        try:
            async with self.session.get(self.market_url, params=params) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch market page: {response.status}")
                    return []
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                # Find the script tag containing market data
                script_tags = soup.find_all('script')
                market_data = None
                
                for script in script_tags:
                    if script.string and 'g_rgListingInfo' in script.string:
                        # Extract JSON data from JavaScript
                        script_content = script.string
                        # Parse the market data (this is simplified - actual parsing would be more complex)
                        break
                
                # Alternative: Parse HTML directly
                items = []
                market_listing_rows = soup.find_all('a', class_='market_listing_row_link')
                
                for row in market_listing_rows[:max_items]:
                    item_data = await self._parse_item_row(row)
                    if item_data:
                        items.append(item_data)
                
                return items
                
        except Exception as e:
            logger.error(f"Error scraping market items: {e}")
            return []
    
    async def _parse_item_row(self, row_element) -> Optional[Dict]:
        """Parse individual item row from market listing"""
        try:
            # Extract item name
            name_elem = row_element.find('span', class_='market_listing_item_name')
            if not name_elem:
                return None
            
            item_name = name_elem.get_text(strip=True)
            
            # Extract price
            price_elem = row_element.find('span', class_='normal_price')
            if not price_elem:
                price_elem = row_element.find('span', class_='sale_price')
            
            if not price_elem:
                return None
            
            price_text = price_elem.get_text(strip=True)
            # Parse price (remove currency symbols, convert to float)
            price = self._parse_price(price_text)
            
            # Extract image URL
            img_elem = row_element.find('img', class_='market_listing_item_img')
            image_url = img_elem.get('src') if img_elem else ''
            
            # Extract market hash name from URL
            item_url = row_element.get('href', '')
            market_hash_name = self._extract_market_hash_name(item_url)
            
            return {
                'name': item_name,
                'current_price': price,
                'image_url': image_url,
                'url': f"{self.base_url}{item_url}",
                'market_hash_name': market_hash_name
            }
            
        except Exception as e:
            logger.error(f"Error parsing item row: {e}")
            return None
    
    def _parse_price(self, price_text: str) -> float:
        """Parse price string to float"""
        try:
            # Remove currency symbols and whitespace
            price_clean = price_text.replace('$', '').replace('€', '').replace('£', '').strip()
            return float(price_clean)
        except:
            return 0.0
    
    def _extract_market_hash_name(self, url: str) -> str:
        """Extract market hash name from item URL"""
        try:
            # URL format: /market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29
            parts = url.split('/')
            if len(parts) >= 4:
                return parts[-1].replace('%20', ' ').replace('%7C', '|')
            return ''
        except:
            return ''
    
    async def get_price_history(self, market_hash_name: str, days: int = 30) -> List[tuple]:
        """Get price history for an item"""
        url = f"{self.base_url}/market/pricehistory/"
        
        # Properly encode the market hash name
        import urllib.parse
        encoded_name = urllib.parse.quote(market_hash_name)
        
        params = {
            'appid': '730',
            'market_hash_name': encoded_name
        }
        
        # Add additional headers to appear more legitimate
        headers = {
            'Referer': f'{self.base_url}/market/listings/730/{encoded_name}',
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json, text/plain, */*',
        }
        
        try:
            # Longer delay to avoid rate limiting
            await asyncio.sleep(random.uniform(3, 8))
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    # Rate limited - wait longer
                    logger.warning(f"Rate limited for {market_hash_name}, waiting 30 seconds...")
                    await asyncio.sleep(30)
                    return await self.get_price_history(market_hash_name, days)
                
                if response.status != 200:
                    logger.warning(f"Failed to get price history for {market_hash_name}: {response.status}")
                    # Return empty list instead of failing completely
                    return []
                
                try:
                    data = await response.json()
                except:
                    logger.warning(f"Invalid JSON response for {market_hash_name}")
                    return []
                
                if not data.get('success'):
                    logger.info(f"No price history available for {market_hash_name}")
                    return []
                
                prices = data.get('prices', [])
                
                # Convert to list of tuples (timestamp, price)
                price_history = []
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for price_entry in prices:
                    try:
                        # Format: ["Feb 01 2024 01: +0", 1.23, "45"]
                        if len(price_entry) >= 2:
                            date_str = price_entry[0]
                            price = float(price_entry[1])
                            
                            # Parse date - handle different formats
                            date_obj = None
                            try:
                                # Try different date formats
                                if len(date_str) >= 11:
                                    date_obj = datetime.strptime(date_str[:11], "%b %d %Y")
                                elif len(date_str) >= 6:
                                    date_obj = datetime.strptime(date_str[:6], "%b %d")
                                    date_obj = date_obj.replace(year=datetime.now().year)
                            except:
                                continue
                            
                            if date_obj and date_obj >= cutoff_date:
                                price_history.append((date_obj, price))
                    except:
                        continue
                
                return sorted(price_history, key=lambda x: x[0])
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout getting price history for {market_hash_name}")
            return []
        except Exception as e:
            logger.warning(f"Error getting price history for {market_hash_name}: {e}")
            return []

class PriceAnalyzer:
    @staticmethod
    def calculate_price_change(price_history: List[tuple], hours: int = 24) -> Optional[float]:
        """Calculate price change percentage over specified hours"""
        if len(price_history) < 2:
            return None
        
        now = datetime.now()
        cutoff_time = now - timedelta(hours=hours)
        
        # Find the most recent price
        current_price = price_history[-1][1]
        
        # Find price from 24 hours ago (or closest)
        old_price = None
        for timestamp, price in reversed(price_history):
            if timestamp <= cutoff_time:
                old_price = price
                break
        
        if old_price is None:
            old_price = price_history[0][1]  # Use oldest available price
        
        if old_price == 0:
            return None
        
        change_percent = ((current_price - old_price) / old_price) * 100
        return change_percent
    
    @staticmethod
    def create_price_chart(item_name: str, price_history: List[tuple]) -> bytes:
        """Create price chart image"""
        if not price_history:
            return b""
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dates = [entry[0] for entry in price_history]
        prices = [entry[1] for entry in price_history]
        
        ax.plot(dates, prices, linewidth=2, color='#1f77b4')
        ax.fill_between(dates, prices, alpha=0.3, color='#1f77b4')
        
        ax.set_title(f'Price History - {item_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        plt.xticks(rotation=45)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save to bytes
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        chart_bytes = buffer.getvalue()
        plt.close(fig)
        
        return chart_bytes

class Database:
    def __init__(self, db_path: str = "steam_monitor.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                market_hash_name TEXT,
                url TEXT,
                image_url TEXT,
                last_price REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item_id INTEGER,
                price REAL,
                timestamp TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES items (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_item(self, item_data: Dict) -> int:
        """Save or update item in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO items 
            (name, market_hash_name, url, image_url, last_price, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            item_data['name'],
            item_data['market_hash_name'],
            item_data['url'],
            item_data['image_url'],
            item_data['current_price'],
            datetime.now()
        ))
        
        item_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return item_id
    
    def get_last_price(self, item_name: str) -> Optional[float]:
        """Get last recorded price for an item"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT last_price FROM items WHERE name = ?', (item_name,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
    
    async def send_price_alert(self, alert: PriceAlert):
        """Send price alert to Telegram"""
        try:
            # Format message
            direction = "📈" if alert.price_change_percent > 0 else "📉"
            message = f"""
{direction} *PRICE ALERT* {direction}

*{alert.item.name}*

💰 Price Change: *{alert.price_change_percent:.2f}%*
📊 Old Price: ${alert.old_price:.2f}
💲 New Price: ${alert.new_price:.2f}

🔗 [View on Steam Market]({alert.item.url})
            """.strip()
            
            # Send chart image
            if alert.chart_image:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=BytesIO(alert.chart_image),
                    caption=message,
                    parse_mode='Markdown'
                )
            else:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode='Markdown'
                )
            
            logger.info(f"Alert sent for {alert.item.name}")
            
        except TelegramError as e:
            logger.error(f"Failed to send Telegram message: {e}")

class SteamPriceMonitor:
    def __init__(self, bot_token: str, chat_id: str, price_threshold: float = 6.0):
        self.scraper = SteamMarketScraper()
        self.analyzer = PriceAnalyzer()
        self.db = Database()
        self.notifier = TelegramNotifier(bot_token, chat_id)
        self.price_threshold = price_threshold
        self.running = False
    
    async def start(self):
        """Start the monitoring loop"""
        self.running = True
        await self.scraper.init_session()
        
        logger.info("Steam Price Monitor started")
        
        try:
            while self.running:
                await self.check_price_changes()
                # Wait 45 minutes between checks to avoid rate limiting (was 30)
                await asyncio.sleep(2700)  # 45 minutes
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitor"""
        self.running = False
        await self.scraper.close_session()
        logger.info("Steam Price Monitor stopped")
    
    async def check_price_changes(self):
        """Check for significant price changes"""
        try:
            logger.info("Checking for price changes...")
            
            # Get current market items (reduced from 100 to 50 to be gentler)
            items = await self.scraper.get_market_items(max_items=50)
            
            alerts = []
            
            for item_data in items:
                try:
                    # Get price history
                    price_history = await self.scraper.get_price_history(
                        item_data['market_hash_name'], 
                        days=30
                    )
                    
                    if not price_history:
                        continue
                    
                    # Calculate 24h price change
                    change_percent = self.analyzer.calculate_price_change(price_history, hours=24)
                    
                    if change_percent is None:
                        continue
                    
                    # Check if change exceeds threshold
                    if abs(change_percent) >= self.price_threshold:
                        # Create price chart
                        chart_image = self.analyzer.create_price_chart(
                            item_data['name'], 
                            price_history
                        )
                        
                        # Create market item object
                        market_item = MarketItem(
                            name=item_data['name'],
                            url=item_data['url'],
                            current_price=item_data['current_price'],
                            price_history=price_history,
                            image_url=item_data['image_url'],
                            market_hash_name=item_data['market_hash_name']
                        )
                        
                        # Calculate old price
                        old_price = price_history[0][1] if price_history else item_data['current_price']
                        
                        # Create alert
                        alert = PriceAlert(
                            item=market_item,
                            price_change_percent=change_percent,
                            old_price=old_price,
                            new_price=item_data['current_price'],
                            chart_image=chart_image
                        )
                        
                        alerts.append(alert)
                        
                        # Save to database
                        self.db.save_item(item_data)
                    
                    # Longer delay between items to be more respectful
                    await asyncio.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    logger.error(f"Error processing item {item_data.get('name', 'unknown')}: {e}")
                    continue
            
            # Send alerts
            for alert in alerts:
                await self.notifier.send_price_alert(alert)
                await asyncio.sleep(1)  # Delay between messages
            
            logger.info(f"Price check completed. Found {len(alerts)} alerts.")
            
        except Exception as e:
            logger.error(f"Error during price check: {e}")

# Main execution
async def main():
    # Configuration
    BOT_TOKEN = "7988225538:AAEK2ADSC-nudUqAKJHEzw24hvadkcXxhzE"
    CHAT_ID = "7986974989"
    PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "6.0"))
    
    # Create and start monitor
    monitor = SteamPriceMonitor(BOT_TOKEN, CHAT_ID, PRICE_THRESHOLD)
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())
