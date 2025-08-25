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
import urllib.parse

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
        self.request_count = 0
        self.last_request_time = time.time()
        self.session_cookies = {}
        
        # Rotate through realistic user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
        ]
        
    def get_random_headers(self):
        """Generate realistic browser headers with rotation"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': random.choice([
                'en-US,en;q=0.9',
                'en-GB,en;q=0.9',
                'en-US,en;q=0.9,es;q=0.8',
                'en-US,en;q=0.5'
            ]),
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': random.choice(['"Windows"', '"macOS"', '"Linux"'])
        }
        
    async def rate_limit_delay(self):
        """Intelligent rate limiting based on Steam's patterns"""
        current_time = time.time()
        
        # Steam allows ~10-15 requests per minute
        # We'll be very conservative: 8 requests per minute
        time_since_last = current_time - self.last_request_time
        min_delay = 60 / 8  # 7.5 seconds minimum between requests
        
        if time_since_last < min_delay:
            delay = min_delay - time_since_last
            # Add random jitter to appear more human
            delay += random.uniform(1, 4)
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
        self.request_count += 1
        
        # Every 10 requests, take a longer break (simulate human behavior)
        if self.request_count % 10 == 0:
            logger.info("Taking extended break to mimic human behavior...")
            await asyncio.sleep(random.uniform(30, 90))
        
    async def init_session(self):
        """Initialize aiohttp session with stealth configuration"""
        # Create session with rotating headers
        connector = aiohttp.TCPConnector(
            limit=5,  # Lower connection limit
            limit_per_host=2,  # Very conservative
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=45, connect=15)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            cookie_jar=aiohttp.CookieJar(),
        )
        
        # Initialize session by visiting main page (like a real user)
        await self.warm_up_session()
        
    async def warm_up_session(self):
        """Warm up session by visiting pages like a real user"""
        try:
            logger.info("Warming up session to appear more legitimate...")
            
            # Visit main Steam Community page first
            headers = self.get_random_headers()
            async with self.session.get(
                "https://steamcommunity.com/", 
                headers=headers
            ) as response:
                if response.status == 200:
                    logger.info("Successfully warmed up session")
                    
            await asyncio.sleep(random.uniform(2, 5))
            
            # Visit market main page
            headers = self.get_random_headers()
            headers['Referer'] = 'https://steamcommunity.com/'
            async with self.session.get(
                "https://steamcommunity.com/market/", 
                headers=headers
            ) as response:
                pass
                
            await asyncio.sleep(random.uniform(3, 7))
            
        except Exception as e:
            logger.warning(f"Session warmup failed: {e}")
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def get_market_items(self, max_items: int = 30) -> List[Dict]:
        """Scrape items from Steam Market search page with stealth techniques"""
        await self.rate_limit_delay()
        
        params = {
            'descriptions': '1',
            'category_730_ItemSet%5B0%5D': 'any',
            'category_730_Weapon%5B0%5D': 'any', 
            'category_730_Quality%5B0%5D': 'any',
            'category_730_ProPlayer%5B0%5D': 'any',
            'category_730_StickerCapsule%5B0%5D': 'any',
            'appid': '730',
            'q': '',
            'count': str(min(max_items, 30)),  # Even more conservative
            'start': '0'
        }
        
        headers = self.get_random_headers()
        headers['Referer'] = 'https://steamcommunity.com/market/'
        
        try:
            async with self.session.get(
                self.market_url, 
                params=params, 
                headers=headers
            ) as response:
                if response.status == 429:
                    logger.warning("Rate limited on market items, waiting 2 minutes...")
                    await asyncio.sleep(120)
                    return await self.get_market_items(max_items)
                
                if response.status != 200:
                    logger.error(f"Failed to fetch market page: {response.status}")
                    return []
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                items = []
                market_listing_rows = soup.find_all('a', class_='market_listing_row_link')
                
                for row in market_listing_rows[:max_items]:
                    item_data = await self._parse_item_row(row)
                    if item_data:
                        items.append(item_data)
                
                logger.info(f"Successfully scraped {len(items)} market items")
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
            price_clean = price_text.replace('
    
    async def get_price_history(self, market_hash_name: str, days: int = 30) -> List[tuple]:
        """Get price history for an item with advanced stealth techniques"""
        await self.rate_limit_delay()
        
        url = f"{self.base_url}/market/pricehistory/"
        
        # Properly encode the market hash name
        encoded_name = urllib.parse.quote(market_hash_name)
        
        params = {
            'appid': '730',
            'market_hash_name': encoded_name
        }
        
        # Realistic headers that mimic AJAX requests from Steam's own frontend
        headers = self.get_random_headers()
        headers.update({
            'Referer': f'{self.base_url}/market/listings/730/{encoded_name}',
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
        })
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 429:
                    # Rate limited - wait with exponential backoff
                    wait_time = random.uniform(60, 180)  # 1-3 minutes
                    logger.warning(f"Rate limited for {market_hash_name}, waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                    return await self.get_price_history(market_hash_name, days)
                
                if response.status == 403:
                    logger.warning(f"Forbidden access for {market_hash_name} - might be region locked or require login")
                    return []
                
                if response.status != 200:
                    logger.warning(f"Failed to get price history for {market_hash_name}: {response.status}")
                    return []
                
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
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
                                # Try different date formats that Steam uses
                                if len(date_str) >= 11:
                                    date_obj = datetime.strptime(date_str[:11], "%b %d %Y")
                                elif len(date_str) >= 6:
                                    date_obj = datetime.strptime(date_str[:6], "%b %d")
                                    date_obj = date_obj.replace(year=datetime.now().year)
                            except ValueError:
                                # Try alternative parsing
                                continue
                            
                            if date_obj and date_obj >= cutoff_date and price > 0:
                                price_history.append((date_obj, price))
                    except (ValueError, IndexError):
                        continue
                
                sorted_history = sorted(price_history, key=lambda x: x[0])
                logger.info(f"Successfully retrieved {len(sorted_history)} price points for {market_hash_name}")
                return sorted_history
                
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
                # Wait 60-90 minutes between full cycles (even more conservative)
                wait_time = random.uniform(3600, 5400)  # 60-90 minutes
                logger.info(f"Waiting {wait_time/60:.1f} minutes until next price check...")
                await asyncio.sleep(wait_time)
                
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
            
            # Get current market items (reduced to 20 for maximum stealth)
            items = await self.scraper.get_market_items(max_items=20)
            
            alerts = []
            successful_checks = 0
            
            for item_data in items:
                try:
                    # Get price history
                    price_history = await self.scraper.get_price_history(
                        item_data['market_hash_name'], 
                        days=30
                    )
                    
                    # If we can't get price history, try simpler price tracking
                    if not price_history:
                        # Use database to track price changes
                        last_price = self.db.get_last_price(item_data['name'])
                        current_price = item_data['current_price']
                        
                        if last_price and current_price > 0 and last_price > 0:
                            change_percent = ((current_price - last_price) / last_price) * 100
                            
                            if abs(change_percent) >= self.price_threshold:
                                # Create a simple price alert without chart
                                market_item = MarketItem(
                                    name=item_data['name'],
                                    url=item_data['url'],
                                    current_price=current_price,
                                    price_history=[],
                                    image_url=item_data['image_url'],
                                    market_hash_name=item_data['market_hash_name']
                                )
                                
                                alert = PriceAlert(
                                    item=market_item,
                                    price_change_percent=change_percent,
                                    old_price=last_price,
                                    new_price=current_price,
                                    chart_image=b""  # No chart available
                                )
                                
                                alerts.append(alert)
                                logger.info(f"Price alert triggered for {item_data['name']}: {change_percent:.2f}%")
                        
                        # Save current price for next comparison
                        self.db.save_item(item_data)
                        continue
                    
                    successful_checks += 1
                    
                    # Calculate 24h price change
                    change_percent = self.analyzer.calculate_price_change(price_history, hours=24)
                    
                    if change_percent is None:
                        # Save item for future comparison
                        self.db.save_item(item_data)
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
                        logger.info(f"Full price alert triggered for {item_data['name']}: {change_percent:.2f}%")
                    
                    # Save to database
                    self.db.save_item(item_data)
                    
                    # Much longer delay between items (10-30 seconds)
                    await asyncio.sleep(random.uniform(10, 30))
                    
                except Exception as e:
                    logger.error(f"Error processing item {item_data.get('name', 'unknown')}: {e}")
                    continue
            
            # Send alerts
            for alert in alerts:
                await self.notifier.send_price_alert(alert)
                await asyncio.sleep(1)  # Delay between messages
            
            logger.info(f"Price check completed. Processed {len(items)} items, {successful_checks} with full history, found {len(alerts)} alerts.")
            
            # If we're getting mostly failures, send a status update
            if len(items) > 0 and successful_checks < len(items) * 0.1:  # Less than 10% success
                try:
                    await self.notifier.bot.send_message(
                        chat_id=self.notifier.chat_id,
                        text=f"🤖 Bot Status Update\n\nBot is running but Steam is blocking most price history requests ({successful_checks}/{len(items)} successful).\n\nUsing fallback method to track price changes. You'll still get alerts when prices move significantly!",
                        parse_mode='Markdown'
                    )
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error during price check: {e}")

# Main execution
async def main():
    # Configuration
    BOT_TOKEN = "7988225538:AAEK2ADSC-nudUqAKJHEzw24hvadkcXxhzE"
    CHAT_ID = "7986974989"
    PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "6.0"))
    
    # Start a simple HTTP server for Render (if PORT env var exists)
    PORT = os.getenv("PORT")
    if PORT:
        # Start simple health check server
        from aiohttp import web
        
        async def health_check(request):
            return web.Response(text="Steam Price Monitor is running!")
        
        app = web.Application()
        app.router.add_get('/', health_check)
        app.router.add_get('/health', health_check)
        
        # Start web server in background
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(PORT))
        await site.start()
        logger.info(f"Health check server started on port {PORT}")
    
    # Create and start monitor
    monitor = SteamPriceMonitor(BOT_TOKEN, CHAT_ID, PRICE_THRESHOLD)
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(main()), '').replace('€', '').replace('£', '').replace(',', '').strip()
            return float(price_clean)
        except:
            return 0.0
    
    def _extract_market_hash_name(self, url: str) -> str:
        """Extract market hash name from item URL"""
        try:
            # URL format: /market/listings/730/AK-47%20%7C%20Redline%20%28Field-Tested%29
            parts = url.split('/')
            if len(parts) >= 4:
                encoded_name = parts[-1]
                # Decode URL encoding
                return urllib.parse.unquote(encoded_name)
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
            successful_checks = 0
            
            for item_data in items:
                try:
                    # Get price history
                    price_history = await self.scraper.get_price_history(
                        item_data['market_hash_name'], 
                        days=30
                    )
                    
                    # If we can't get price history, try simpler price tracking
                    if not price_history:
                        # Use database to track price changes
                        last_price = self.db.get_last_price(item_data['name'])
                        current_price = item_data['current_price']
                        
                        if last_price and current_price > 0 and last_price > 0:
                            change_percent = ((current_price - last_price) / last_price) * 100
                            
                            if abs(change_percent) >= self.price_threshold:
                                # Create a simple price alert without chart
                                market_item = MarketItem(
                                    name=item_data['name'],
                                    url=item_data['url'],
                                    current_price=current_price,
                                    price_history=[],
                                    image_url=item_data['image_url'],
                                    market_hash_name=item_data['market_hash_name']
                                )
                                
                                alert = PriceAlert(
                                    item=market_item,
                                    price_change_percent=change_percent,
                                    old_price=last_price,
                                    new_price=current_price,
                                    chart_image=b""  # No chart available
                                )
                                
                                alerts.append(alert)
                                logger.info(f"Price alert triggered for {item_data['name']}: {change_percent:.2f}%")
                        
                        # Save current price for next comparison
                        self.db.save_item(item_data)
                        continue
                    
                    successful_checks += 1
                    
                    # Calculate 24h price change
                    change_percent = self.analyzer.calculate_price_change(price_history, hours=24)
                    
                    if change_percent is None:
                        # Save item for future comparison
                        self.db.save_item(item_data)
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
                        logger.info(f"Full price alert triggered for {item_data['name']}: {change_percent:.2f}%")
                    
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
            
            logger.info(f"Price check completed. Processed {len(items)} items, {successful_checks} with full history, found {len(alerts)} alerts.")
            
            # If we're getting mostly failures, send a status update
            if len(items) > 0 and successful_checks < len(items) * 0.1:  # Less than 10% success
                try:
                    await self.notifier.bot.send_message(
                        chat_id=self.notifier.chat_id,
                        text=f"🤖 Bot Status Update\n\nBot is running but Steam is blocking most price history requests ({successful_checks}/{len(items)} successful).\n\nUsing fallback method to track price changes. You'll still get alerts when prices move significantly!",
                        parse_mode='Markdown'
                    )
                except:
                    pass
            
        except Exception as e:
            logger.error(f"Error during price check: {e}")

# Main execution
async def main():
    # Configuration
    BOT_TOKEN = "7988225538:AAEK2ADSC-nudUqAKJHEzw24hvadkcXxhzE"
    CHAT_ID = "7986974989"
    PRICE_THRESHOLD = float(os.getenv("PRICE_THRESHOLD", "6.0"))
    
    # Start a simple HTTP server for Render (if PORT env var exists)
    PORT = os.getenv("PORT")
    if PORT:
        # Start simple health check server
        from aiohttp import web
        
        async def health_check(request):
            return web.Response(text="Steam Price Monitor is running!")
        
        app = web.Application()
        app.router.add_get('/', health_check)
        app.router.add_get('/health', health_check)
        
        # Start web server in background
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', int(PORT))
        await site.start()
        logger.info(f"Health check server started on port {PORT}")
    
    # Create and start monitor
    monitor = SteamPriceMonitor(BOT_TOKEN, CHAT_ID, PRICE_THRESHOLD)
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())
