# Steam Market Price Monitor Bot - Improved Version
import asyncio
import aiohttp
import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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
from concurrent.futures import ThreadPoolExecutor
import gc
import contextlib

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
    price_history: List[Tuple[datetime, float]]
    image_url: str
    market_hash_name: str

@dataclass
class PriceAlert:
    item: MarketItem
    price_change_percent: float
    old_price: float
    new_price: float
    chart_image: bytes

@dataclass
class MonitoringStats:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    rate_limited_requests: int = 0
    last_success_time: Optional[datetime] = None
    
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

class DatabaseManager:
    """Thread-safe database manager with connection pooling"""
    
    def __init__(self, db_path: str = "steam_monitor.db"):
        self.db_path = db_path
        self.lock = threading.RLock()
        self.init_db()
    
    def init_db(self):
        """Initialize database tables with proper indexing"""
        with self.lock:
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
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id INTEGER,
                    price REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (item_id) REFERENCES items (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monitoring_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_requests INTEGER,
                    successful_requests INTEGER,
                    failed_requests INTEGER,
                    rate_limited_requests INTEGER,
                    success_rate REAL
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_items_name ON items(name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_item_id ON price_history(item_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_history_timestamp ON price_history(timestamp)')
            
            conn.commit()
            conn.close()
    
    @contextlib.contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def save_item(self, item_data: Dict) -> int:
        """Save or update item in database"""
        with self.get_connection() as conn:
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
            
            item_id = cursor.lastrowid or cursor.execute(
                'SELECT id FROM items WHERE name = ?', (item_data['name'],)
            ).fetchone()[0]
            
            # Store price in history
            cursor.execute('''
                INSERT INTO price_history (item_id, price, timestamp)
                VALUES (?, ?, ?)
            ''', (item_id, item_data['current_price'], datetime.now()))
            
            conn.commit()
            return item_id
    
    def get_last_price(self, item_name: str) -> Optional[float]:
        """Get last recorded price for an item"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            result = cursor.execute(
                'SELECT last_price FROM items WHERE name = ?', 
                (item_name,)
            ).fetchone()
            return result[0] if result else None
    
    def get_price_history(self, item_name: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get price history from database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            result = cursor.execute('''
                SELECT ph.timestamp, ph.price 
                FROM price_history ph
                JOIN items i ON ph.item_id = i.id
                WHERE i.name = ? AND ph.timestamp >= ?
                ORDER BY ph.timestamp ASC
            ''', (item_name, cutoff_date)).fetchall()
            
            return [(datetime.fromisoformat(row[0]), row[1]) for row in result]
    
    def save_monitoring_stats(self, stats: MonitoringStats):
        """Save monitoring statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO monitoring_stats 
                (total_requests, successful_requests, failed_requests, 
                 rate_limited_requests, success_rate)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                stats.total_requests,
                stats.successful_requests, 
                stats.failed_requests,
                stats.rate_limited_requests,
                stats.success_rate()
            ))
            conn.commit()
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old data to prevent database bloat"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean old price history
            cursor.execute(
                'DELETE FROM price_history WHERE timestamp < ?', 
                (cutoff_date,)
            )
            
            # Clean old monitoring stats
            cursor.execute(
                'DELETE FROM monitoring_stats WHERE timestamp < ?', 
                (cutoff_date,)
            )
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days} days")

class SteamMarketScraper:
    def __init__(self):
        self.base_url = "https://steamcommunity.com"
        self.market_url = "https://steamcommunity.com/market/search"
        self.session = None
        self.stats = MonitoringStats()
        self.last_request_time = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Rotate through realistic user agents
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
        ]
        
    def get_random_headers(self) -> Dict[str, str]:
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
        
    async def exponential_backoff_delay(self) -> None:
        """Implement exponential backoff based on consecutive failures"""
        if self.consecutive_failures > 0:
            delay = min(300, (2 ** self.consecutive_failures) + random.uniform(1, 5))  # Max 5 min
            logger.warning(f"Applying exponential backoff: {delay:.1f}s (failures: {self.consecutive_failures})")
            await asyncio.sleep(delay)
        
    async def rate_limit_delay(self) -> None:
        """Intelligent rate limiting based on Steam's patterns"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # More conservative: 6 requests per minute
        min_delay = 60 / 6  # 10 seconds minimum between requests
        
        if time_since_last < min_delay:
            delay = min_delay - time_since_last
            delay += random.uniform(2, 6)  # More jitter
            await asyncio.sleep(delay)
        
        self.last_request_time = time.time()
        self.stats.total_requests += 1
        
        # Every 8 requests, take a longer break
        if self.stats.total_requests % 8 == 0:
            logger.info("Taking extended break to mimic human behavior...")
            await asyncio.sleep(random.uniform(45, 120))  # 45-120 seconds
        
    async def init_session(self) -> None:
        """Initialize aiohttp session with stealth configuration"""
        connector = aiohttp.TCPConnector(
            limit=3,  # Lower connection limit
            limit_per_host=1,  # Very conservative
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=60, connect=20)
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            cookie_jar=aiohttp.CookieJar(),
        )
        
        await self.warm_up_session()
        
    async def warm_up_session(self) -> None:
        """Warm up session by visiting pages like a real user"""
        try:
            logger.info("Warming up session...")
            
            headers = self.get_random_headers()
            async with self.session.get("https://steamcommunity.com/", headers=headers) as response:
                if response.status == 200:
                    logger.info("Successfully warmed up session")
                    
            await asyncio.sleep(random.uniform(3, 8))
            
            headers = self.get_random_headers()
            headers['Referer'] = 'https://steamcommunity.com/'
            async with self.session.get("https://steamcommunity.com/market/", headers=headers) as response:
                pass
                
            await asyncio.sleep(random.uniform(5, 10))
            
        except Exception as e:
            logger.warning(f"Session warmup failed: {e}")
    
    async def close_session(self) -> None:
        if self.session:
            await self.session.close()
    
    def _parse_price(self, price_text: str) -> float:
        """Parse price string to float with comprehensive currency handling"""
        try:
            # Remove common currency symbols and whitespace
            price_clean = (price_text
                          .replace('$', '')
                          .replace('€', '')
                          .replace('£', '')
                          .replace('¥', '')
                          .replace('₽', '')
                          .replace('₹', '')
                          .replace(',', '')
                          .replace(' ', '')
                          .strip())
            
            # Handle cases where price might have additional text
            import re
            price_match = re.search(r'(\d+\.?\d*)', price_clean)
            if price_match:
                return float(price_match.group(1))
            
            return 0.0
        except (ValueError, AttributeError):
            return 0.0
    
    def _extract_market_hash_name(self, url: str) -> str:
        """Extract market hash name from item URL"""
        try:
            parts = url.split('/')
            if len(parts) >= 4:
                encoded_name = parts[-1]
                return urllib.parse.unquote(encoded_name)
            return ''
        except Exception:
            return ''
    
    async def get_market_items(self, max_items: int = 20) -> List[Dict]:
        """Scrape items from Steam Market search page with enhanced error handling"""
        await self.exponential_backoff_delay()
        await self.rate_limit_delay()
        
        params = {
            'descriptions': '1',
            'category_730_ItemSet%5B0%5D': 'any',
            'category_730_Weapon%5B0%5D': 'any', 
            'category_730_Quality%5B0%5D': 'any',
            'appid': '730',
            'q': '',
            'count': str(min(max_items, 20)),
            'start': '0'
        }
        
        headers = self.get_random_headers()
        headers['Referer'] = 'https://steamcommunity.com/market/'
        
        try:
            async with self.session.get(self.market_url, params=params, headers=headers) as response:
                if response.status == 429:
                    self.stats.rate_limited_requests += 1
                    self.consecutive_failures += 1
                    logger.warning("Rate limited on market items, implementing backoff...")
                    return []
                
                if response.status == 403:
                    self.consecutive_failures += 1
                    logger.warning("Access forbidden - possible IP block")
                    return []
                
                if response.status != 200:
                    self.stats.failed_requests += 1
                    self.consecutive_failures += 1
                    logger.error(f"Failed to fetch market page: {response.status}")
                    return []
                
                text = await response.text()
                soup = BeautifulSoup(text, 'html.parser')
                
                items = []
                market_listing_rows = soup.find_all('a', class_='market_listing_row_link')
                
                for row in market_listing_rows[:max_items]:
                    item_data = self._parse_item_row(row)
                    if item_data:
                        items.append(item_data)
                
                self.stats.successful_requests += 1
                self.stats.last_success_time = datetime.now()
                self.consecutive_failures = 0  # Reset on success
                
                logger.info(f"Successfully scraped {len(items)} market items")
                return items
                
        except asyncio.TimeoutError:
            self.stats.failed_requests += 1
            self.consecutive_failures += 1
            logger.error("Timeout fetching market items")
            return []
        except Exception as e:
            self.stats.failed_requests += 1
            self.consecutive_failures += 1
            logger.error(f"Error scraping market items: {e}")
            return []
    
    def _parse_item_row(self, row_element) -> Optional[Dict]:
        """Parse individual item row from market listing"""
        try:
            name_elem = row_element.find('span', class_='market_listing_item_name')
            if not name_elem:
                return None
            
            item_name = name_elem.get_text(strip=True)
            
            price_elem = row_element.find('span', class_='normal_price')
            if not price_elem:
                price_elem = row_element.find('span', class_='sale_price')
            
            if not price_elem:
                return None
            
            price_text = price_elem.get_text(strip=True)
            price = self._parse_price(price_text)
            
            img_elem = row_element.find('img', class_='market_listing_item_img')
            image_url = img_elem.get('src') if img_elem else ''
            
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
    
    async def get_price_history(self, market_hash_name: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get price history for an item with enhanced error handling"""
        await self.exponential_backoff_delay()
        await self.rate_limit_delay()
        
        url = f"{self.base_url}/market/pricehistory/"
        encoded_name = urllib.parse.quote(market_hash_name)
        
        params = {
            'appid': '730',
            'market_hash_name': encoded_name
        }
        
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
                    self.stats.rate_limited_requests += 1
                    self.consecutive_failures += 1
                    return []
                
                if response.status == 403:
                    self.consecutive_failures += 1
                    logger.warning(f"Forbidden access for {market_hash_name}")
                    return []
                
                if response.status != 200:
                    self.stats.failed_requests += 1
                    self.consecutive_failures += 1
                    return []
                
                try:
                    data = await response.json()
                except aiohttp.ContentTypeError:
                    self.stats.failed_requests += 1
                    return []
                
                if not data.get('success'):
                    return []
                
                prices = data.get('prices', [])
                price_history = []
                cutoff_date = datetime.now() - timedelta(days=days)
                
                for price_entry in prices:
                    try:
                        if len(price_entry) >= 2:
                            date_str = price_entry[0]
                            price = float(price_entry[1])
                            
                            date_obj = None
                            try:
                                if len(date_str) >= 11:
                                    date_obj = datetime.strptime(date_str[:11], "%b %d %Y")
                                elif len(date_str) >= 6:
                                    date_obj = datetime.strptime(date_str[:6], "%b %d")
                                    date_obj = date_obj.replace(year=datetime.now().year)
                            except ValueError:
                                continue
                            
                            if date_obj and date_obj >= cutoff_date and price > 0:
                                price_history.append((date_obj, price))
                    except (ValueError, IndexError):
                        continue
                
                sorted_history = sorted(price_history, key=lambda x: x[0])
                self.stats.successful_requests += 1
                self.stats.last_success_time = datetime.now()
                self.consecutive_failures = 0  # Reset on success
                
                return sorted_history
                
        except asyncio.TimeoutError:
            self.stats.failed_requests += 1
            self.consecutive_failures += 1
            return []
        except Exception as e:
            self.stats.failed_requests += 1
            self.consecutive_failures += 1
            logger.warning(f"Error getting price history for {market_hash_name}: {e}")
            return []

class PriceAnalyzer:
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
    
    @staticmethod
    def calculate_price_change(price_history: List[Tuple[datetime, float]], hours: int = 24) -> Optional[float]:
        """Calculate price change percentage over specified hours"""
        if len(price_history) < 2:
            return None
        
        now = datetime.now()
        cutoff_time = now - timedelta(hours=hours)
        
        current_price = price_history[-1][1]
        
        old_price = None
        for timestamp, price in reversed(price_history):
            if timestamp <= cutoff_time:
                old_price = price
                break
        
        if old_price is None:
            old_price = price_history[0][1]
        
        if old_price == 0:
            return None
        
        change_percent = ((current_price - old_price) / old_price) * 100
        return change_percent
    
    async def create_price_chart(self, item_name: str, price_history: List[Tuple[datetime, float]]) -> bytes:
        """Create price chart image using thread pool to avoid blocking"""
        if not price_history:
            return b""
        
        def _create_chart():
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                dates = [entry[0] for entry in price_history]
                prices = [entry[1] for entry in price_history]
                
                ax.plot(dates, prices, linewidth=2, color='#1f77b4')
                ax.fill_between(dates, prices, alpha=0.3, color='#1f77b4')
                
                ax.set_title(f'Price History - {item_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price ($)')
                
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
                plt.xticks(rotation=45)
                
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')  # Lower DPI for memory
                buffer.seek(0)
                chart_bytes = buffer.getvalue()
                
                plt.close(fig)  # Important: close figure to free memory
                plt.clf()       # Clear the current figure
                gc.collect()    # Force garbage collection
                
                return chart_bytes
                
            except Exception as e:
                logger.error(f"Error creating chart: {e}")
                return b""
        
        # Run chart creation in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, _create_chart)

class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
    
    async def send_price_alert(self, alert: PriceAlert) -> bool:
        """Send price alert to Telegram with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                direction = "📈" if alert.price_change_percent > 0 else "📉"
                message = f"""
{direction} *PRICE ALERT* {direction}

*{alert.item.name}*

💰 Price Change: *{alert.price_change_percent:.2f}%*
📊 Old Price: ${alert.old_price:.2f}
💲 New Price: ${alert.new_price:.2f}

🔗 [View on Steam Market]({alert.item.url})
                """.strip()
                
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
                return True
                
            except TelegramError as e:
                logger.error(f"Telegram error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
        
        return False
    
    async def send_status_update(self, stats: MonitoringStats) -> bool:
        """Send monitoring status update"""
        try:
            message = f"""
🤖 *Bot Status Update*

📊 *Statistics:*
• Total Requests: {stats.total_requests}
• Success Rate: {stats.success_rate():.1f}%
• Rate Limited: {stats.rate_limited_requests}
• Last Success: {stats.last_success_time.strftime('%H:%M:%S') if stats.last_success_time else 'Never'}

Bot is running and monitoring prices!
            """.strip()
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            return True
            
        except TelegramError as e:
            logger.error(f"Failed to send status update: {e}")
            return False

class SteamPriceMonitor:
    def __init__(self, bot_token: str, chat_id: str, price_threshold: float = 6.0):
        self.executor = ThreadPoolExecutor(max_workers=2)  # For chart generation
        self.scraper = SteamMarketScraper()
        self.analyzer = PriceAnalyzer(self.executor)
        self.db = DatabaseManager()
        self.notifier = TelegramNotifier(bot_token, chat_id)
        self.price_threshold = price_threshold
        self.running = False
        self.cycle_count = 0
    
    async def start(self):
        """Start the monitoring loop"""
        self.running = True
        await self.scraper.init_session()
        
        logger.info("Steam Price Monitor started")
        
        try:
            while self.running:
                await self.check_price_changes()
                
                # Send status update every 6 hours
                if self.cycle_count % 6 == 0 and self.cycle_count > 0:
                    await self.notifier.send_status_update(self.scraper.stats)
                
                # Clean up old data weekly
                if self.cycle_count % 168 == 0 and self.cycle_count > 0:  # 168 hours = 1 week
                    self.db.cleanup_old_data()
                
                # Wait 60-90 minutes between cycles
                wait_time = random.uniform(3600, 5400)
                logger.info(f"Waiting {wait_time/60:.1f} minutes until next cycle...")
                await asyncio.sleep(wait_time)
                
                self.cycle_count += 1
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitor and cleanup resources"""
        self.running = False
        await self.scraper.close_session()
        self.executor.shutdown(wait=True)
        logger.info("Steam Price Monitor stopped")
    
    async def check_price_changes(self):
        """Check for significant price changes with comprehensive error handling"""
        try:
            logger.info("Starting price change check...")
            
            items = await self.scraper.get_market_items(max_items=15)  # Even more conservative
            
            if not items:
                logger.warning("No items retrieved from market scraping")
                return
            
            alerts = []
            successful_checks = 0
            
            for item_data in items:
                try:
                    # Get price history from Steam API
                    steam_price_history = await self.scraper.get_price_history(
                        item_data['market_hash_name'], 
                        days=30
                    )
                    
                    # Get database price history as fallback
                    db_price_history = self.db.get_price_history(item_data['name'], days=30)
                    
                    # Use Steam data if available, otherwise use database
                    if steam_price_history:
                        price_history = steam_price_history
                        successful_checks += 1
                        logger.debug(f"Using Steam API data for {item_data['name']}")
                    elif db_price_history:
                        price_history = db_price_history
                        logger.debug(f"Using database data for {item_data['name']}")
                    else:
                        # First time seeing this item - just store it
                        self.db.save_item(item_data)
                        logger.debug(f"New item stored: {item_data['name']}")
                        continue
                    
                    # Calculate price change
                    change_percent = self.analyzer.calculate_price_change(price_history, hours=24)
                    
                    if change_percent is None:
                        # Not enough data for comparison
                        self.db.save_item(item_data)
                        continue
                    
                    # Check if change exceeds threshold
                    if abs(change_percent) >= self.price_threshold:
                        logger.info(f"Price alert triggered for {item_data['name']}: {change_percent:.2f}%")
                        
                        # Create price chart
                        chart_image = await self.analyzer.create_price_chart(
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
                        
                        # Calculate old price from history
                        cutoff_time = datetime.now() - timedelta(hours=24)
                        old_price = None
                        
                        for timestamp, price in reversed(price_history):
                            if timestamp <= cutoff_time:
                                old_price = price
                                break
                        
                        if old_price is None and price_history:
                            old_price = price_history[0][1]
                        
                        if old_price is None:
                            old_price = item_data['current_price']
                        
                        # Create alert
                        alert = PriceAlert(
                            item=market_item,
                            price_change_percent=change_percent,
                            old_price=old_price,
                            new_price=item_data['current_price'],
                            chart_image=chart_image
                        )
                        
                        alerts.append(alert)
                    
                    # Always save current data
                    self.db.save_item(item_data)
                    
                    # Delay between items to be respectful
                    await asyncio.sleep(random.uniform(8, 15))
                    
                except Exception as e:
                    logger.error(f"Error processing item {item_data.get('name', 'unknown')}: {e}")
                    continue
            
            # Send alerts
            alert_sent_count = 0
            for alert in alerts:
                success = await self.notifier.send_price_alert(alert)
                if success:
                    alert_sent_count += 1
                await asyncio.sleep(2)  # Delay between messages
            
            # Save monitoring stats
            self.db.save_monitoring_stats(self.scraper.stats)
            
            # Log cycle summary
            success_rate = self.scraper.stats.success_rate()
            logger.info(f"Cycle completed: {len(items)} items processed, {successful_checks} successful API calls, "
                       f"{len(alerts)} alerts generated, {alert_sent_count} alerts sent. "
                       f"Overall success rate: {success_rate:.1f}%")
            
            # Send status update if success rate is very low
            if (self.scraper.stats.total_requests > 10 and success_rate < 20 and 
                self.cycle_count % 3 == 0):  # Every 3rd cycle if low success
                
                try:
                    await self.notifier.bot.send_message(
                        chat_id=self.notifier.chat_id,
                        text=f"⚠️ *Low Success Rate Warning*\n\n"
                             f"Steam is blocking {100-success_rate:.1f}% of requests.\n"
                             f"Bot continues running with database fallback.\n"
                             f"Success rate: {success_rate:.1f}%",
                        parse_mode='Markdown'
                    )
                except Exception:
                    pass
            
        except Exception as e:
            logger.error(f"Critical error during price check: {e}")

# Configuration and main execution
def get_config():
    """Get configuration from environment variables"""
    config = {
        'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
        'chat_id': os.getenv('TELEGRAM_CHAT_ID'), 
        'price_threshold': float(os.getenv('PRICE_THRESHOLD', '6.0')),
        'max_items': int(os.getenv('MAX_ITEMS', '15')),
        'port': os.getenv('PORT')
    }
    
    # Validate required config
    if not config['bot_token']:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    if not config['chat_id']:
        raise ValueError("TELEGRAM_CHAT_ID environment variable is required")
    
    return config

async def setup_health_server(port: int):
    """Setup health check server for deployment platforms"""
    from aiohttp import web
    
    async def health_check(request):
        return web.Response(
            text="Steam Price Monitor is running!",
            headers={'Content-Type': 'text/plain'}
        )
    
    async def status(request):
        # You could add more detailed status here
        return web.json_response({
            'status': 'running',
            'service': 'steam-price-monitor',
            'timestamp': datetime.now().isoformat()
        })
    
    app = web.Application()
    app.router.add_get('/', health_check)
    app.router.add_get('/health', health_check)
    app.router.add_get('/status', status)
    
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health check server started on port {port}")
    
    return runner

async def main():
    """Main execution function with proper error handling"""
    try:
        # Get configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Start health check server if needed
        health_runner = None
        if config['port']:
            health_runner = await setup_health_server(int(config['port']))
        
        # Create and start monitor
        monitor = SteamPriceMonitor(
            bot_token=config['bot_token'],
            chat_id=config['chat_id'],
            price_threshold=config['price_threshold']
        )
        
        # Send startup notification
        try:
            await monitor.notifier.bot.send_message(
                chat_id=monitor.notifier.chat_id,
                text="🚀 *Steam Price Monitor Started*\n\n"
                     f"Monitoring threshold: {config['price_threshold']}%\n"
                     f"Max items per cycle: {config['max_items']}\n"
                     "Bot is now running!",
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.warning(f"Could not send startup notification: {e}")
        
        # Start monitoring
        await monitor.start()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
    finally:
        # Cleanup
        try:
            if 'monitor' in locals():
                await monitor.stop()
            if health_runner:
                await health_runner.cleanup()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

if __name__ == "__main__":
    # Set up proper signal handling for graceful shutdown
    import signal
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        raise KeyboardInterrupt
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the main function
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Application crashed: {e}")
        exit(1)
