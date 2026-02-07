"""Internal utility functions."""
from datetime import datetime

now = lambda: datetime.now()
now_date = lambda: datetime.now().date()
