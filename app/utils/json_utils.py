import json
import uuid
from datetime import date, datetime
from decimal import Decimal


# Custom JSON encoder to handle dates and other special types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle date objects
        if isinstance(obj, date):
            return obj.isoformat()
        # Handle datetime objects
        elif isinstance(obj, datetime):
            return obj.isoformat()
        # Handle Decimal objects (common in PostgreSQL)
        elif isinstance(obj, Decimal):
            return float(obj)
        # Handle UUID objects
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        # Let the base class handle the rest or raise TypeError
        return super().default(obj)
