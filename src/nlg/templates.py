"""Template-based Natural Language Generation for Vietnamese AI Call Center.

Generates polite Vietnamese responses using templates with slot filling.
All responses use formal, polite language with "ạ" suffix and "anh/chị" pronouns.
"""

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# Vietnamese city name mappings for natural output
CITY_DISPLAY_NAMES = {
    "ha noi": "Hà Nội",
    "hanoi": "Hà Nội",
    "ho chi minh": "Hồ Chí Minh",
    "hcm": "Hồ Chí Minh",
    "sai gon": "Sài Gòn",
    "saigon": "Sài Gòn",
    "da nang": "Đà Nẵng",
    "danang": "Đà Nẵng",
    "nha trang": "Nha Trang",
    "can tho": "Cần Thơ",
    "hai phong": "Hải Phòng",
    "hue": "Huế",
    "da lat": "Đà Lạt",
    "dalat": "Đà Lạt",
    "phu quoc": "Phú Quốc",
    "quy nhon": "Quy Nhơn",
    "vinh": "Vinh",
    "buon ma thuot": "Buôn Ma Thuột",
    "pleiku": "Pleiku",
    "rach gia": "Rạch Giá",
    "con dao": "Côn Đảo",
    "dien bien": "Điện Biên",
    # US cities (ATIS dataset includes these)
    "boston": "Boston",
    "new york": "New York",
    "los angeles": "Los Angeles",
    "san francisco": "San Francisco",
    "chicago": "Chicago",
    "dallas": "Dallas",
    "denver": "Denver",
    "seattle": "Seattle",
    "atlanta": "Atlanta",
    "miami": "Miami",
    "washington": "Washington",
    "philadelphia": "Philadelphia",
    "phoenix": "Phoenix",
    "pittsburgh": "Pittsburgh",
    "detroit": "Detroit",
    "minneapolis": "Minneapolis",
    "st. louis": "St. Louis",
    "baltimore": "Baltimore",
    "oakland": "Oakland",
    "tampa": "Tampa",
    "orlando": "Orlando",
    "san diego": "San Diego",
    "houston": "Houston",
    "las vegas": "Las Vegas",
    "salt lake city": "Salt Lake City",
    "kansas city": "Kansas City",
    "milwaukee": "Milwaukee",
    "cleveland": "Cleveland",
    "cincinnati": "Cincinnati",
    "indianapolis": "Indianapolis",
    "columbus": "Columbus",
    "nashville": "Nashville",
    "memphis": "Memphis",
    "charlotte": "Charlotte",
    "newark": "Newark",
    "burbank": "Burbank",
    "long beach": "Long Beach",
    "ontario": "Ontario",
    "san jose": "San Jose",
    "westchester county": "Westchester County",
    "montreal": "Montreal",
    "toronto": "Toronto",
}

# Airline display names
AIRLINE_DISPLAY_NAMES = {
    "vietnam airlines": "Vietnam Airlines",
    "vn": "Vietnam Airlines",
    "vietjet": "Vietjet Air",
    "vj": "Vietjet Air",
    "bamboo": "Bamboo Airways",
    "bamboo airways": "Bamboo Airways",
    "pacific airlines": "Pacific Airlines",
    # US airlines from ATIS
    "american": "American Airlines",
    "aa": "American Airlines",
    "united": "United Airlines",
    "ua": "United Airlines",
    "delta": "Delta Air Lines",
    "dl": "Delta Air Lines",
    "northwest": "Northwest Airlines",
    "nw": "Northwest Airlines",
    "continental": "Continental Airlines",
    "co": "Continental Airlines",
    "usair": "US Airways",
    "us": "US Airways",
    "twa": "TWA",
    "tw": "TWA",
    "southwest": "Southwest Airlines",
    "wn": "Southwest Airlines",
    "america west": "America West",
    "hp": "America West",
    "midwest express": "Midwest Express",
    "canadian airlines": "Canadian Airlines",
    "cp": "Canadian Airlines",
    "air canada": "Air Canada",
    "ac": "Air Canada",
}

# Day name mappings
DAY_DISPLAY_NAMES = {
    "monday": "thứ Hai",
    "tuesday": "thứ Ba",
    "wednesday": "thứ Tư",
    "thursday": "thứ Năm",
    "friday": "thứ Sáu",
    "saturday": "thứ Bảy",
    "sunday": "Chủ Nhật",
    "thu hai": "thứ Hai",
    "thu ba": "thứ Ba",
    "thu tu": "thứ Tư",
    "thu nam": "thứ Năm",
    "thu sau": "thứ Sáu",
    "thu bay": "thứ Bảy",
    "chu nhat": "Chủ Nhật",
}

# Month name mappings
MONTH_DISPLAY_NAMES = {
    "january": "tháng 1",
    "february": "tháng 2",
    "march": "tháng 3",
    "april": "tháng 4",
    "may": "tháng 5",
    "june": "tháng 6",
    "july": "tháng 7",
    "august": "tháng 8",
    "september": "tháng 9",
    "october": "tháng 10",
    "november": "tháng 11",
    "december": "tháng 12",
    "thang 1": "tháng 1",
    "thang 2": "tháng 2",
    "thang 3": "tháng 3",
    "thang 4": "tháng 4",
    "thang 5": "tháng 5",
    "thang 6": "tháng 6",
    "thang 7": "tháng 7",
    "thang 8": "tháng 8",
    "thang 9": "tháng 9",
    "thang 10": "tháng 10",
    "thang 11": "tháng 11",
    "thang 12": "tháng 12",
}


# Response templates organized by action type
TEMPLATES: Dict[str, List[str]] = {
    # Greeting
    "greet": [
        "Xin chào! Em là trợ lý hàng không ảo. Em có thể giúp gì cho anh/chị ạ?",
        "Chào anh/chị! Em là trợ lý đặt vé máy bay. Anh/chị cần em hỗ trợ gì ạ?",
        "Xin chào! Em sẵn sàng hỗ trợ anh/chị đặt vé và tra cứu thông tin chuyến bay ạ.",
    ],
    # Clarification
    "clarify": [
        "Xin lỗi, em chưa nghe rõ. Anh/chị có thể nói lại được không ạ?",
        "Em chưa hiểu ý anh/chị. Anh/chị có thể nói rõ hơn được không ạ?",
        "Xin lỗi anh/chị, em không nghe rõ. Anh/chị vui lòng nhắc lại giúp em ạ.",
    ],
    # Slot requests - from location
    "request_fromloc.city_name": [
        "Anh/chị muốn bay từ thành phố nào ạ?",
        "Xin cho em biết điểm khởi hành của anh/chị ạ?",
        "Anh/chị sẽ bay từ đâu ạ?",
    ],
    # Slot requests - to location
    "request_toloc.city_name": [
        "Anh/chị muốn bay đến thành phố nào ạ?",
        "Điểm đến của anh/chị là đâu ạ?",
        "Anh/chị muốn đến đâu ạ?",
    ],
    # Slot requests - departure date
    "request_depart_date.day_name": [
        "Anh/chị muốn bay vào ngày nào trong tuần ạ?",
        "Anh/chị dự định bay vào thứ mấy ạ?",
    ],
    "request_depart_date.month_name": [
        "Anh/chị muốn bay vào tháng mấy ạ?",
        "Xin cho em biết tháng anh/chị muốn đi ạ?",
    ],
    "request_depart_date.day_number": [
        "Anh/chị muốn bay vào ngày mấy ạ?",
        "Xin cho em biết ngày cụ thể anh/chị muốn đi ạ?",
    ],
    "request_depart_date.today_relative": [
        "Anh/chị muốn bay vào ngày nào ạ?",
        "Khi nào anh/chị muốn khởi hành ạ?",
    ],
    # Slot requests - departure time
    "request_depart_time.time": [
        "Anh/chị muốn bay lúc mấy giờ ạ?",
        "Anh/chị thích bay vào giờ nào ạ?",
    ],
    "request_depart_time.period_of_day": [
        "Anh/chị muốn bay buổi sáng, chiều hay tối ạ?",
        "Anh/chị thích bay vào buổi nào trong ngày ạ?",
    ],
    # Slot requests - airline
    "request_airline_name": [
        "Anh/chị có muốn chọn hãng hàng không nào không ạ?",
        "Anh/chị thích đi hãng bay nào ạ?",
    ],
    # Slot requests - class
    "request_class_type": [
        "Anh/chị muốn đặt vé hạng nào ạ? Phổ thông hay thương gia?",
        "Anh/chị muốn bay hạng phổ thông hay thương gia ạ?",
    ],
    # Confirmation
    "confirm": [
        "Xác nhận: Chuyến bay từ {fromloc} đến {toloc}{date_info}{airline_info}. Đúng không ạ?",
        "Em xác nhận lại: Anh/chị muốn bay từ {fromloc} đến {toloc}{date_info}{airline_info}. Anh/chị xác nhận giúp em ạ?",
    ],
    # Execution - booking success
    "execute_flight": [
        "Đã tìm thấy chuyến bay cho anh/chị! Mã đặt chỗ: {booking_id}. Em có thể hỗ trợ gì thêm không ạ?",
        "Em đã đặt vé thành công! Mã xác nhận của anh/chị là {booking_id}. Cảm ơn anh/chị đã sử dụng dịch vụ ạ!",
    ],
    # Information responses
    "respond_flight_time": [
        "Chuyến bay từ {fromloc} đến {toloc} thường mất khoảng {duration}. Anh/chị cần thêm thông tin gì không ạ?",
        "Thời gian bay từ {fromloc} đến {toloc} là khoảng {duration} ạ.",
    ],
    "respond_airline": [
        "Các hãng bay phổ biến trên tuyến này gồm: Vietnam Airlines, Vietjet Air, và Bamboo Airways ạ.",
        "Em có thể tìm vé của Vietnam Airlines, Vietjet Air hoặc Bamboo Airways cho anh/chị ạ.",
    ],
    "respond_airfare": [
        "Giá vé từ {fromloc} đến {toloc} dao động từ {price_range} ạ. Anh/chị muốn em tìm chuyến cụ thể không?",
        "Vé máy bay từ {fromloc} đến {toloc} có giá từ {price_range} tùy hãng và thời điểm ạ.",
    ],
    "respond_airport": [
        "Thông tin sân bay đã được tra cứu ạ. Anh/chị cần biết thêm gì không?",
    ],
    "respond_general": [
        "Em đã ghi nhận thông tin của anh/chị. Em có thể giúp gì thêm không ạ?",
        "Vâng ạ, em hiểu rồi. Anh/chị cần hỗ trợ thêm gì không ạ?",
    ],
    "respond_ground_transport": [
        "Em có thể tìm dịch vụ xe đưa đón sân bay cho anh/chị ạ. Anh/chị cần xe đón ở đâu?",
    ],
    # Escalation
    "escalate": [
        "Em sẽ chuyển anh/chị đến nhân viên hỗ trợ. Vui lòng chờ trong giây lát ạ.",
        "Trường hợp của anh/chị cần nhân viên hỗ trợ trực tiếp. Em sẽ chuyển máy ngay ạ.",
        "Xin lỗi anh/chị, em cần chuyển anh/chị đến bộ phận chăm sóc khách hàng để hỗ trợ tốt hơn ạ.",
    ],
    # Goodbye
    "goodbye": [
        "Cảm ơn anh/chị đã sử dụng dịch vụ. Chúc anh/chị có chuyến bay vui vẻ ạ!",
        "Cảm ơn anh/chị! Hẹn gặp lại ạ!",
        "Chúc anh/chị một ngày tốt lành! Hẹn gặp lại ạ!",
    ],
}


@dataclass
class NLGConfig:
    """Configuration for NLG.

    Attributes:
        randomize: Whether to randomly select from multiple templates
        default_duration: Default flight duration for info responses
        default_price_range: Default price range for airfare responses
    """

    randomize: bool = True
    default_duration: str = "1-2 giờ"
    default_price_range: str = "1.500.000 - 3.000.000 VNĐ"


class TemplateNLG:
    """Template-based Natural Language Generator.

    Generates Vietnamese responses by selecting templates and filling
    slots with formatted values.

    Usage:
        nlg = TemplateNLG()
        action = {"type": "request_slot", "slot": "fromloc.city_name"}
        response = nlg.generate(action)
        # "Anh/chị muốn bay từ thành phố nào ạ?"

        action = {"type": "confirm", "params": {"fromloc.city_name": "ha noi", "toloc.city_name": "da nang"}}
        response = nlg.generate(action)
        # "Xác nhận: Chuyến bay từ Hà Nội đến Đà Nẵng. Đúng không ạ?"
    """

    def __init__(self, config: Optional[NLGConfig] = None):
        """Initialize NLG.

        Args:
            config: NLG configuration. Uses defaults if not provided.
        """
        self.config = config or NLGConfig()
        self.templates = TEMPLATES

    def generate(self, action: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> str:
        """Generate Vietnamese response for given action.

        Args:
            action: Action dictionary from policy containing:
                - type: Action type (greet, clarify, request_slot, etc.)
                - slot: Slot to request (for request_slot actions)
                - params: Slot values (for confirm/execute actions)
                - intent: Intent being handled
            state: Optional dialogue state for additional context.

        Returns:
            Generated Vietnamese response string.
        """
        action_type = action.get("type", "")

        # Map action type to template key and generate
        if action_type == "greet":
            return self._select_template("greet")

        elif action_type == "clarify":
            return self._select_template("clarify")

        elif action_type == "request_slot":
            slot = action.get("slot", "")
            template_key = f"request_{slot}"
            # Fall back to generic slot request if specific template not found
            if template_key not in self.templates:
                return self._generate_generic_slot_request(slot)
            return self._select_template(template_key)

        elif action_type == "confirm":
            return self._generate_confirmation(action)

        elif action_type == "execute":
            return self._generate_execution(action)

        elif action_type == "respond":
            return self._generate_response(action)

        elif action_type == "escalate":
            return self._select_template("escalate")

        elif action_type == "goodbye":
            return self._select_template("goodbye")

        else:
            # Unknown action type - return generic response
            return self._select_template("respond_general")

    def _select_template(self, template_key: str) -> str:
        """Select a template for the given key.

        Args:
            template_key: Key to look up in templates.

        Returns:
            Selected template string.
        """
        templates = self.templates.get(template_key, self.templates["respond_general"])
        if self.config.randomize:
            return random.choice(templates)
        return templates[0]

    def _generate_generic_slot_request(self, slot: str) -> str:
        """Generate generic request for unknown slot type.

        Args:
            slot: Slot type being requested.

        Returns:
            Generic request string.
        """
        # Extract readable slot name
        slot_name = slot.replace(".", " ").replace("_", " ")
        return f"Xin cho em biết {slot_name} ạ?"

    def _generate_confirmation(self, action: Dict[str, Any]) -> str:
        """Generate confirmation message with filled slots.

        Args:
            action: Action containing params to confirm.

        Returns:
            Confirmation message.
        """
        params = action.get("params", {})

        # Format locations
        fromloc = self._format_city(
            params.get("fromloc.city_name", params.get("fromloc.airport_name", ""))
        )
        toloc = self._format_city(
            params.get("toloc.city_name", params.get("toloc.airport_name", ""))
        )

        # Format date info
        date_info = self._format_date_info(params)

        # Format airline info
        airline_info = ""
        airline = params.get("airline_name", "")
        if airline:
            airline_info = f", hãng {self._format_airline(airline)}"

        # Select and fill template
        template = self._select_template("confirm")
        return template.format(
            fromloc=fromloc or "[điểm đi]",
            toloc=toloc or "[điểm đến]",
            date_info=date_info,
            airline_info=airline_info,
        )

    def _generate_execution(self, action: Dict[str, Any]) -> str:
        """Generate execution response (booking confirmation).

        Args:
            action: Action containing execution details.

        Returns:
            Execution confirmation message.
        """
        # Generate booking ID
        booking_id = f"VN{uuid.uuid4().hex[:6].upper()}"

        intent = action.get("intent", "flight")
        template_key = f"execute_{intent}" if f"execute_{intent}" in self.templates else "execute_flight"
        template = self._select_template(template_key)

        return template.format(booking_id=booking_id)

    def _generate_response(self, action: Dict[str, Any]) -> str:
        """Generate informational response.

        Args:
            action: Action containing response details.

        Returns:
            Information response message.
        """
        params = action.get("params", {})
        query_type = action.get("query_type", action.get("intent", "general"))

        # Format common values
        fromloc = self._format_city(params.get("fromloc.city_name", ""))
        toloc = self._format_city(params.get("toloc.city_name", ""))

        # Select template based on query type
        template_key = f"respond_{query_type}"
        if template_key not in self.templates:
            template_key = "respond_general"

        template = self._select_template(template_key)

        # Fill template with available values
        try:
            return template.format(
                fromloc=fromloc or "điểm đi",
                toloc=toloc or "điểm đến",
                duration=self.config.default_duration,
                price_range=self.config.default_price_range,
            )
        except KeyError:
            # If template has unfillable placeholders, return as-is
            return template

    def _format_city(self, city: str) -> str:
        """Format city name for display.

        Args:
            city: Raw city name.

        Returns:
            Formatted city name with proper Vietnamese diacritics.
        """
        if not city:
            return ""
        city_lower = city.lower().strip()
        return CITY_DISPLAY_NAMES.get(city_lower, city.title())

    def _format_airline(self, airline: str) -> str:
        """Format airline name for display.

        Args:
            airline: Raw airline name.

        Returns:
            Formatted airline name.
        """
        if not airline:
            return ""
        airline_lower = airline.lower().strip()
        return AIRLINE_DISPLAY_NAMES.get(airline_lower, airline.title())

    def _format_date_info(self, params: Dict[str, str]) -> str:
        """Format date information from params.

        Args:
            params: Slot parameters.

        Returns:
            Formatted date string.
        """
        parts = []

        # Day name
        day_name = params.get("depart_date.day_name", "")
        if day_name:
            day_lower = day_name.lower().strip()
            formatted_day = DAY_DISPLAY_NAMES.get(day_lower, day_name)
            parts.append(formatted_day)

        # Month name
        month_name = params.get("depart_date.month_name", "")
        if month_name:
            month_lower = month_name.lower().strip()
            formatted_month = MONTH_DISPLAY_NAMES.get(month_lower, month_name)
            parts.append(formatted_month)

        # Day number
        day_number = params.get("depart_date.day_number", "")
        if day_number:
            parts.append(f"ngày {day_number}")

        # Today relative
        today_rel = params.get("depart_date.today_relative", "")
        if today_rel:
            parts.append(today_rel)

        if parts:
            return ", " + " ".join(parts)
        return ""


def create_nlg(config: Optional[Dict[str, Any]] = None) -> TemplateNLG:
    """Factory function to create NLG instance.

    Args:
        config: Configuration dictionary.

    Returns:
        Configured TemplateNLG instance.
    """
    if config:
        nlg_config = NLGConfig(
            randomize=config.get("randomize", True),
            default_duration=config.get("default_duration", "1-2 giờ"),
            default_price_range=config.get("default_price_range", "1.500.000 - 3.000.000 VNĐ"),
        )
        return TemplateNLG(config=nlg_config)
    return TemplateNLG()
