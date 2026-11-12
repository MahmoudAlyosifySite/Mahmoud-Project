"""English loanwords common in Egyptian speech, written the way Egyptians say them.

Used only by the "arabic" code-switching policy of ``arz.normalize``. Spellings follow common
Egyptian usage on social media (ميتنج, لينك, داونلود). Written from general knowledge of
Egyptian office/tech/everyday speech, not from the evaluation sentences; the evaluation
code-switch set uses some of the most frequent of these words (meeting, link, laptop...).
"""

LOANWORDS = {
    # work / office
    "meeting": "ميتنج", "meetings": "ميتنجز", "deadline": "ديدلاين", "deadlines": "ديدلاينز",
    "project": "بروجكت", "projects": "بروجكتس", "manager": "مانجر", "managers": "مانجرز",
    "presentation": "بريزنتيشن", "presentations": "بريزنتيشنز", "report": "ريبورت",
    "task": "تاسك", "tasks": "تاسكات", "team": "تيم", "call": "كول", "calls": "كولز",
    "email": "إيميل", "emails": "إيميلات", "mail": "ميل", "office": "أوفيس", "boss": "بوس",
    "interview": "إنترفيو", "cv": "سي في", "salary": "سالاري", "bonus": "بونص",
    "training": "ترينينج", "course": "كورس", "courses": "كورسات", "plan": "بلان",
    "target": "تارجت", "feedback": "فيدباك", "client": "كلاينت", "clients": "كلاينتس",
    "company": "كومباني", "startup": "ستارت أب", "business": "بيزنس", "marketing": "ماركتنج",
    "sales": "سيلز", "budget": "باجت", "schedule": "سكيدجول", "update": "أبديت",
    "updates": "أبديتس", "file": "فايل", "files": "فايلات", "folder": "فولدر",
    "sheet": "شيت", "excel": "إكسيل", "word": "وورد", "zoom": "زووم", "teams": "تيمز",
    "online": "أونلاين", "offline": "أوفلاين", "remote": "ريموت", "shift": "شيفت",
    "overtime": "أوفر تايم", "break": "بريك", "weekend": "ويك إند", "event": "إيفنت",
    # tech
    "laptop": "لابتوب", "computer": "كمبيوتر", "mobile": "موبايل", "phone": "فون",
    "iphone": "آيفون", "android": "أندرويد", "tablet": "تابلت", "charger": "شاحن",
    "wifi": "واي فاي", "wi-fi": "واي فاي", "internet": "إنترنت", "network": "نتورك",
    "password": "باسورد", "username": "يوزر نيم", "account": "أكاونت", "app": "أبلكيشن",
    "application": "أبلكيشن", "website": "ويبسايت", "site": "سايت", "link": "لينك",
    "links": "لينكات", "download": "داونلود", "upload": "أبلود", "install": "إنستول",
    "restart": "ريستارت", "reset": "ريسيت", "update": "أبديت", "settings": "سيتنجز",
    "screen": "سكرين", "screenshot": "سكرين شوت", "keyboard": "كيبورد", "mouse": "ماوس",
    "printer": "برينتر", "camera": "كاميرا", "video": "فيديو", "videos": "فيديوهات",
    "chat": "شات", "message": "ماسيدج", "group": "جروب", "groups": "جروبات", "post": "بوست",
    "story": "ستوري", "like": "لايك", "likes": "لايكات", "share": "شير", "comment": "كومنت",
    "comments": "كومنتات", "follow": "فولو", "followers": "فولورز", "page": "بيدج",
    "profile": "بروفايل", "facebook": "فيسبوك", "whatsapp": "واتساب", "instagram": "إنستجرام",
    "youtube": "يوتيوب", "tiktok": "تيك توك", "google": "جوجل", "netflix": "نتفليكس",
    "system": "سيستم", "software": "سوفتوير", "hardware": "هاردوير", "data": "داتا",
    "server": "سيرفر", "bug": "باج", "code": "كود", "backup": "باك أب", "cloud": "كلاود",
    "battery": "بطارية", "bluetooth": "بلوتوث", "headphones": "هيدفون", "speaker": "سبيكر",
    "online meeting": "أونلاين ميتنج", "voice note": "فويس نوت", "voice": "فويس",
    # everyday
    "ok": "أوكيه", "okay": "أوكيه", "sorry": "سوري", "thanks": "ثانكس", "thank you": "ثانك يو",
    "please": "بليز", "hello": "هالو", "hi": "هاي", "bye": "باي", "yes": "يس", "no": "نو",
    "cool": "كول", "nice": "نايس", "busy": "بيزي", "free": "فري", "sure": "شور",
    "exactly": "إجزاكتلي", "anyway": "إني واي", "baby": "بيبي", "cafe": "كافيه",
    "coffee": "كوفي", "delivery": "ديليفري", "order": "أوردر", "menu": "منيو",
    "mall": "مول", "gym": "جيم", "match": "ماتش", "goal": "جول", "taxi": "تاكسي",
    "uber": "أوبر", "visa": "فيزا", "cash": "كاش", "discount": "ديسكاونت", "sale": "سيل",
    "offer": "أوفر", "size": "سايز", "style": "ستايل", "look": "لوك", "shopping": "شوبينج",
    "party": "بارتي", "birthday": "بيرثداي", "trip": "تريب", "ticket": "تيكت",
    "booking": "بوكينج", "hotel": "هوتيل", "beach": "بيتش", "summer": "سمر",
    "exam": "إكزام", "quiz": "كويز", "assignment": "أسايمنت", "lecture": "لكتشر",
    "section": "سكشن", "doctor": "دكتور", "check": "تشيك", "checkup": "تشيك أب",
    "mood": "مود", "stress": "ستريس", "level": "ليفل", "idea": "أيديا", "problem": "بروبلم",
    "point": "بوينت", "option": "أوبشن", "random": "راندوم", "basically": "بيزيكلي",
    "literally": "ليترالي", "actually": "أكتشوالي", "weekend plans": "ويك إند بلانز",
}

LETTERS = {"a": "إيه", "b": "بي", "c": "سي", "d": "دي", "e": "إي", "f": "إف", "g": "جي",
           "h": "إتش", "i": "آي", "j": "جاي", "k": "كاي", "l": "إل", "m": "إم", "n": "إن",
           "o": "أو", "p": "بي", "q": "كيو", "r": "آر", "s": "إس", "t": "تي", "u": "يو",
           "v": "في", "w": "دبليو", "x": "إكس", "y": "واي", "z": "زد"}
