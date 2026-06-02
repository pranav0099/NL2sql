"""
Generate 600+ NL-SQL training pairs for Hotel Management dataset.
Covers all 10 tables with real column names and values.
Includes AUGMENTED noisy variants (typos, misspellings, bad grammar)
so the model handles real-world messy user input.
"""
import json, random, re, string

# ── Schema ────────────────────────────────────────────────────────────────────
SCHEMA = {
    "staff": {
        "cols": ["staff_id","hotel_id","name","role","department","salary","hire_date","email"],
        "types": ["number","number","text","text","text","number","date","text"],
        "roles": ["Manager","Chef","Receptionist","Housekeeping"],
        "departments": ["Administration","Kitchen","Front Desk","Operations"],
        "names": ["Deepak Kumar","Sneha Patel","Arjun Verma","Amit Singh","Kavita Rao","Rahul Joshi","Meena Iyer","Pooja Desai","Sanjay Gupta","Pradeep Nair"],
        "salaries": [85000,35000,50000,70000,75000,25000,90000,30000,38000,65000],
    },
    "hotels": {
        "cols": ["hotel_id","hotel_name","city","country","star_rating","phone","email","address"],
        "types": ["number","text","text","text","number","text","text","text"],
        "names": ["Grand Palace","Sunrise Inn","Blue Lagoon","The Regal","Palm Breeze"],
        "cities": ["Mumbai","Pune","Goa","Delhi","Chennai"],
        "stars": [5,4,3],
    },
    "guests": {
        "cols": ["guest_id","first_name","last_name","email","phone","nationality","dob","loyalty_points"],
        "types": ["number","text","text","text","text","text","date","number"],
        "nationalities": ["Indian","British","Spanish","Chinese","American"],
        "points": [1200,800,2300,450,650,100,980,200],
    },
    "rooms": {
        "cols": ["room_id","hotel_id","room_number","room_type","price_per_night","floor","max_occupancy","is_available"],
        "types": ["number","number","text","text","number","number","number","number"],
        "types_vals": ["Deluxe","Standard","Suite","Presidential"],
        "prices": [8000,5000,15000,35000],
    },
    "bookings": {
        "cols": ["booking_id","guest_id","room_id","check_in","check_out","status","total_nights","special_requests"],
        "types": ["number","number","number","date","date","text","number","text"],
        "statuses": ["Completed","Active","Upcoming"],
    },
    "payments": {
        "cols": ["payment_id","booking_id","amount","method","payment_date","status","transaction_ref"],
        "types": ["number","number","number","text","number","text","text"],
        "methods": ["Credit Card","Debit Card","UPI","Cash","Bank Transfer"],
        "statuses": ["Paid","Pending"],
        "amounts": [32000,72000,45000,210000,50000,5000,9000,3600],
    },
    "reviews": {
        "cols": ["review_id","booking_id","guest_id","hotel_id","rating","cleanliness","service","food","value","comment","review_date"],
        "types": ["number","number","number","number","number","number","number","number","number","text","date"],
        "ratings": [3,4,5],
    },
    "maintenance_logs": {
        "cols": ["log_id","hotel_id","room_id","issue_type","description","reported_date","resolved_date","status","assigned_to","cost"],
        "types": ["number","number","number","text","text","date","date","text","text","number"],
        "issue_types": ["Plumbing","Electrical","HVAC","Furniture","Housekeeping"],
        "statuses": ["Resolved","Open"],
        "costs": [1500,800,3000,2500,500,700,900,1200,4000,0],
    },
    "amenities": {
        "cols": ["amenity_id","hotel_id","amenity_name","category","is_free","charge_per_use"],
        "types": ["number","number","text","text","number","number"],
        "categories": ["Recreation","Wellness","Fitness","Dining","Facility"],
    },
    "room_service_orders": {
        "cols": ["order_id","booking_id","room_id","item_name","category","quantity","unit_price","total_price","order_time","status"],
        "types": ["number","number","number","text","text","number","number","number","date","text"],
        "categories": ["Food","Beverage"],
        "items": ["Club Sandwich","Butter Chicken","Caesar Salad","Pasta","Breakfast Set","Masala Dosa","Fresh Juice","Tea","Mineral Water","Mocktail"],
    },
}

def make_record(idx, question, sql, table, intent, columns, types):
    nl_tokens = re.sub(r'[^\w\s]','',question.lower()).split()
    sql_tokens = re.sub(r'([>=<!(),])', r' \1 ', sql).split()
    return {
        "id": f"hotel_{idx:05d}",
        "question": question,
        "query": sql,
        "table_id": table,
        "schema": {"table": table, "columns": columns, "types": types},
        "intent": intent,
        "source": "hotel_training",
        "nl_tokens": nl_tokens,
        "sql_tokens": sql_tokens,
        "nl_ids": [],
        "sql_ids": []
    }

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION: Typos, Misspellings, Bad Grammar
# ══════════════════════════════════════════════════════════════════════════════

# Common misspellings for hotel/SQL domain words
MISSPELLINGS = {
    "show":        ["shw", "sho", "shwo", "sohw", "shoe", "showw"],
    "list":        ["lst", "lsit", "lisr", "lits", "lis"],
    "display":     ["dispaly", "displya", "dsiplay", "disply", "displa"],
    "find":        ["fnd", "fined", "fidn", "fin"],
    "all":         ["al", "aall", "alll"],
    "staff":       ["staf", "staaf", "stff", "satff", "staf"],
    "salary":      ["salry", "salar", "slary", "salery", "sallary", "salarry"],
    "hotel":       ["hotl", "hotal", "hotle", "hotell", "hote"],
    "hotels":      ["hotls", "hotals", "hotles", "hotells", "hotes"],
    "guests":      ["guets", "gusets", "geusts", "gests", "gusest"],
    "rooms":       ["roms", "romms", "roomes", "roooms"],
    "bookings":    ["bokings", "bokkings", "bookins", "bookngs", "bookigns"],
    "payments":    ["paymens", "paymnets", "paymets", "payements"],
    "reviews":     ["reveiws", "rveiws", "reviws", "revews", "reviewss"],
    "maintenance": ["maintanance", "maintnance", "maintaince", "maintenace", "maitenance"],
    "amenities":   ["ameneties", "amenitis", "amenites", "ammenities"],
    "average":     ["avrage", "averge", "avarage", "averag", "avergae"],
    "maximum":     ["maxium", "maximun", "maxmum", "maxximum"],
    "minimum":     ["minmum", "minimun", "minium", "mnimum"],
    "total":       ["totl", "toatl", "totla", "toal"],
    "count":       ["cont", "coutn", "ccount", "countt"],
    "highest":     ["highst", "higest", "highets", "heighest"],
    "lowest":      ["lowst", "lowets", "loweset", "lwest"],
    "department":  ["departmnt", "deparment", "depatment", "departement"],
    "employees":   ["employes", "emplyees", "employess", "emplyes"],
    "members":     ["membes", "membrs", "memebers", "membrs"],
    "above":       ["abve", "abov", "aboev", "abov"],
    "below":       ["belw", "belo", "bellow", "beolw"],
    "earning":     ["earining", "earnign", "eaning", "earnig"],
    "earning":     ["earining", "earnign", "eaning", "earnig"],
    "records":     ["recods", "recrds", "recordss", "recors"],
    "information": ["informaton", "infromation", "informtion", "infomation"],
    "customer":    ["custmer", "customar", "customr", "cutomer"],
    "Mumbai":      ["Mubmai", "Mumabi", "mumbai", "Mumbaii"],
    "Delhi":       ["Dehli", "Dellhi", "delhi", "Dlhi"],
    "Pune":        ["Pne", "Punne", "pune"],
    "Chennai":     ["Chenai", "Channai", "chennai", "Chennnai"],
    "Goa":         ["goa", "Goaa"],
    "Manager":     ["Managar", "Managr", "Manger", "manger"],
    "Chef":        ["Cheff", "cheff", "Chf"],
    "Receptionist":["Receptonist", "Recptionist", "Receptinist"],
    "Housekeeping":["Houskeeping", "Housekeepng", "Housekkeping"],
    "Administration":["Administation", "Adminstration", "Administraton"],
    "Kitchen":     ["Kitchn", "Kichen", "Kitcen"],
    "Deluxe":      ["Delux", "Duluxe", "Deleuxe"],
    "Standard":    ["Standrd", "Standar", "Standart"],
    "Suite":       ["Suit", "Suitte", "Sute"],
    "Presidential":["Presidental", "Presidantial", "Presidentail"],
    "Completed":   ["Complted", "Completd", "Compleated"],
    "Active":      ["Actve", "Activee", "Acive"],
    "Upcoming":    ["Upcomming", "Upcomig", "Upcomin"],
    "Plumbing":    ["Plumbin", "Plubming", "Plumbling"],
    "Electrical":  ["Electical", "Electricl", "Eletrical"],
    "Resolved":    ["Resovled", "Resolvd", "Reolved"],
    "nationality":  ["nationlity", "natioanality", "nationaliy"],
    "loyalty":      ["loyality", "loyalti", "loaylty"],
    "points":       ["ponits", "poins", "piints"],
    "available":    ["avialable", "availble", "avilable"],
    "occupied":     ["ocupied", "occuped", "occpied"],
    "expensive":    ["expensve", "expnsive", "expenisve"],
    "cheapest":     ["cheapst", "cheapes", "chepest"],
    "rating":       ["ratng", "raiting", "rateing"],
    "cleanliness":  ["cleanlines", "cleanlness", "cleanliess"],
    "service":      ["servce", "sevice", "servcice"],
    "revenue":      ["revnue", "reveune", "revenu"],
    "order":        ["ordr", "oder", "ordeer"],
    "orders":       ["ordrs", "oders", "orderes"],
    "grouped":      ["gruoped", "groped", "groupd"],
    "number":       ["numbr", "numer", "numbre"],
    "with":         ["wth", "wiht", "witth"],
    "from":         ["frm", "fom", "froom"],
    "than":         ["thn", "tahn", "thna"],
    "more":         ["mroe", "mor", "moer"],
    "less":         ["les", "lees", "lss"],
    "greater":      ["greator", "graeter", "greter"],
    "status":       ["staus", "statsu", "stattus"],
    "paid":         ["piad", "paide"],
    "pending":      ["pendng", "pendin", "pendeing"],
}

# Bad grammar templates — maps a clean pattern to a messy version
# {TABLE}, {COL}, {VAL}, {NUM} are placeholders filled at runtime
BAD_GRAMMAR_TEMPLATES = [
    # Missing articles / wrong word order
    ("Show all {table}",         "show me all the {table}"),
    ("Show all {table}",         "all {table} show"),
    ("Show all {table}",         "give {table} all"),
    ("Show all {table}",         "i want see all {table}"),
    ("Show all {table}",         "plz show {table}"),
    ("Show all {table}",         "can u show {table}"),
    ("Show all {table}",         "show {table} plz"),
    ("List all {table}",         "list me {table}"),
    ("List all {table}",         "all {table} list"),
    # Wrong verb / informal
    ("How many {table}",         "how much {table} is there"),
    ("How many {table}",         "how many {table} is there"),
    ("How many {table}",         "total {table} how many"),
    ("Count all {table}",        "count {table}"),
    ("Count all {table}",        "cnt all {table}"),
    # Salary related
    ("salary above",             "salary more then"),
    ("salary above",             "salary abve"),
    ("salary below",             "salary less then"),
    ("salary below",             "salary belw"),
    ("earning more than",        "earning more then"),
    ("earning less than",        "earning less then"),
    # Common user mistakes
    ("What is the",              "what the"),
    ("What is the",              "whats the"),
    ("What is the",              "wht is the"),
    ("What is the",              "wat is the"),
    ("with salary",              "salary with"),
    ("with salary",              "having salary"),
    ("Show staff",               "shoe staff"),
    ("Show staff",               "shw staff"),
    ("Show staff",               "staff show"),
    ("Show hotels",              "show the hotles"),
    ("Show hotels",              "hotels show me"),
]


def _random_typo(word: str) -> str:
    """Introduce a single random typo into a word."""
    if len(word) <= 2:
        return word
    typo_type = random.choice(["swap", "drop", "double", "wrong"])
    i = random.randint(0, len(word) - 2)

    if typo_type == "swap" and len(word) > 2:
        # Swap two adjacent characters
        w = list(word)
        w[i], w[i+1] = w[i+1], w[i]
        return "".join(w)
    elif typo_type == "drop":
        # Drop a random character
        return word[:i] + word[i+1:]
    elif typo_type == "double":
        # Double a random character
        return word[:i] + word[i] + word[i:]
    elif typo_type == "wrong":
        # Replace with nearby keyboard char
        nearby = {
            'a':'sq','b':'vn','c':'xv','d':'sf','e':'wr','f':'dg',
            'g':'fh','h':'gj','i':'uo','j':'hk','k':'jl','l':'ko',
            'm':'nb','n':'bm','o':'ip','p':'ol','q':'wa','r':'et',
            's':'ad','t':'ry','u':'yi','v':'cb','w':'qe','x':'zc',
            'y':'tu','z':'xa',
        }
        ch = word[i].lower()
        if ch in nearby:
            replacement = random.choice(nearby[ch])
            if word[i].isupper():
                replacement = replacement.upper()
            return word[:i] + replacement + word[i+1:]
    return word


def generate_misspelled_question(question: str) -> str:
    """
    Create a misspelled/typo version of a question.
    Strategy:
    1. First try dictionary-based misspellings for known words.
    2. Then apply random typos to 1-2 remaining words.
    3. Sometimes mess up grammar (drop words, wrong word order).
    """
    words = question.split()
    result = []
    misspelled_count = 0

    for word in words:
        # Strip punctuation for lookup
        clean = word.strip(".,?!;:'\"")

        # Try dictionary misspelling (40% chance per known word)
        if clean in MISSPELLINGS and random.random() < 0.40:
            misspelled = random.choice(MISSPELLINGS[clean])
            result.append(word.replace(clean, misspelled))
            misspelled_count += 1
        # Random typo (15% chance per word, max 2 typos total)
        elif len(clean) > 3 and random.random() < 0.15 and misspelled_count < 2:
            result.append(_random_typo(word))
            misspelled_count += 1
        else:
            result.append(word)

    # Grammar mess-ups (30% chance)
    if random.random() < 0.30:
        strategy = random.choice(["drop_word", "lower_all", "add_filler", "than_then"])

        if strategy == "drop_word" and len(result) > 3:
            # Drop a random non-essential word
            skip_words = {"the", "a", "an", "is", "are", "all", "from", "in", "of"}
            droppable = [i for i, w in enumerate(result)
                         if w.lower() in skip_words]
            if droppable:
                drop_idx = random.choice(droppable)
                result.pop(drop_idx)

        elif strategy == "lower_all":
            result = [w.lower() for w in result]

        elif strategy == "add_filler":
            fillers = ["plz", "pls", "plese", "can u", "i want", "gimme"]
            result.insert(0, random.choice(fillers))

        elif strategy == "than_then":
            result = [w.replace("than", "then").replace("Than", "Then") for w in result]

    return " ".join(result)


def generate_bad_grammar_question(question: str) -> str:
    """
    Create a grammatically broken version of a question.
    Simulates a user who writes broken English.
    """
    transforms = [
        # Wrong verb forms
        lambda q: q.replace("Show all ", "show me the "),
        lambda q: q.replace("List all ", "list me all "),
        lambda q: q.replace("How many ", "how much "),
        lambda q: q.replace("What is the ", "whats "),
        lambda q: q.replace("What is the ", "wat is "),
        lambda q: q.replace("Find all ", "find me "),
        lambda q: q.replace("Display ", "show me "),
        lambda q: q.replace(" above ", " more then "),
        lambda q: q.replace(" below ", " less then "),
        lambda q: q.replace(" greater than ", " more then "),
        lambda q: q.replace(" less than ", " less then "),
        lambda q: q.replace(" more than ", " more then "),
        lambda q: q.replace(" with ", " having "),
        lambda q: q.replace(" with ", " wid "),
        lambda q: q.replace("earning", "who earn"),
        # Informal rewrites
        lambda q: "pls " + q.lower(),
        lambda q: "i want " + q.lower(),
        lambda q: q.lower() + " plz",
        lambda q: "give me " + q.lower().replace("show ", "").replace("list ", ""),
        lambda q: "tell me " + q.lower().replace("show ", "").replace("list ", ""),
        lambda q: q.lower(),  # just lowercase everything
    ]

    transform = random.choice(transforms)
    return transform(question)


records = []
idx = 1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1: STAFF
# ══════════════════════════════════════════════════════════════════════════════
s = SCHEMA["staff"]
sc = s["cols"]; st = s["types"]

# Basic SELECT
for q in ["Show all staff","List all staff members","Display all hotel staff",
          "Get all staff records","Fetch all staff","Show me all employees",
          "Show staff information","List every staff member"]:
    records.append(make_record(idx,q,"SELECT * FROM staff","staff","SELECT",sc,st)); idx+=1

# WHERE = (role)
for role in s["roles"]:
    for q in [f"Show all {role}s",f"List {role}s",f"Find all {role}s",
              f"Show staff with role {role}",f"Display {role} staff"]:
        records.append(make_record(idx,q,f"SELECT * FROM staff WHERE role = '{role}'","staff","SELECT_WHERE",sc,st)); idx+=1

# WHERE = (department)
for dept in s["departments"]:
    for q in [f"Show staff in {dept}",f"List {dept} staff",
              f"Find employees in {dept} department",f"Show {dept} team"]:
        records.append(make_record(idx,q,f"SELECT * FROM staff WHERE department = '{dept}'","staff","SELECT_WHERE",sc,st)); idx+=1

# WHERE > salary
for sal in [30000,40000,50000,60000,70000,80000]:
    for q in [f"Show staff with salary above {sal}",
              f"List employees earning more than {sal}",
              f"Find staff with salary greater than {sal}",
              f"Show staff salary over {sal}"]:
        records.append(make_record(idx,q,f"SELECT * FROM staff WHERE salary > {sal}","staff","SELECT_WHERE",sc,st)); idx+=1

# WHERE < salary
for sal in [30000,40000,50000,60000]:
    for q in [f"Show staff with salary below {sal}",
              f"List employees earning less than {sal}",
              f"Find staff with salary under {sal}"]:
        records.append(make_record(idx,q,f"SELECT * FROM staff WHERE salary < {sal}","staff","SELECT_WHERE",sc,st)); idx+=1

# WHERE >= salary
for sal in [50000,60000,70000]:
    records.append(make_record(idx,f"Show staff with salary at least {sal}",f"SELECT * FROM staff WHERE salary >= {sal}","staff","SELECT_WHERE",sc,st)); idx+=1
    records.append(make_record(idx,f"Show staff earning no less than {sal}",f"SELECT * FROM staff WHERE salary >= {sal}","staff","SELECT_WHERE",sc,st)); idx+=1

# WHERE <= salary
for sal in [40000,50000,60000]:
    records.append(make_record(idx,f"Show staff with salary at most {sal}",f"SELECT * FROM staff WHERE salary <= {sal}","staff","SELECT_WHERE",sc,st)); idx+=1
    records.append(make_record(idx,f"Show staff with salary no more than {sal}",f"SELECT * FROM staff WHERE salary <= {sal}","staff","SELECT_WHERE",sc,st)); idx+=1

# COUNT
for q in ["How many staff are there","Count all staff members","Total number of staff",
          "How many employees work here","Count staff"]:
    records.append(make_record(idx,q,"SELECT COUNT(*) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1

# COUNT with WHERE
records.append(make_record(idx,"How many Managers are there","SELECT COUNT(*) FROM staff WHERE role = 'Manager'","staff","SELECT_AGGREGATE",sc,st)); idx+=1
records.append(make_record(idx,"Count staff in Administration","SELECT COUNT(*) FROM staff WHERE department = 'Administration'","staff","SELECT_AGGREGATE",sc,st)); idx+=1
records.append(make_record(idx,"How many staff earn above 50000","SELECT COUNT(*) FROM staff WHERE salary > 50000","staff","SELECT_AGGREGATE",sc,st)); idx+=1

# AVG
for q in ["What is the average salary","Average salary of all staff",
          "Mean salary of employees","What is the mean salary"]:
    records.append(make_record(idx,q,"SELECT AVG(salary) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1

# MAX/MIN
records.append(make_record(idx,"What is the highest salary","SELECT MAX(salary) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1
records.append(make_record(idx,"Who has the maximum salary","SELECT MAX(salary) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1
records.append(make_record(idx,"What is the lowest salary","SELECT MIN(salary) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1
records.append(make_record(idx,"Minimum salary of staff","SELECT MIN(salary) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1
records.append(make_record(idx,"Total salary of all staff","SELECT SUM(salary) FROM staff","staff","SELECT_AGGREGATE",sc,st)); idx+=1

# ORDER BY
for n in [5,10,3]:
    records.append(make_record(idx,f"Show top {n} highest paid staff",f"SELECT * FROM staff ORDER BY salary DESC LIMIT {n}","staff","SELECT_ORDER",sc,st)); idx+=1
    records.append(make_record(idx,f"Show top {n} staff by salary",f"SELECT * FROM staff ORDER BY salary DESC LIMIT {n}","staff","SELECT_ORDER",sc,st)); idx+=1
    records.append(make_record(idx,f"Show bottom {n} lowest paid staff",f"SELECT * FROM staff ORDER BY salary ASC LIMIT {n}","staff","SELECT_ORDER",sc,st)); idx+=1

records.append(make_record(idx,"Sort staff by salary highest first","SELECT * FROM staff ORDER BY salary DESC","staff","SELECT_ORDER",sc,st)); idx+=1
records.append(make_record(idx,"Sort staff by salary lowest first","SELECT * FROM staff ORDER BY salary ASC","staff","SELECT_ORDER",sc,st)); idx+=1
records.append(make_record(idx,"List staff by name alphabetically","SELECT * FROM staff ORDER BY name ASC","staff","SELECT_ORDER",sc,st)); idx+=1

# GROUP BY
records.append(make_record(idx,"Count staff by department","SELECT department, COUNT(*) FROM staff GROUP BY department","staff","SELECT_GROUP",sc,st)); idx+=1
records.append(make_record(idx,"How many staff in each department","SELECT department, COUNT(*) FROM staff GROUP BY department","staff","SELECT_GROUP",sc,st)); idx+=1
records.append(make_record(idx,"Average salary by department","SELECT department, AVG(salary) FROM staff GROUP BY department","staff","SELECT_GROUP",sc,st)); idx+=1
records.append(make_record(idx,"Total salary per department","SELECT department, SUM(salary) FROM staff GROUP BY department","staff","SELECT_GROUP",sc,st)); idx+=1
records.append(make_record(idx,"Count staff by role","SELECT role, COUNT(*) FROM staff GROUP BY role","staff","SELECT_GROUP",sc,st)); idx+=1
records.append(make_record(idx,"Average salary per role","SELECT role, AVG(salary) FROM staff GROUP BY role","staff","SELECT_GROUP",sc,st)); idx+=1

# SELECT specific columns
records.append(make_record(idx,"Show only staff names and salaries","SELECT name, salary FROM staff","staff","SELECT",sc,st)); idx+=1
records.append(make_record(idx,"Show staff names and roles","SELECT name, role FROM staff","staff","SELECT",sc,st)); idx+=1
records.append(make_record(idx,"List staff emails","SELECT email FROM staff","staff","SELECT",sc,st)); idx+=1
records.append(make_record(idx,"Show only emails of staff","SELECT email FROM staff","staff","SELECT",sc,st)); idx+=1
records.append(make_record(idx,"Get staff names and departments","SELECT name, department FROM staff","staff","SELECT",sc,st)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2: HOTELS
# ══════════════════════════════════════════════════════════════════════════════
h = SCHEMA["hotels"]
hc = h["cols"]; ht = h["types"]

for q in ["Show all hotels","List all hotels","Display hotel information","Get all hotel records"]:
    records.append(make_record(idx,q,"SELECT * FROM hotels","hotels","SELECT",hc,ht)); idx+=1

for city in h["cities"]:
    for q in [f"Show hotels in {city}",f"List hotels from {city}",f"Find hotels located in {city}"]:
        records.append(make_record(idx,q,f"SELECT * FROM hotels WHERE city = '{city}'","hotels","SELECT_WHERE",hc,ht)); idx+=1

for star in h["stars"]:
    records.append(make_record(idx,f"Show {star} star hotels",f"SELECT * FROM hotels WHERE star_rating = {star}","hotels","SELECT_WHERE",hc,ht)); idx+=1
    records.append(make_record(idx,f"List hotels with {star} star rating",f"SELECT * FROM hotels WHERE star_rating = {star}","hotels","SELECT_WHERE",hc,ht)); idx+=1

records.append(make_record(idx,"Show hotels with star rating above 3","SELECT * FROM hotels WHERE star_rating > 3","hotels","SELECT_WHERE",hc,ht)); idx+=1
records.append(make_record(idx,"List luxury hotels with rating above 4","SELECT * FROM hotels WHERE star_rating > 4","hotels","SELECT_WHERE",hc,ht)); idx+=1
records.append(make_record(idx,"How many hotels are there","SELECT COUNT(*) FROM hotels","hotels","SELECT_AGGREGATE",hc,ht)); idx+=1
records.append(make_record(idx,"Count all hotels","SELECT COUNT(*) FROM hotels","hotels","SELECT_AGGREGATE",hc,ht)); idx+=1
records.append(make_record(idx,"Average star rating of hotels","SELECT AVG(star_rating) FROM hotels","hotels","SELECT_AGGREGATE",hc,ht)); idx+=1
records.append(make_record(idx,"Count hotels by city","SELECT city, COUNT(*) FROM hotels GROUP BY city","hotels","SELECT_GROUP",hc,ht)); idx+=1
records.append(make_record(idx,"How many hotels in each city","SELECT city, COUNT(*) FROM hotels GROUP BY city","hotels","SELECT_GROUP",hc,ht)); idx+=1
records.append(make_record(idx,"Show hotel names and cities","SELECT hotel_name, city FROM hotels","hotels","SELECT",hc,ht)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3: BOOKINGS
# ══════════════════════════════════════════════════════════════════════════════
b = SCHEMA["bookings"]
bc = b["cols"]; bt = b["types"]

for q in ["Show all bookings","List all bookings","Display all reservations","Get booking records"]:
    records.append(make_record(idx,q,"SELECT * FROM bookings","bookings","SELECT",bc,bt)); idx+=1

for status in b["statuses"]:
    for q in [f"Show {status.lower()} bookings",f"List all {status} bookings",
              f"Find bookings with status {status}",f"Show bookings that are {status}"]:
        records.append(make_record(idx,q,f"SELECT * FROM bookings WHERE status = '{status}'","bookings","SELECT_WHERE",bc,bt)); idx+=1

for n in [2,3,4,5,6]:
    records.append(make_record(idx,f"Show bookings with {n} nights stay",f"SELECT * FROM bookings WHERE total_nights = {n}","bookings","SELECT_WHERE",bc,bt)); idx+=1

records.append(make_record(idx,"Show bookings with more than 3 nights","SELECT * FROM bookings WHERE total_nights > 3","bookings","SELECT_WHERE",bc,bt)); idx+=1
records.append(make_record(idx,"Show bookings with less than 5 nights","SELECT * FROM bookings WHERE total_nights < 5","bookings","SELECT_WHERE",bc,bt)); idx+=1
records.append(make_record(idx,"How many bookings are there","SELECT COUNT(*) FROM bookings","bookings","SELECT_AGGREGATE",bc,bt)); idx+=1
records.append(make_record(idx,"Count total bookings","SELECT COUNT(*) FROM bookings","bookings","SELECT_AGGREGATE",bc,bt)); idx+=1
records.append(make_record(idx,"How many completed bookings","SELECT COUNT(*) FROM bookings WHERE status = 'Completed'","bookings","SELECT_AGGREGATE",bc,bt)); idx+=1
records.append(make_record(idx,"Count active bookings","SELECT COUNT(*) FROM bookings WHERE status = 'Active'","bookings","SELECT_AGGREGATE",bc,bt)); idx+=1
records.append(make_record(idx,"Average nights per booking","SELECT AVG(total_nights) FROM bookings","bookings","SELECT_AGGREGATE",bc,bt)); idx+=1
records.append(make_record(idx,"Maximum nights in a booking","SELECT MAX(total_nights) FROM bookings","bookings","SELECT_AGGREGATE",bc,bt)); idx+=1
records.append(make_record(idx,"Count bookings by status","SELECT status, COUNT(*) FROM bookings GROUP BY status","bookings","SELECT_GROUP",bc,bt)); idx+=1
records.append(make_record(idx,"How many bookings per status","SELECT status, COUNT(*) FROM bookings GROUP BY status","bookings","SELECT_GROUP",bc,bt)); idx+=1
records.append(make_record(idx,"Total nights per status","SELECT status, SUM(total_nights) FROM bookings GROUP BY status","bookings","SELECT_GROUP",bc,bt)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4: PAYMENTS
# ══════════════════════════════════════════════════════════════════════════════
p = SCHEMA["payments"]
pc = p["cols"]; pt = p["types"]

for q in ["Show all payments","List all payment records","Display payment information"]:
    records.append(make_record(idx,q,"SELECT * FROM payments","payments","SELECT",pc,pt)); idx+=1

for method in p["methods"]:
    for q in [f"Show payments by {method}",f"List {method} payments",
              f"Find payments using {method}"]:
        records.append(make_record(idx,q,f"SELECT * FROM payments WHERE method = '{method}'","payments","SELECT_WHERE",pc,pt)); idx+=1

for status in p["statuses"]:
    for q in [f"Show {status.lower()} payments",f"List payments with status {status}"]:
        records.append(make_record(idx,q,f"SELECT * FROM payments WHERE status = '{status}'","payments","SELECT_WHERE",pc,pt)); idx+=1

for amt in [10000,50000,100000]:
    records.append(make_record(idx,f"Show payments above {amt}",f"SELECT * FROM payments WHERE amount > {amt}","payments","SELECT_WHERE",pc,pt)); idx+=1
    records.append(make_record(idx,f"Show payments below {amt}",f"SELECT * FROM payments WHERE amount < {amt}","payments","SELECT_WHERE",pc,pt)); idx+=1

records.append(make_record(idx,"Total payment amount","SELECT SUM(amount) FROM payments","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"Sum of all payments","SELECT SUM(amount) FROM payments","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"Average payment amount","SELECT AVG(amount) FROM payments","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"Maximum payment amount","SELECT MAX(amount) FROM payments","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"Minimum payment amount","SELECT MIN(amount) FROM payments","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"How many payments are there","SELECT COUNT(*) FROM payments","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"Total amount of paid payments","SELECT SUM(amount) FROM payments WHERE status = 'Paid'","payments","SELECT_AGGREGATE",pc,pt)); idx+=1
records.append(make_record(idx,"Count payments by method","SELECT method, COUNT(*) FROM payments GROUP BY method","payments","SELECT_GROUP",pc,pt)); idx+=1
records.append(make_record(idx,"Total amount by payment method","SELECT method, SUM(amount) FROM payments GROUP BY method","payments","SELECT_GROUP",pc,pt)); idx+=1
records.append(make_record(idx,"Show top 5 highest payments","SELECT * FROM payments ORDER BY amount DESC LIMIT 5","payments","SELECT_ORDER",pc,pt)); idx+=1
records.append(make_record(idx,"Show lowest 3 payments","SELECT * FROM payments ORDER BY amount ASC LIMIT 3","payments","SELECT_ORDER",pc,pt)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5: MAINTENANCE LOGS
# ══════════════════════════════════════════════════════════════════════════════
m = SCHEMA["maintenance_logs"]
mc = m["cols"]; mt = m["types"]

for q in ["Show all maintenance logs","List maintenance issues","Display all maintenance records"]:
    records.append(make_record(idx,q,"SELECT * FROM maintenance_logs","maintenance_logs","SELECT",mc,mt)); idx+=1

for issue in m["issue_types"]:
    for q in [f"Show {issue} issues",f"List {issue} maintenance",
              f"Find maintenance logs for {issue}"]:
        records.append(make_record(idx,q,f"SELECT * FROM maintenance_logs WHERE issue_type = '{issue}'","maintenance_logs","SELECT_WHERE",mc,mt)); idx+=1

for status in m["statuses"]:
    for q in [f"Show {status.lower()} maintenance",f"List maintenance with status {status}",
              f"Find {status} maintenance issues"]:
        records.append(make_record(idx,q,f"SELECT * FROM maintenance_logs WHERE status = '{status}'","maintenance_logs","SELECT_WHERE",mc,mt)); idx+=1

for cost in [1000,2000,3000]:
    records.append(make_record(idx,f"Show maintenance with cost above {cost}",f"SELECT * FROM maintenance_logs WHERE cost > {cost}","maintenance_logs","SELECT_WHERE",mc,mt)); idx+=1
    records.append(make_record(idx,f"Show maintenance with cost below {cost}",f"SELECT * FROM maintenance_logs WHERE cost < {cost}","maintenance_logs","SELECT_WHERE",mc,mt)); idx+=1

records.append(make_record(idx,"How many maintenance issues are there","SELECT COUNT(*) FROM maintenance_logs","maintenance_logs","SELECT_AGGREGATE",mc,mt)); idx+=1
records.append(make_record(idx,"Total maintenance cost","SELECT SUM(cost) FROM maintenance_logs","maintenance_logs","SELECT_AGGREGATE",mc,mt)); idx+=1
records.append(make_record(idx,"Average maintenance cost","SELECT AVG(cost) FROM maintenance_logs","maintenance_logs","SELECT_AGGREGATE",mc,mt)); idx+=1
records.append(make_record(idx,"Maximum maintenance cost","SELECT MAX(cost) FROM maintenance_logs","maintenance_logs","SELECT_AGGREGATE",mc,mt)); idx+=1
records.append(make_record(idx,"Count open maintenance issues","SELECT COUNT(*) FROM maintenance_logs WHERE status = 'Open'","maintenance_logs","SELECT_AGGREGATE",mc,mt)); idx+=1
records.append(make_record(idx,"Total cost of resolved issues","SELECT SUM(cost) FROM maintenance_logs WHERE status = 'Resolved'","maintenance_logs","SELECT_AGGREGATE",mc,mt)); idx+=1
records.append(make_record(idx,"Count maintenance by issue type","SELECT issue_type, COUNT(*) FROM maintenance_logs GROUP BY issue_type","maintenance_logs","SELECT_GROUP",mc,mt)); idx+=1
records.append(make_record(idx,"Total cost by issue type","SELECT issue_type, SUM(cost) FROM maintenance_logs GROUP BY issue_type","maintenance_logs","SELECT_GROUP",mc,mt)); idx+=1
records.append(make_record(idx,"Show top 5 most expensive maintenance","SELECT * FROM maintenance_logs ORDER BY cost DESC LIMIT 5","maintenance_logs","SELECT_ORDER",mc,mt)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6: GUESTS
# ══════════════════════════════════════════════════════════════════════════════
g = SCHEMA["guests"]
gc = g["cols"]; gtype = g["types"]

for q in ["Show all guests","List all guests","Display guest information","Get all guest records"]:
    records.append(make_record(idx,q,"SELECT * FROM guests","guests","SELECT",gc,gtype)); idx+=1

for nat in g["nationalities"]:
    for q in [f"Show {nat} guests",f"List guests from {nat}",f"Find {nat} nationality guests"]:
        records.append(make_record(idx,q,f"SELECT * FROM guests WHERE nationality = '{nat}'","guests","SELECT_WHERE",gc,gtype)); idx+=1

for pts in [500,1000,1500,2000]:
    records.append(make_record(idx,f"Show guests with more than {pts} loyalty points",f"SELECT * FROM guests WHERE loyalty_points > {pts}","guests","SELECT_WHERE",gc,gtype)); idx+=1
    records.append(make_record(idx,f"Find guests with loyalty points above {pts}",f"SELECT * FROM guests WHERE loyalty_points > {pts}","guests","SELECT_WHERE",gc,gtype)); idx+=1

records.append(make_record(idx,"How many guests are there","SELECT COUNT(*) FROM guests","guests","SELECT_AGGREGATE",gc,gtype)); idx+=1
records.append(make_record(idx,"Average loyalty points of guests","SELECT AVG(loyalty_points) FROM guests","guests","SELECT_AGGREGATE",gc,gtype)); idx+=1
records.append(make_record(idx,"Maximum loyalty points","SELECT MAX(loyalty_points) FROM guests","guests","SELECT_AGGREGATE",gc,gtype)); idx+=1
records.append(make_record(idx,"Total loyalty points","SELECT SUM(loyalty_points) FROM guests","guests","SELECT_AGGREGATE",gc,gtype)); idx+=1
records.append(make_record(idx,"Count guests by nationality","SELECT nationality, COUNT(*) FROM guests GROUP BY nationality","guests","SELECT_GROUP",gc,gtype)); idx+=1
records.append(make_record(idx,"Show top 3 guests by loyalty points","SELECT * FROM guests ORDER BY loyalty_points DESC LIMIT 3","guests","SELECT_ORDER",gc,gtype)); idx+=1
records.append(make_record(idx,"Show guest names and emails","SELECT first_name, last_name, email FROM guests","guests","SELECT",gc,gtype)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 7: ROOMS
# ══════════════════════════════════════════════════════════════════════════════
r = SCHEMA["rooms"]
rc = r["cols"]; rtype = r["types"]

for q in ["Show all rooms","List all rooms","Display room information"]:
    records.append(make_record(idx,q,"SELECT * FROM rooms","rooms","SELECT",rc,rtype)); idx+=1

for rtype_val in r["types_vals"]:
    for q in [f"Show {rtype_val} rooms",f"List all {rtype_val} rooms",
              f"Find {rtype_val} room type"]:
        records.append(make_record(idx,q,f"SELECT * FROM rooms WHERE room_type = '{rtype_val}'","rooms","SELECT_WHERE",rc,rtype)); idx+=1

records.append(make_record(idx,"Show available rooms","SELECT * FROM rooms WHERE is_available = 1","rooms","SELECT_WHERE",rc,rtype)); idx+=1
records.append(make_record(idx,"List rooms that are available","SELECT * FROM rooms WHERE is_available = 1","rooms","SELECT_WHERE",rc,rtype)); idx+=1
records.append(make_record(idx,"Show occupied rooms","SELECT * FROM rooms WHERE is_available = 0","rooms","SELECT_WHERE",rc,rtype)); idx+=1

for price in [5000,8000,10000,20000]:
    records.append(make_record(idx,f"Show rooms with price above {price}",f"SELECT * FROM rooms WHERE price_per_night > {price}","rooms","SELECT_WHERE",rc,rtype)); idx+=1
    records.append(make_record(idx,f"Show rooms with price below {price}",f"SELECT * FROM rooms WHERE price_per_night < {price}","rooms","SELECT_WHERE",rc,rtype)); idx+=1

records.append(make_record(idx,"How many rooms are available","SELECT COUNT(*) FROM rooms WHERE is_available = 1","rooms","SELECT_AGGREGATE",rc,rtype)); idx+=1
records.append(make_record(idx,"Count total rooms","SELECT COUNT(*) FROM rooms","rooms","SELECT_AGGREGATE",rc,rtype)); idx+=1
records.append(make_record(idx,"Average room price","SELECT AVG(price_per_night) FROM rooms","rooms","SELECT_AGGREGATE",rc,rtype)); idx+=1
records.append(make_record(idx,"Most expensive room price","SELECT MAX(price_per_night) FROM rooms","rooms","SELECT_AGGREGATE",rc,rtype)); idx+=1
records.append(make_record(idx,"Cheapest room price","SELECT MIN(price_per_night) FROM rooms","rooms","SELECT_AGGREGATE",rc,rtype)); idx+=1
records.append(make_record(idx,"Count rooms by room type","SELECT room_type, COUNT(*) FROM rooms GROUP BY room_type","rooms","SELECT_GROUP",rc,rtype)); idx+=1
records.append(make_record(idx,"Average price by room type","SELECT room_type, AVG(price_per_night) FROM rooms GROUP BY room_type","rooms","SELECT_GROUP",rc,rtype)); idx+=1
records.append(make_record(idx,"Show top 5 most expensive rooms","SELECT * FROM rooms ORDER BY price_per_night DESC LIMIT 5","rooms","SELECT_ORDER",rc,rtype)); idx+=1
records.append(make_record(idx,"Show cheapest 5 rooms","SELECT * FROM rooms ORDER BY price_per_night ASC LIMIT 5","rooms","SELECT_ORDER",rc,rtype)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 8: REVIEWS
# ══════════════════════════════════════════════════════════════════════════════
rv = SCHEMA["reviews"]
rvc = rv["cols"]; rvt = rv["types"]

for q in ["Show all reviews","List all guest reviews","Display review records"]:
    records.append(make_record(idx,q,"SELECT * FROM reviews","reviews","SELECT",rvc,rvt)); idx+=1

for rating in rv["ratings"]:
    for q in [f"Show {rating} star reviews",f"List reviews with rating {rating}",
              f"Find reviews rated {rating}"]:
        records.append(make_record(idx,q,f"SELECT * FROM reviews WHERE rating = {rating}","reviews","SELECT_WHERE",rvc,rvt)); idx+=1

records.append(make_record(idx,"Show reviews with rating above 3","SELECT * FROM reviews WHERE rating > 3","reviews","SELECT_WHERE",rvc,rvt)); idx+=1
records.append(make_record(idx,"Show reviews with high rating above 4","SELECT * FROM reviews WHERE rating > 4","reviews","SELECT_WHERE",rvc,rvt)); idx+=1
records.append(make_record(idx,"How many reviews are there","SELECT COUNT(*) FROM reviews","reviews","SELECT_AGGREGATE",rvc,rvt)); idx+=1
records.append(make_record(idx,"Average rating of all reviews","SELECT AVG(rating) FROM reviews","reviews","SELECT_AGGREGATE",rvc,rvt)); idx+=1
records.append(make_record(idx,"Average cleanliness rating","SELECT AVG(cleanliness) FROM reviews","reviews","SELECT_AGGREGATE",rvc,rvt)); idx+=1
records.append(make_record(idx,"Average service rating","SELECT AVG(service) FROM reviews","reviews","SELECT_AGGREGATE",rvc,rvt)); idx+=1
records.append(make_record(idx,"Highest rating given","SELECT MAX(rating) FROM reviews","reviews","SELECT_AGGREGATE",rvc,rvt)); idx+=1
records.append(make_record(idx,"Count reviews by rating","SELECT rating, COUNT(*) FROM reviews GROUP BY rating","reviews","SELECT_GROUP",rvc,rvt)); idx+=1
records.append(make_record(idx,"Show top 5 reviews by rating","SELECT * FROM reviews ORDER BY rating DESC LIMIT 5","reviews","SELECT_ORDER",rvc,rvt)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 9: AMENITIES
# ══════════════════════════════════════════════════════════════════════════════
a = SCHEMA["amenities"]
ac = a["cols"]; at_types = a["types"]

for q in ["Show all amenities","List all hotel amenities","Display amenity information"]:
    records.append(make_record(idx,q,"SELECT * FROM amenities","amenities","SELECT",ac,at_types)); idx+=1

for cat in a["categories"]:
    records.append(make_record(idx,f"Show {cat} amenities",f"SELECT * FROM amenities WHERE category = '{cat}'","amenities","SELECT_WHERE",ac,at_types)); idx+=1
    records.append(make_record(idx,f"List {cat.lower()} amenities",f"SELECT * FROM amenities WHERE category = '{cat}'","amenities","SELECT_WHERE",ac,at_types)); idx+=1

records.append(make_record(idx,"Show free amenities","SELECT * FROM amenities WHERE is_free = 1","amenities","SELECT_WHERE",ac,at_types)); idx+=1
records.append(make_record(idx,"List amenities that are free","SELECT * FROM amenities WHERE is_free = 1","amenities","SELECT_WHERE",ac,at_types)); idx+=1
records.append(make_record(idx,"Show paid amenities","SELECT * FROM amenities WHERE is_free = 0","amenities","SELECT_WHERE",ac,at_types)); idx+=1
records.append(make_record(idx,"How many amenities are there","SELECT COUNT(*) FROM amenities","amenities","SELECT_AGGREGATE",ac,at_types)); idx+=1
records.append(make_record(idx,"Count free amenities","SELECT COUNT(*) FROM amenities WHERE is_free = 1","amenities","SELECT_AGGREGATE",ac,at_types)); idx+=1
records.append(make_record(idx,"Average charge per amenity","SELECT AVG(charge_per_use) FROM amenities","amenities","SELECT_AGGREGATE",ac,at_types)); idx+=1
records.append(make_record(idx,"Count amenities by category","SELECT category, COUNT(*) FROM amenities GROUP BY category","amenities","SELECT_GROUP",ac,at_types)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 10: ROOM SERVICE ORDERS
# ══════════════════════════════════════════════════════════════════════════════
rs = SCHEMA["room_service_orders"]
rsc = rs["cols"]; rst = rs["types"]

for q in ["Show all room service orders","List room service orders","Display room service records"]:
    records.append(make_record(idx,q,"SELECT * FROM room_service_orders","room_service_orders","SELECT",rsc,rst)); idx+=1

for cat in rs["categories"]:
    records.append(make_record(idx,f"Show {cat.lower()} room service orders",f"SELECT * FROM room_service_orders WHERE category = '{cat}'","room_service_orders","SELECT_WHERE",rsc,rst)); idx+=1

for item in rs["items"][:5]:
    records.append(make_record(idx,f"Show orders for {item}",f"SELECT * FROM room_service_orders WHERE item_name = '{item}'","room_service_orders","SELECT_WHERE",rsc,rst)); idx+=1

records.append(make_record(idx,"How many room service orders","SELECT COUNT(*) FROM room_service_orders","room_service_orders","SELECT_AGGREGATE",rsc,rst)); idx+=1
records.append(make_record(idx,"Total revenue from room service","SELECT SUM(total_price) FROM room_service_orders","room_service_orders","SELECT_AGGREGATE",rsc,rst)); idx+=1
records.append(make_record(idx,"Average order value","SELECT AVG(total_price) FROM room_service_orders","room_service_orders","SELECT_AGGREGATE",rsc,rst)); idx+=1
records.append(make_record(idx,"Most expensive room service order","SELECT MAX(total_price) FROM room_service_orders","room_service_orders","SELECT_AGGREGATE",rsc,rst)); idx+=1
records.append(make_record(idx,"Count orders by category","SELECT category, COUNT(*) FROM room_service_orders GROUP BY category","room_service_orders","SELECT_GROUP",rsc,rst)); idx+=1
records.append(make_record(idx,"Total revenue by item","SELECT item_name, SUM(total_price) FROM room_service_orders GROUP BY item_name","room_service_orders","SELECT_GROUP",rsc,rst)); idx+=1
records.append(make_record(idx,"Show top 5 highest value orders","SELECT * FROM room_service_orders ORDER BY total_price DESC LIMIT 5","room_service_orders","SELECT_ORDER",rsc,rst)); idx+=1

# ══════════════════════════════════════════════════════════════════════════════
# JOIN, HAVING, SUBQUERY, and TOP-N
# ══════════════════════════════════════════════════════════════════════════════

# JOINS
join_pairs = [
    ("Show staff name and their hotel name", "SELECT staff.name, hotels.hotel_name FROM staff JOIN hotels ON staff.hotel_id = hotels.hotel_id", "staff"),
    ("List all employees with their hotel names", "SELECT staff.name, hotels.hotel_name FROM staff JOIN hotels ON staff.hotel_id = hotels.hotel_id", "staff"),
    ("Show bookings along with guest details", "SELECT bookings.booking_id, guests.first_name, guests.last_name FROM bookings JOIN guests ON bookings.guest_id = guests.guest_id", "bookings"),
    ("Find reviews for hotels in Mumbai", "SELECT reviews.comment, hotels.hotel_name FROM reviews JOIN hotels ON reviews.hotel_id = hotels.hotel_id WHERE hotels.city = 'Mumbai'", "reviews"),
    ("Get room details and hotel names for all rooms", "SELECT rooms.room_number, rooms.room_type, hotels.hotel_name FROM rooms JOIN hotels ON rooms.hotel_id = hotels.hotel_id", "rooms"),
    ("Show total payments for Grand Palace hotel", "SELECT SUM(payments.amount) FROM payments JOIN bookings ON payments.booking_id = bookings.booking_id JOIN rooms ON bookings.room_id = rooms.room_id JOIN hotels ON rooms.hotel_id = hotels.hotel_id WHERE hotels.hotel_name = 'Grand Palace'", "payments"),
    ("List amenities available at Palm Breeze", "SELECT amenities.amenity_name FROM amenities JOIN hotels ON amenities.hotel_id = hotels.hotel_id WHERE hotels.hotel_name = 'Palm Breeze'", "amenities"),
    ("Show room service orders with room details", "SELECT room_service_orders.item_name, rooms.room_number FROM room_service_orders JOIN rooms ON room_service_orders.room_id = rooms.room_id", "room_service_orders")
]

for q, sql, tbl in join_pairs:
    cols = SCHEMA[tbl]["cols"]
    types = SCHEMA[tbl]["types"]
    records.append(make_record(idx, q, sql, tbl, "SELECT_JOIN", cols, types))
    idx += 1

# HAVING
having_pairs = [
    ("Show departments with average salary above 50000", "SELECT department, AVG(salary) FROM staff GROUP BY department HAVING AVG(salary) > 50000", "staff"),
    ("List departments where mean salary is greater than 60000", "SELECT department, AVG(salary) FROM staff GROUP BY department HAVING AVG(salary) > 60000", "staff"),
    ("Find roles with total salary more than 100000", "SELECT role, SUM(salary) FROM staff GROUP BY role HAVING SUM(salary) > 100000", "staff"),
    ("Show room types with average price above 10000", "SELECT room_type, AVG(price_per_night) FROM rooms GROUP BY room_type HAVING AVG(price_per_night) > 10000", "rooms"),
    ("List cities with more than 2 hotels", "SELECT city, COUNT(*) FROM hotels GROUP BY city HAVING COUNT(*) > 2", "hotels")
]

for q, sql, tbl in having_pairs:
    cols = SCHEMA[tbl]["cols"]
    types = SCHEMA[tbl]["types"]
    records.append(make_record(idx, q, sql, tbl, "SELECT_GROUP", cols, types))
    idx += 1

# SUBQUERY
subquery_pairs = [
    ("Show staff earning more than the average salary", "SELECT * FROM staff WHERE salary > (SELECT AVG(salary) FROM staff)", "staff"),
    ("List employees with salary above the mean salary", "SELECT * FROM staff WHERE salary > (SELECT AVG(salary) FROM staff)", "staff"),
    ("Find rooms that cost more than the average price", "SELECT * FROM rooms WHERE price_per_night > (SELECT AVG(price_per_night) FROM rooms)", "rooms"),
    ("Show payments greater than the average payment amount", "SELECT * FROM payments WHERE amount > (SELECT AVG(amount) FROM payments)", "payments"),
    ("List staff who earn more than the average Manager salary", "SELECT * FROM staff WHERE salary > (SELECT AVG(salary) FROM staff WHERE role = 'Manager')", "staff")
]

for q, sql, tbl in subquery_pairs:
    cols = SCHEMA[tbl]["cols"]
    types = SCHEMA[tbl]["types"]
    records.append(make_record(idx, q, sql, tbl, "COMPLEX", cols, types))
    idx += 1

# TOP-N (already has some, but let's add explicitly classified ones)
topn_pairs = [
    ("Show top 5 highest paid staff", "SELECT * FROM staff ORDER BY salary DESC LIMIT 5", "staff"),
    ("Show top 3 most expensive rooms", "SELECT * FROM rooms ORDER BY price_per_night DESC LIMIT 3", "rooms"),
    ("Show top 10 guests by loyalty points", "SELECT * FROM guests ORDER BY loyalty_points DESC LIMIT 10", "guests")
]

for q, sql, tbl in topn_pairs:
    cols = SCHEMA[tbl]["cols"]
    types = SCHEMA[tbl]["types"]
    records.append(make_record(idx, q, sql, tbl, "SELECT_LIMIT", cols, types))
    idx += 1

# ══════════════════════════════════════════════════════════════════════════════
# AUGMENTATION: Generate noisy variants of all clean records
# Each clean question gets 2-3 noisy variants with the SAME correct SQL
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nClean records: {len(records)}")
print("Generating noisy augmented variants...")

augmented = []
for rec in records:
    clean_q = rec["question"]
    sql     = rec["query"]
    table   = rec["table_id"]
    intent  = rec["intent"]
    cols    = rec["schema"]["columns"]
    types   = rec["schema"]["types"]

    # Generate 2-3 noisy variants per clean record
    num_variants = random.randint(2, 3)
    seen_variants = {clean_q.lower()}  # avoid duplicates

    for _ in range(num_variants):
        # Alternate between misspelling and bad grammar
        if random.random() < 0.6:
            noisy_q = generate_misspelled_question(clean_q)
        else:
            noisy_q = generate_bad_grammar_question(clean_q)

        # Skip if it came out identical to the original or a previous variant
        if noisy_q.lower() in seen_variants:
            continue
        seen_variants.add(noisy_q.lower())

        augmented.append(make_record(
            idx, noisy_q, sql, table, intent, cols, types
        ))
        idx += 1

print(f"Augmented records: {len(augmented)}")

# Merge clean + augmented
records.extend(augmented)
print(f"Total records (clean + augmented): {len(records)}")


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUT
# ══════════════════════════════════════════════════════════════════════════════
import os, sys
from pathlib import Path

# Output to project data directory
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.shuffle(records)
n = len(records)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)

splits = {
    "train": records[:n_train],
    "val":   records[n_train:n_train+n_val],
    "test":  records[n_train+n_val:]
}

for split, data in splits.items():
    out_path = OUTPUT_DIR / f"hotel_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(data)} records -> {out_path}")

# Stats
from collections import Counter
intents = Counter(r["intent"] for r in records)
tables  = Counter(r["table_id"] for r in records)
sources = Counter(
    "augmented" if r["question"] != r["question"] else "clean"
    for r in records
)

print(f"\n{'='*55}")
print(f"Hotel Training Data Generated (with Augmentation)")
print(f"{'='*55}")
print(f"Total records : {n}")
print(f"  Clean       : {n - len(augmented)}")
print(f"  Augmented   : {len(augmented)}")
print(f"Train         : {n_train}")
print(f"Val           : {n-n_train-len(splits['test'])}")
print(f"Test          : {len(splits['test'])}")
print(f"\nIntent distribution:")
for intent, count in sorted(intents.items(), key=lambda x:-x[1]):
    print(f"  {intent:<25} {count:>4}")
print(f"\nTable distribution:")
for table, count in sorted(tables.items(), key=lambda x:-x[1]):
    print(f"  {table:<30} {count:>4}")

# Show some augmented examples
print(f"\n{'='*55}")
print(f"Sample augmented records (noisy -> correct SQL):")
print(f"{'='*55}")
for sample in augmented[:10]:
    print(f"  Q: {sample['question']}")
    print(f"  -> {sample['query']}")
    print()

print(f"Files saved in: {OUTPUT_DIR}")
