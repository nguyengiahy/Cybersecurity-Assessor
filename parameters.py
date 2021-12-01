# ---------------------------
# GLOBAL PARAMETERS
# ---------------------------

# all elements within the list must be in lowercase
SENTENCE_BOUNDARY_KEYWORDS = []

# all elements within the list must be in lowercase
SENTENCE_NONE_BOUNDARY_KEYWORDS = []

# APA latin abbreviations are not sentence boundaries
# https://blog.apastyle.org/files/apa-latin-abbreviations-table-2.pdf
NONE_BOUNDARY_ABBREVIATIONS = ["cf", "i.e", "viz", "vs", "e.g", "al", "ibid"]

# Characters that need to be removed.
REMOVE_CHARACTERS = "[\n\t\r*#]"

# The minimum number of tokens to be valid for BERT input
NUM_TOKENS_LOWER_LIMIT = 3

# Size of the multiprocessing.Queues for the pipeline.py class to send new data to the criteria Processes.
# This parameter is not expected to change
INPUT_QUEUE_SIZE = 1

# Total number of working criteria, needed for setting the size of the return data multiprocessing.Queue
NUM_CRITERIA = 6

# ---------------------------
# CRITERIA-SPECIFIC CONSTANTS
# ---------------------------

# C9 - The app collects personal identifiable information (PII).
C9_KEYWORDS = [
    "email",
    "phone",
    "name",
    "contact",
    "address",
    "address book",
    "password",
    "age",
    "date of birth",
    "id",
    "cookie",
    "imei",
    "imsi",
    "ip",
    "mac",
    "sim serial",
    "ssid",
    "bssid",
    "location",
    "location bluetooth",
    "cell tower",
    "gps",
    "wifi location",
    "sso",
    "single sign on",
    "identifier",
    "facebook sso",
    "facebook single sign on",
]

# C10 - The app ensures the right of access to collected information.
C10_KEYWORDS = [
    "access",
    "accessible",
    "right",
    "request",
    "contact",
    ]

# C11 - The app gives information about any kind of agreements with third parties.
C11_KEYWORDS = [
    "agree",
    "agreement",
    "party",
    "contract",
    "affiliate",
    "vendor",
    # TODO: add more keywords
]

# C12 - The app requires users to always give their explicit consent before any action is taken.
C12_KEYWORDS = [
    "consent",
    "opt-out",
    "opt out",
    "opt-in",
    "opt in",
    "control",
    "choice",
]

# C24 - The app gives users control over their data.
C24_KEYWORDS = [
    "control",
    "edit",
    "update",
    "rectify",
    "modify",
    "alter",
    "right",
    "correct",
    "delete",
    "opt in",
    "opt-in",
    "opt out",
    "opt-out",
]

# C49 - The app applies appropriate measures to protect minor users in accordance with applicable legislations.
C49_KEYWORDS = [
    "student",
    "minor",
    "child",
    "13",
    "thirteen",
    "children",
    "age",
]