"""Constants shared across the SetSwitch codebase."""

SETSWITCH_SPECIAL_TOKENS = [
    "<set>",
    "</set>",
    "<item>",
    "</item>",
    "<read_0>",
    "<read_1>",
    "<read_2>",
    "<read_3>",
    "<read_4>",
    "<read_5>",
    "<read_6>",
    "<read_7>",
    "<gather_0>",
    "<gather_1>",
    "<gather_2>",
    "<gather_3>",
    "<gather_4>",
    "<gather_5>",
    "<gather_6>",
    "<gather_7>",
]

SET_TOKEN = "<set>"
END_SET_TOKEN = "</set>"
ITEM_TOKEN = "<item>"
END_ITEM_TOKEN = "</item>"

READ_TOKENS = [f"<read_{idx}>" for idx in range(8)]
GATHER_TOKENS = [f"<gather_{idx}>" for idx in range(8)]

DEFAULT_INSTRUCTION = "Use the provided documents to answer the question."
DEFAULT_NUM_READS_PER_DOC = 2
DEFAULT_NUM_GATHER_TOKENS = 4

ROLE_PREFIX = 0
ROLE_SET_SPECIAL = 1
ROLE_ITEM_SPECIAL = 2
ROLE_DOC = 3
ROLE_READ = 4
ROLE_GATHER = 5
ROLE_ANSWER = 6
ROLE_PAD = 7

ROLE_NAMES = {
    ROLE_PREFIX: "prefix",
    ROLE_SET_SPECIAL: "set_special",
    ROLE_ITEM_SPECIAL: "item_special",
    ROLE_DOC: "doc",
    ROLE_READ: "read",
    ROLE_GATHER: "gather",
    ROLE_ANSWER: "answer",
    ROLE_PAD: "pad",
}

DOC_CAUSAL = "doc_causal"
DOC_BIDIR = "doc_bidir"
DOC_ATTENTION_MODES = {DOC_CAUSAL, DOC_BIDIR}

IGNORE_INDEX = -100
BLOCKED_ATTENTION_VALUE = -1.0e4
