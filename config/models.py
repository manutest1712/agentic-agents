from enum import Enum

class OpenAIModel(str, Enum):
    GPT_41 = "gpt-4.1"
    GPT_41_MINI = "gpt-4.1-mini"
    GPT_41_NANO = "gpt-4.1-nano"
    GPT_35_TURBO ="gpt-3.5-turbo"
