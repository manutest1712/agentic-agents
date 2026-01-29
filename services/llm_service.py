from config.models import OpenAIModel
from config.openai_client import get_openai_client
from config.settings import DEFAULT_MODEL


class LLMService:
    def __init__(self, chat_model: OpenAIModel = DEFAULT_MODEL,
                 embedding_model: str = "text-embedding-3-large"):
        self.client = get_openai_client()
        self.chat_model = chat_model
        self.embedding_model = embedding_model


    def run(
            self,
            prompt: str,
            system_message: str = "You are a helpful assistant"
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return response.choices[0].message.content

    def get_embedding(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding

