from pydantic import BaseModel, Field

class EvaluationReport(BaseModel):
    useful: bool = Field(description="Whether the retrieved information is sufficient")
    score: float = Field(description="Confidence score between 0.0 and 1.0")
    description: str = Field(description="Detailed reasoning")

    @staticmethod
    def parse(raw_output: str):
        import json
        data = json.loads(raw_output)
        return EvaluationReport(**data)
