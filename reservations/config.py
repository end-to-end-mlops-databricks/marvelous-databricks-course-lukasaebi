from pydantic import BaseModel


class DataConfig(BaseModel):
    target: str
    numerical_variables: list[str]
    categorical_variables: list[str]
    test_size: float
    random_state: int
