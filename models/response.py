from typing import TypeVar, Generic, Dict, Any

T = TypeVar("T")


class Response(Generic[T]):
    def __init__(self, message: str, code: int, data: T):
        self.message = message
        self.code = code
        self.data = data

    @property
    def to_response(self) -> Dict[str, Any]:
        return {"message": self.message, "code": self.code, "data": self.data}
