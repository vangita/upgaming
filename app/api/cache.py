from collections import OrderedDict
from threading import Lock

from app.api.schemas import AskRequest, AskResponse

CacheKey = tuple[str, str, str]


class AnswerCache:
    """Answers to recent questions, so the same question asked again costs no LLM tokens.

    The free tier allows 200K tokens a day, and demo visitors tend to ask the same example questions.
    The cache lives in the server's memory (lost on restart) and is cleared whenever the documents
    change, because an answer may then be out of date. When full, the least recently used answer goes.
    """

    def __init__(self, max_size: int = 256) -> None:
        self.max_size = max_size
        self._answers: OrderedDict[CacheKey, AskResponse] = OrderedDict()
        # Routes run in a thread pool: two requests can use the cache at the same time
        self._lock = Lock()

    @staticmethod
    def key(request: AskRequest) -> CacheKey:
        # "Who is X?" and " who  is x? " are the same question; the method and search mode are not
        question = " ".join(request.question.split()).casefold()
        return question, request.method, request.search_mode

    def get(self, request: AskRequest) -> AskResponse | None:
        key = self.key(request)
        with self._lock:
            answer = self._answers.get(key)
            if answer is not None:
                self._answers.move_to_end(key)
            return answer

    def put(self, request: AskRequest, answer: AskResponse) -> None:
        if self.max_size <= 0:
            return
        key = self.key(request)
        with self._lock:
            self._answers[key] = answer
            self._answers.move_to_end(key)
            if len(self._answers) > self.max_size:
                self._answers.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._answers.clear()

    def __len__(self) -> int:
        return len(self._answers)
