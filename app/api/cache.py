import time
from collections import OrderedDict
from threading import Lock

from app.api.schemas import AskRequest, AskResponse

CacheKey = tuple[str, str, str]


class AnswerCache:
    """Answers to recent questions, so the same question asked again costs no LLM tokens.

    The free tier allows 200K tokens a day, and demo visitors tend to ask the same example questions.
    The cache lives in the server's memory (lost on restart) and is cleared whenever the documents
    change through the API, because an answer may then be out of date. Files indexed with the
    command-line scripts do not pass through the API, so an answer also expires after `max_age_s`.
    When full, the least recently used answer goes.
    """

    def __init__(self, max_size: int = 256, max_age_s: float = 3600) -> None:
        self.max_size = max_size
        self.max_age_s = max_age_s
        # key -> (time it was stored, answer)
        self._answers: OrderedDict[CacheKey, tuple[float, AskResponse]] = OrderedDict()
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
            entry = self._answers.get(key)
            if entry is None:
                return None
            stored, answer = entry
            if time.monotonic() - stored > self.max_age_s:
                del self._answers[key]
                return None
            self._answers.move_to_end(key)
            return answer

    def put(self, request: AskRequest, answer: AskResponse) -> None:
        if self.max_size <= 0:
            return
        key = self.key(request)
        with self._lock:
            self._answers[key] = (time.monotonic(), answer)
            self._answers.move_to_end(key)
            if len(self._answers) > self.max_size:
                self._answers.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._answers.clear()

    def __len__(self) -> int:
        return len(self._answers)
