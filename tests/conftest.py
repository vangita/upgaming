import os

# Settings require a Groq key. The tests never call Groq, so where there is no .env (e.g. in CI)
# a placeholder is enough; an environment variable also takes precedence over a real key in .env
os.environ.setdefault("GROQ_API_KEY", "test-key")
