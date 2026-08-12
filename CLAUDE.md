# Working agreements for this repository

## Git: suggest, do not execute

Claude must **never** run `git commit` or `git push`. Staging, committing, and
pushing are the user's decisions and the user's actions.

When work is at a natural stopping point, say so and suggest a commit — a
proposed commit message is welcome — but leave the running of it to the user.
`git status`, `git diff`, `git log` and other read-only inspection are fine.

## Fail fast: no defensive error handling

This is scientific analysis code, not production software. An error that
surfaces immediately as an ugly traceback is *better* than one that is caught,
softened, and allowed to propagate as a plausible-looking number. A wrong result
that looks right is the only truly expensive failure here.

So:

- **No `try`/`except`** around anything unless the exception is genuinely
  expected and handling it is the point. Let exceptions propagate.
- **No fallbacks, defaults, or sentinel returns** that paper over missing or
  malformed data. Do not return an empty list, a NaN, or a zero when the honest
  answer is that something is absent — let the `KeyError`, `FileNotFoundError`,
  or `IndexError` happen.
- **No pre-flight validation** that merely re-raises what the underlying library
  would have raised anyway. Skip the `if not path.exists(): raise ...` and let
  the open fail.
- **No silent coercion** — no `errors="ignore"`, no bare `except Exception`, no
  `.get(key, default)` where a missing key means a real problem.
- Crashing ungracefully is acceptable. Hiding a problem is not.

Validation is worth writing only when it catches something the language would
*not* catch on its own and that would otherwise produce a silently wrong
number — mismatched units, misaligned coordinates, an unnoticed NaN mask. Assert
those loudly; leave everything else alone.

## Style

- Prefer clear, direct code over defensive code.
- Comments should explain *why* — especially anything about the data's quirks
  (partial years, missing variables, unit conventions).
